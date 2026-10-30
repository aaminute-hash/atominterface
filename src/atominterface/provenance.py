"""Run provenance: metadata.yaml writer.

Captures the minimal set of facts needed to reproduce a run:
random seed, model/backend identity, environment version, CLI args,
and an artifact hash.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from atominterface import __version__ as _tool_version


@dataclass
class RunMeta:
    """Reproducibility metadata for one AtomInterface run."""

    tool: str = "atominterface"
    # Follows the package version instead of repeating it, so a release bump
    # cannot leave the recorded provenance behind the shipped code.
    version: str = _tool_version
    run_id: str = ""
    seed: int | None = None
    backend: str = "unknown"
    backend_version: str = ""
    model_hash: str = ""
    units: str = "eV_A"
    cli_args: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _short_hash(data: bytes, length: int = 12) -> str:
    return hashlib.sha256(data).hexdigest()[:length]


def _collect_env() -> dict[str, str]:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod in ("ase", "pymatgen", "torch"):
        try:
            m = __import__(mod)
            # str(...) is load-bearing: torch exposes __version__ as a
            # str subclass (TorchVersion), which yaml.safe_dump refuses
            # to represent and which would silently downgrade the whole
            # metadata file to the JSON fallback.
            env[mod] = str(getattr(m, "__version__", "unknown"))
        except Exception:  # noqa: BLE001
            env[mod] = "not-installed"
    return env


def hash_bytes(data: bytes) -> str:
    """Full SHA-256 hex digest of an artifact."""
    return hashlib.sha256(data).hexdigest()


def current_seed() -> int | None:
    """Deterministic fingerprint of the current numpy global RNG state.

    This is **not** a seed that was ever handed to ``np.random.seed``.
    numpy's ``get_state()`` returns ``('MT19937', 624-word array, pos,
    has_gauss, cached_gaussian)`` -- the first element is the algorithm
    name, not a number -- so the historical ``int(state[0])`` always
    raised and was swallowed into ``None``.

    The fingerprint is the first 16 hex digits of the SHA-256 over the
    624-word state array plus the stream position, reusing
    :func:`hash_bytes` so there is exactly one hashing implementation:

        int(hash_bytes(state_words.tobytes() + pos_bytes)[:16], 16)

    It is deterministic (same RNG state -> same integer, different
    states collide with probability ~2^-64) and non-negative.

    .. warning::
       This is a *state* fingerprint for provenance records, **not** a
       seed for restoration: ``np.random.seed(fingerprint)`` does not
       recover the state that produced it (it replays a different
       stream).  Use it to audit "was this run's randomness identical
       to that run's?", nothing else.

    Returns:
        The fingerprint integer, or ``None`` when numpy is unavailable
        or the state is not in the expected MT19937 shape.
    """
    try:
        st = np_state()
        if not st:
            return None
        state = st["state"]
        name, words, pos = state[0], state[1], state[2]
        if name != "MT19937":
            return None
        # The MT19937 words array carries its own ``tobytes``; fall back
        # to ``bytes()`` only for exotic mocks, then degrade to None.
        try:
            words_bytes = words.tobytes()
        except Exception:  # noqa: BLE001
            words_bytes = bytes(words)
        payload = words_bytes + int(pos).to_bytes(8, "little")
        return int(hash_bytes(payload)[:16], 16)
    except Exception:  # noqa: BLE001
        return None


def np_state():  # noqa: D401
    """Return numpy RNG state dict, or None."""
    try:
        import numpy as np

        return {"state": np.random.get_state()}
    except Exception:  # noqa: BLE001
        return None


def build_meta(
    *,
    backend: str = "unknown",
    backend_version: str = "",
    model_hash: str = "",
    seed: int | None = None,
    units: str = "eV_A",
    cli_args: dict[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> RunMeta:
    """Construct a RunMeta with environment auto-collection.

    ``model_hash`` convention: pass the short (12 hex chars) SHA-256 of
    the weights the backend actually used. Obtain it with
    ``atominterface.backends.resolve_model_meta(backend, **backend_kwargs)``
    and read the ``model_hash`` key; backends without verifiable weights
    (e.g. ``dummy``) leave it empty.

    ``seed`` convention: an explicit ``seed`` always wins.  When it is
    omitted (``None``), the fingerprint of numpy's current global RNG
    state (:func:`current_seed`) is recorded instead, so every run's
    metadata.yaml carries the randomness it actually used even when the
    code path never passes a seed.  It stays ``None`` only when numpy is
    unavailable -- see :func:`current_seed` for what the value means.
    """
    import uuid

    from atominterface import __version__

    if seed is None:
        seed = current_seed()

    return RunMeta(
        tool="atominterface",
        version=__version__,
        run_id=uuid.uuid4().hex[:10],
        seed=seed,
        backend=backend,
        backend_version=backend_version,
        model_hash=model_hash,
        units=units,
        cli_args=cli_args or {},
        artifacts=artifacts or {},
        env=_collect_env(),
        extra=extra or {},
    )


def write_metadata(meta: RunMeta, path: str | Path = "metadata.yaml") -> Path:
    """Write metadata as YAML if available, else JSON."""
    p = Path(path)
    data = meta.to_dict()
    try:
        import yaml  # type: ignore[import-untyped]

        text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
        suffix = ".yaml"
    except Exception:  # noqa: BLE001
        text = json.dumps(data, indent=2, sort_keys=False)
        suffix = ".json"
    if p.suffix not in (".yaml", ".yml", ".json"):
        p = p.with_suffix(suffix)
    p.write_text(text, encoding="utf-8")
    return p
