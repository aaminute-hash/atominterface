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


@dataclass
class RunMeta:
    """Reproducibility metadata for one AtomInterface run."""

    tool: str = "atominterface"
    version: str = "0.1.0"
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
    """Best-effort current global RNG seed (numpy)."""
    try:
        st = np_state()
        return int(st["state"][0]) if st else None  # type: ignore[attr-defined]
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
    """
    import uuid

    from atominterface import __version__

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
