"""Model identity resolution for provenance metadata.

Turns a backend name into the facts needed to reproduce a run: which
package version provides the backend, which pretrained weights it loads,
and a short content hash of those weights. The resulting metadata feeds
``atominterface.provenance.build_meta(model_hash=...)``.

Hashing strategy per backend:

* ``mace``: hash the bytes of the checkpoint file that will be loaded --
  the explicit user override if given, otherwise the cached default
  MACE-MP-0 medium download. The cache location is resolved by
  ``mace_.mace_cache_path``, which replicates mace-torch's own cache
  rule so loader and recorder always agree on the file. When the
  weights cannot be located on disk (not downloaded yet), the URL
  string itself is hashed and ``model_source`` is ``"url"`` so the
  record still identifies the exact artifact a future run would fetch.
* ``chgnet``: hash the loaded model's ``state_dict`` deterministically
  (parameters sorted by name, each hashed with its dtype and shape), so
  the digest depends only on the weights themselves.
* ``dummy`` / unknown backends: ``model_hash`` is empty -- there are no
  verifiable weights.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any

from .mace_ import MACE_MP0_MEDIUM_URL, mace_cache_path

#: Model name recorded for the MACE backend (foundation model + variant).
MACE_MODEL_NAME = "MACE-MP-0/medium"

#: Model name recorded for the CHGNet backend (bundled pretrained weights).
CHGNET_MODEL_NAME = "CHGNet/pretrained"

#: Model name recorded for the zero-dependency dummy backend.
DUMMY_MODEL_NAME = "dummy/reference"

#: Number of leading hex characters kept in every hash.
HASH_LENGTH = 12

#: Distribution name per backend, used in the identity string.
_BACKEND_PACKAGE = {"mace": "mace-torch", "chgnet": "chgnet"}


def _meta(
    backend: str,
    backend_version: str,
    model_name: str,
    model_hash: str,
    model_source: str,
) -> dict[str, str]:
    """Assemble a model metadata record."""
    return {
        "backend": backend,
        "backend_version": backend_version,
        "model_name": model_name,
        "model_hash": model_hash,
        "model_source": model_source,
    }


def _package_version(module_name: str) -> str:
    """Version of an importable package, or ``"unknown"``."""
    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001
        return "unknown"
    return str(getattr(module, "__version__", "unknown"))


def _short_sha256(data: bytes) -> str:
    """Leading ``HASH_LENGTH`` hex characters of the SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()[:HASH_LENGTH]


def _file_sha256(path: Path) -> str:
    """Short SHA-256 of a file's bytes, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:HASH_LENGTH]


def _state_dict_sha256(model: Any) -> str:
    """Deterministic short hash of a torch module's parameters.

    Parameters are hashed in sorted-name order together with their dtype
    and shape, so the digest depends only on the weights themselves --
    not on ``state_dict`` iteration order or the device they live on.
    """
    state = model.state_dict()
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:HASH_LENGTH]


def _existing_paths(value: Any) -> list[Path]:
    """Expand a weights kwarg value into the existing local files it names.

    Accepts a single path or a sequence of paths; URLs and non-existent
    paths expand to nothing.
    """
    if value is None:
        return []
    candidates = list(value) if isinstance(value, (list, tuple)) else [value]
    paths: list[Path] = []
    for candidate in candidates:
        text = str(candidate)
        if text.startswith(("http://", "https://")):
            continue
        path = Path(text)
        if path.is_file():
            paths.append(path)
    return paths


def _resolve_mace_meta(**kwargs: Any) -> dict[str, str]:
    """Resolve MACE model metadata (see module docstring for the strategy)."""
    version = _package_version("mace")

    # Explicit user weights (local path or URL) win over the default.
    for key in ("model_paths", "model_path"):
        value = kwargs.get(key)
        if value is None:
            continue
        paths = _existing_paths(value)
        if paths:
            return _meta("mace", version, MACE_MODEL_NAME, _file_sha256(paths[0]), str(paths[0]))
        text = str(value)
        source = "url" if text.startswith(("http://", "https://")) else "unresolved"
        return _meta("mace", version, MACE_MODEL_NAME, _short_sha256(text.encode()), source)

    # Default weights: hash the cached download when present.
    cache = mace_cache_path()
    if cache.is_file():
        return _meta("mace", version, MACE_MODEL_NAME, _file_sha256(cache), MACE_MP0_MEDIUM_URL)

    # Weights not on disk yet: hash the URL a run would fetch.
    return _meta(
        "mace",
        version,
        MACE_MODEL_NAME,
        _short_sha256(MACE_MP0_MEDIUM_URL.encode()),
        "url",
    )


def _resolve_chgnet_meta(**kwargs: Any) -> dict[str, str]:
    """Resolve CHGNet model metadata (see module docstring for the strategy)."""
    version = _package_version("chgnet")
    try:
        from chgnet.model.model import CHGNet  # type: ignore[import-untyped]

        model = CHGNet.load()
        digest = _state_dict_sha256(model)
    except Exception:  # noqa: BLE001
        # chgnet missing or unloadable: keep provenance writable, but do
        # not pretend the weights were verified.
        return _meta("chgnet", version, CHGNET_MODEL_NAME, "", "unavailable")
    return _meta("chgnet", version, CHGNET_MODEL_NAME, digest, "state_dict")


def resolve_model_meta(name: str, **kwargs: Any) -> dict[str, str]:
    """Resolve model identity metadata for a backend.

    Args:
        name: Backend name, e.g. ``"mace"``, ``"chgnet"`` or ``"dummy"``.
        **kwargs: Backend construction kwargs, honoured where they select
            weights (``model_path``/``model_paths`` for mace).

    Returns:
        A dict with keys ``backend``, ``backend_version``,
        ``model_name``, ``model_hash`` and ``model_source``. Unknown or
        weight-less backends report an empty ``model_hash``.
    """
    if name == "mace":
        return _resolve_mace_meta(**kwargs)
    if name == "chgnet":
        return _resolve_chgnet_meta(**kwargs)
    if name == "dummy":
        return _meta(name, "unknown", DUMMY_MODEL_NAME, "", "")
    return _meta(name, "unknown", f"{name}/unknown", "", "")


def model_identity_for_meta(name: str, **kwargs: Any) -> str:
    """One-line identity string for a backend.

    Returns:
        A composite such as
        ``"MACE-MP-0/medium@mace-torch:0.3.16:sha256:abcdef123456"``
        (``model_name@package:version:sha256:hash``).
    """
    meta = resolve_model_meta(name, **kwargs)
    package = _BACKEND_PACKAGE.get(name, name)
    return f"{meta['model_name']}@{package}:{meta['backend_version']}:sha256:{meta['model_hash']}"
