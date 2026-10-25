"""MACE backend adapter (optional dependency)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from .base import Capability
from .registry import backend_registry

#: Canonical download URL of the MACE-MP-0 "medium" foundation model
#: (2 message-passing layers, 128 channels, trained on the Materials
#: Project; ~44 MB checkpoint). The GitHub release asset redirects to the
#: ``mace-foundations`` repository; both endpoints serve the same file.
MACE_MP0_MEDIUM_URL = (
    "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/"
    "2023-12-03-mace-128-L1_epoch-199.model"
)


def mace_cache_path() -> Path:
    """Replicate mace-torch's cache path for the default checkpoint.

    Mirrors ``mace.calculators.download_mace_mp_checkpoint``: the cached
    file name is the URL basename stripped to alphanumerics/underscores,
    placed inside mace-torch's cache directory. Because the same rule is
    used here, a manual download and the official helper resolve to the
    exact same file and never fetch the weights twice.

    Returns:
        The path the MACE-MP-0 medium checkpoint occupies (or would
        occupy) in the local mace-torch cache.
    """
    try:
        from mace.tools.utils import get_cache_dir  # type: ignore[import-untyped]
    except Exception:  # noqa: BLE001
        # mace-torch too old (or broken) to expose the helper: fall back
        # to the XDG cache location mace-torch itself documents.
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        cache_dir = base / "mace"
    else:
        cache_dir = Path(get_cache_dir())
    stem = "".join(c for c in os.path.basename(MACE_MP0_MEDIUM_URL) if c.isalnum() or c in "_")
    return cache_dir / stem


def _download_default_weights() -> Path:
    """Download the default MACE-MP-0 medium checkpoint into the cache.

    The download streams to a ``.part`` sibling and is moved into place
    with :func:`os.replace`, so an interrupted transfer can never leave a
    truncated file that a later run would mistake for a valid checkpoint.

    Returns:
        The path of the cached checkpoint (reused if already present).

    Raises:
        RuntimeError: If the checkpoint cannot be fetched (e.g. offline).
    """
    target = mace_cache_path()
    if target.is_file():
        return target  # already cached by a previous run or the official helper

    partial = target.with_name(target.name + ".part")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(MACE_MP0_MEDIUM_URL, timeout=120) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        os.replace(partial, target)
    except Exception as exc:  # noqa: BLE001
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            "failed to download the default MACE-MP-0 medium weights from "
            f"{MACE_MP0_MEDIUM_URL}; check your network connection, or pass "
            "a local checkpoint via model_path=/model_paths="
        ) from exc
    return target


def _capability() -> Capability:
    return Capability(
        name="mace",
        supports_stress=True,
        supports_uncertainty=False,
        energy_unit="eV",
        force_unit="eV/A",
        stress_unit="eV/A^3",
        license="MIT",  # model weights have their own license
        notes="Wraps mace-torch MACECalculator as an ASE calculator.",
    )


def _instantiate(cls: Any, device: str, **weight_kwargs: Any) -> Any:
    """Instantiate ``cls`` with the first weights kwarg spelling it accepts.

    mace-torch renamed the weights argument across releases
    (``model_path`` -> ``model_paths``); the caller passes the tiers as
    keyword arguments, which dict ordering preserves. A ``TypeError``
    means "this spelling does not exist here", so the next tier is tried
    and an error is raised only once every spelling has been rejected.

    Args:
        cls: ``MACECalculator`` (or a compatible drop-in).
        device: ASE/torch device string forwarded to the calculator.
        **weight_kwargs: Ordered tiers of ``{kwarg_name: value}``.

    Returns:
        A constructed calculator instance.

    Raises:
        RuntimeError: If every weights kwarg spelling was rejected.
    """
    attempts: list[str] = []
    for key, value in weight_kwargs.items():
        try:
            return cls(**{key: value}, device=device)
        except (TypeError, ValueError, OSError) as exc:
            attempts.append(f"{key}: {exc}")
    raise RuntimeError(
        f"could not construct {cls.__name__}; every weights argument was rejected: {attempts}"
    )


def _load_default(cls: Any, device: str) -> Any:
    """Load the default MACE-MP-0 medium foundation model.

    Tier order (first success wins):
      1. ``mace_mp(model="medium")``: official foundation-model helper
         (mace-torch >= 0.3.6) that downloads and caches the checkpoint.
         The model name is passed explicitly because releases >= 0.3.10
         default to the mpa-0 model when it is omitted.
      2. Manual download: fetch the canonical URL into the same cache
         file the helper uses, then construct the calculator from that
         local checkpoint. This covers old mace-torch releases without
         the helper and self-heals a cache the helper cannot read.

    Args:
        cls: ``MACECalculator`` (or a compatible drop-in).
        device: ASE/torch device string forwarded to the calculator.

    Returns:
        A constructed calculator instance bound to the default weights.

    Raises:
        RuntimeError: If the checkpoint cannot be obtained (e.g. offline).
    """
    try:
        from mace.calculators import mace_mp  # type: ignore[import-untyped]
    except Exception:  # noqa: BLE001
        # helper unavailable (older mace-torch): fall through to a manual download
        mace_mp = None  # type: ignore[assignment]
    if mace_mp is not None:
        try:
            return mace_mp(model="medium", device=device)
        except Exception:  # noqa: BLE001
            pass  # helper failed (old signature, unreadable cache, network); retry manually
    local_path = _download_default_weights()
    return _instantiate(
        cls,
        device,
        model_paths=[str(local_path)],
        model_path=str(local_path),
    )


def _factory(**kwargs):  # noqa: D401
    try:
        from mace.calculators import MACECalculator  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "mace-torch is not installed; install it with `pip install mace-torch`"
        ) from exc

    device = str(kwargs.get("device", "cpu"))
    # Optional user override: an explicit checkpoint (local path or URL)
    # wins over the default foundation model. ``model_path`` is the legacy
    # singular spelling and is normalised onto ``model_paths``.
    override = kwargs.get("model_paths", kwargs.get("model_path"))

    def _calc(atoms):  # noqa: A002
        if override is not None:
            return _instantiate(MACECalculator, device, model_paths=override, model_path=override)
        return _load_default(MACECalculator, device)

    return _calc


backend_registry.register("mace", _factory, _capability(), requires="mace")
