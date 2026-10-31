"""MACE backend adapter (optional dependency)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from atominterface.provenance import hash_bytes

from .base import Capability
from .registry import backend_registry

#: Torch dtypes the MACE backend accepts. ``"float32"`` is the default
#: (matching mace-torch's behaviour when no dtype is requested);
#: ``"float64"`` enables double-precision reference runs used to
#: separate physics from float32 rounding in validation studies.
SUPPORTED_DTYPES = ("float32", "float64")


def _validate_dtype(dtype: str) -> str:
    """Return ``dtype`` unchanged, or raise if it is not supported."""
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}; supported values: {list(SUPPORTED_DTYPES)}")
    return dtype


#: Canonical download URL of the MACE-MP-0 "medium" foundation model
#: (2 message-passing layers, 128 channels, trained on the Materials
#: Project; ~44 MB checkpoint). The GitHub release asset redirects to the
#: ``mace-foundations`` repository; both endpoints serve the same file.
MACE_MP0_MEDIUM_URL = (
    "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/"
    "2023-12-03-mace-128-L1_epoch-199.model"
)

#: Expected full SHA-256 (64 hex chars) of the default MACE-MP-0 medium
#: checkpoint -- the integrity anchor checked after every *new* download.
#:
#: The value must only ever come from running
#: ``atominterface.provenance.hash_bytes`` over a genuine download of
#: ``MACE_MP0_MEDIUM_URL`` -- it is never hand-entered from memory or a
#: third-party claim.
#:
#: Recorded 2026-10-31 from a 44,422,970-byte download of the URL above,
#: fetched as six HTTP range requests and reassembled.  Two independent
#: hashers -- ``sha256sum`` and :func:`atominterface.provenance.hash_bytes`
#: -- produced the identical digest.  The file was then confirmed to be a
#: loadable ``mace.modules.models.ScaleShiftMACE`` that yields
#: -4.0844 eV/atom and zero force for ideal fcc Cu at a = 3.615 A, so the
#: anchor describes the real checkpoint rather than merely a self-
#: consistent byte string.
MACE_MP0_MEDIUM_SHA256: str | None = (
    "01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845"
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
    except ImportError:
        # Older mace-torch releases do not expose the helper. Fall back
        # only for that known compatibility case; a broken installation
        # must remain visible to the caller.
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
    After the move the file is verified against
    :data:`MACE_MP0_MEDIUM_SHA256`; a mismatch deletes the freshly
    downloaded file and raises -- a corrupt or substituted checkpoint is
    never silently used.

    Two deliberate limits of the verification:

    * It guards **only this download**.  A cache hit (a file a previous
      run -- or the official mace helper -- put there) is returned as is:
      adding a check there would suddenly break working setups whose
      cached copy predates the anchor.
    * When the anchor is ``None`` (no authoritative hash recorded yet)
      the check is skipped, so offline or unknown environments keep the
      pre-verification behaviour instead of gaining a new failure mode.

    Returns:
        The path of the cached checkpoint (reused if already present).

    Raises:
        RuntimeError: If the checkpoint cannot be fetched (e.g. offline),
            or if the freshly downloaded bytes fail the SHA-256 check.
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

    anchor = MACE_MP0_MEDIUM_SHA256
    if anchor is not None:
        actual = hash_bytes(target.read_bytes())
        if actual != anchor:
            target.unlink(missing_ok=True)
            raise RuntimeError(
                "the downloaded default MACE-MP-0 medium weights failed the "
                f"SHA-256 integrity check (expected {anchor}, got {actual}); "
                f"the file has been deleted from {target}. Delete any local "
                "cache copy and retry, or pass a local checkpoint via "
                "model_path=/model_paths="
            )
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


def _instantiate(cls: Any, device: str, dtype: str = "float32", **weight_kwargs: Any) -> Any:
    """Instantiate ``cls`` with the first weights kwarg spelling it accepts.

    mace-torch renamed the weights argument across releases
    (``model_path`` -> ``model_paths``); the caller passes the tiers as
    keyword arguments, which dict ordering preserves. Every constructor
    failure is retained in the final diagnostic; this includes signature
    mismatches as well as checkpoint, device and runtime failures. The
    final exception is chained from the last attempted constructor.

    Args:
        cls: ``MACECalculator`` (or a compatible drop-in).
        device: ASE/torch device string forwarded to the calculator.
        dtype: torch dtype forwarded as ``default_dtype``; ``"float32"``
            keeps the historical behaviour, ``"float64"`` selects
            double precision for reference runs.
        **weight_kwargs: Ordered tiers of ``{kwarg_name: value}``.

    Returns:
        A constructed calculator instance.

    Raises:
        ValueError: If ``dtype`` is not in :data:`SUPPORTED_DTYPES`.
        RuntimeError: If every weights kwarg spelling was rejected.
    """
    _validate_dtype(dtype)
    attempts: list[str] = []
    last_error: Exception | None = None
    for key, value in weight_kwargs.items():
        try:
            return cls(**{key: value}, device=device, default_dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"{key}: {type(exc).__name__}: {exc}")
            last_error = exc
    raise RuntimeError(
        f"could not construct {cls.__name__}; every weights argument failed: {attempts}"
    ) from last_error


def _load_default(cls: Any, device: str, dtype: str = "float32") -> Any:
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
        dtype: torch dtype, forwarded to both tiers as ``mace_mp``'s
            ``default_dtype`` (the helper's own spelling -- passing
            ``dtype=`` there is absorbed by its ``**kwargs`` and silently
            ignored); ``"float32"`` keeps the historical behaviour,
            ``"float64"`` selects double precision for reference runs.

    Returns:
        A constructed calculator instance bound to the default weights.

    Raises:
        ValueError: If ``dtype`` is not in :data:`SUPPORTED_DTYPES`.
        RuntimeError: If the checkpoint cannot be obtained (e.g. offline).
    """
    _validate_dtype(dtype)
    attempts: list[str] = []
    try:
        from mace.calculators import mace_mp  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        attempts.append(f"mace_mp import: {type(exc).__name__}: {exc}")
        mace_mp = None  # type: ignore[assignment]
    if mace_mp is not None:
        try:
            # ``mace_mp``'s precision knob is spelled ``default_dtype``;
            # passing ``dtype=`` there lands in the helper's ``**kwargs``
            # and is handed to ``MACECalculator``, which reads its own
            # ``default_dtype`` (still "float32") and silently builds a
            # float32 model.  No error, no warning -- the run just is not
            # the precision it was asked for.
            return mace_mp(model="medium", device=device, default_dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"mace_mp helper: {type(exc).__name__}: {exc}")
    try:
        local_path = _download_default_weights()
        return _instantiate(
            cls,
            device,
            dtype,
            model_paths=[str(local_path)],
            model_path=str(local_path),
        )
    except Exception as exc:  # noqa: BLE001
        attempts.append(f"manual checkpoint: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            f"failed to load the default MACE-MP-0 medium model; all attempts failed: {attempts}"
        ) from exc


def _factory(**kwargs):  # noqa: D401
    device = str(kwargs.get("device", "cpu"))
    # Validated before the optional-dependency import so a misspelled
    # dtype raises ValueError even where mace-torch is absent.
    dtype = _validate_dtype(str(kwargs.get("dtype", "float32")))
    try:
        from mace.calculators import MACECalculator  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "mace-torch is not installed; install it with `pip install mace-torch`"
        ) from exc

    # Optional user override: an explicit checkpoint (local path or URL)
    # wins over the default foundation model. ``model_path`` is the legacy
    # singular spelling and is normalised onto ``model_paths``.
    override = kwargs.get("model_paths", kwargs.get("model_path"))

    def _calc(atoms):  # noqa: A002
        if override is not None:
            return _instantiate(
                MACECalculator, device, dtype, model_paths=override, model_path=override
            )
        return _load_default(MACECalculator, device, dtype)

    return _calc


backend_registry.register("mace", _factory, _capability(), requires="mace")
