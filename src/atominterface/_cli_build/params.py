from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import typer

try:  # typer >= 0.27 vendors click (typer._click) and no longer depends on it
    from typer._click.core import ParameterSource
except ImportError:  # pragma: no cover - older typer builds on the real click
    from click.core import ParameterSource  # type: ignore[no-redef]

from atominterface.backends.registry import backend_registry
from atominterface.builders.coherent import REGISTRY_MODES, CoherentInterfaceConfig
from atominterface.builders.registry import builder_registry
from atominterface.run_config import load_run_config

_DEFAULT_OUTDIR = Path("outputs/build")
#: Single source of truth for build-geometry defaults: the CLI advertises
#: exactly what ``CoherentInterfaceConfig`` falls back to, so a recorded
#: run never depends on a second, driftable copy of the same numbers.
_COHERENT_DEFAULTS = CoherentInterfaceConfig()
#: Keys accepted in a ``--config`` YAML file; anything else is rejected so
#: a typo can never be silently ignored during a replay.
_RUN_CONFIG_KEYS = frozenset(
    {
        "builder",
        "sub_miller",
        "film_miller",
        "max_strain",
        "max_area",
        "gap",
        "min_thickness",
        "min_thickness_substrate",
        "min_thickness_film",
        "vacuum",
        "max_candidates",
        "charge_policy",
        "substrate_oxidation",
        "film_oxidation",
        "charge_tolerance",
        "dipole_tolerance",
        "symmetric",
        "registry",
        "max_registry_sites",
        "relax",
        "backend",
        "committee_backend",
        "device",
        "fmax",
        "max_steps",
        "ood_threshold",
    }
)
_CHARGE_POLICIES = ("report", "require-neutral", "require-nonpolar")
_CHARGE_OPTIONS = (
    "charge_policy",
    "substrate_oxidation",
    "film_oxidation",
    "charge_tolerance",
    "dipole_tolerance",
)
_DEVICES = ("cpu", "cuda")
_CSV_FIELDS = (
    "rank",
    "interface_id",
    "builder",
    "substrate",
    "film",
    "substrate_termination",
    "film_termination",
    "strain_substrate",
    "strain_film",
    "area",
    "n_atoms",
)
#: Relaxation validity and OOD bookkeeping columns, appended after the
#: charge summary so the historical column order stays untouched.
_RELAX_FIELDS = (
    "valid",
    "converged",
    "energies_finite",
    "forces_finite",
    "exception",
    "committee_enabled",
    "ood_monitoring",
    "ood_score",
    "ood_step",
    "ood_triggered",
)


def _load_structure(path: Path):
    """Read one bulk structure with pymatgen, with actionable error messages."""
    try:
        from pymatgen.core import Structure
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise typer.BadParameter(
            "pymatgen is required to read structures: pip install 'atominterface[pymatgen]'",
            param_hint="substrate / film",
        ) from exc

    if not path.exists():
        raise typer.BadParameter(f"file not found: {path}", param_hint=str(path))
    return Structure.from_file(str(path))


def _resolve_builder(name: str):
    """Look up a builder class, listing the known names when the lookup fails."""
    try:
        return builder_registry.get(name)
    except KeyError:
        available = ", ".join(sorted(builder_registry.list())) or "<none>"
        raise typer.BadParameter(
            f"unknown builder {name!r}; available: {available}",
            param_hint="--builder",
        ) from None


def _resolve_backend(name: str) -> str:
    """Validate a backend name, listing the known names when the lookup fails.

    ``autodiscover`` only imports the optional backend *registration*
    modules, whose module level is dependency-light; the heavy calculator
    (torch, model weights) is constructed exclusively by ``--relax``.
    """
    from atominterface.backends import autodiscover

    autodiscover()
    if name not in backend_registry.list():
        available = ", ".join(sorted(backend_registry.list())) or "<none>"
        raise typer.BadParameter(
            f"unknown backend {name!r}; available: {available}",
            param_hint="--backend",
        )
    return name


def _validate_device(value: str) -> str:
    """Accept only cpu/cuda, and verify cuda is usable before promising it."""
    if value not in _DEVICES:
        raise typer.BadParameter("must be one of: " + ", ".join(_DEVICES))
    if value != "cuda":
        return value
    try:
        import torch
    except ImportError:
        # Without torch every backend runs on cpu anyway; honour the run
        # with a note instead of failing a request that cannot be met.
        typer.echo("note: torch is not installed; running on cpu instead of cuda.")
        return "cpu"
    if not torch.cuda.is_available():
        raise typer.BadParameter(
            "cuda requested but no CUDA-capable device is available; use --device cpu"
        )
    return value


def _make_config(builder_name: str, **overrides):
    """Return the coherent config, or other builders' defaults.

    Unsupported explicit charge options are rejected before this is called;
    the historical defaults-only behavior remains for other builders.
    """
    if builder_name != "coherent":
        return None

    from atominterface.builders.coherent import CoherentInterfaceConfig

    accepted = set(CoherentInterfaceConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in overrides.items() if k in accepted}
    return CoherentInterfaceConfig(**kwargs)


def _charge_policy(value: str) -> str:
    if value not in _CHARGE_POLICIES:
        raise typer.BadParameter("must be one of: " + ", ".join(_CHARGE_POLICIES))
    return value


def _finite_nonnegative(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise typer.BadParameter("must be finite and non-negative")
    return value


def _finite_positive(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise typer.BadParameter("must be finite and positive")
    return value


def _parse_backend_list(values: list[str] | str | None) -> list[str]:
    """Normalize repeated flags and comma-separated spellings to one flat list.

    ``--committee-backend mace,chgnet`` and the repeated
    ``--committee-backend mace --committee-backend chgnet`` describe the
    same committee; config-file values arrive as whatever YAML holds
    (a list, or a single comma-separated string), so both shapes are
    accepted here.  Empty entries are dropped, which makes an explicitly
    empty value mean "no committee" rather than an unknown backend.
    """
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    names: list[str] = []
    for value in values:
        names.extend(part.strip() for part in str(value).split(",") if part.strip())
    return names


def _optional_finite_positive(value: float | None) -> float | None:
    """Per-side thickness overrides: ``None`` means "fall back to the shared value"."""
    if value is None:
        return None
    return _finite_positive(value)


def _optional_int_positive(value: int | None) -> int | None:
    """Caps ``registry="scan"`` site enumeration; ``None`` means "all"."""
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise typer.BadParameter("must be a positive integer or omitted")
    return value


def _registry(value: str) -> str | tuple[float, float] | None:
    """Parse a requested lateral registry: a mode name, ``u,v``, or ``none``.

    Site names (``trough``, ``ridge``, ``fcc-hollow``) are deliberately
    absent from the vocabulary: which name means which fractional offset
    is orientation- and lattice-specific, so a name that cannot be
    resolved is answered with the list of what can rather than guessed
    at. ``u,v`` is always available, so nothing is out of reach.

    ``none`` is the spelling of ``registry=None`` -- the pre-0.5.0
    behaviour, in which the join is left wherever the two slab cuts put
    it.  It exists so a number quoted from an older release can be
    reproduced on purpose, which is the only way to check that the change
    this field made is the change it claims to be.
    """
    text = value.strip()
    if text == "none":
        return None
    if "," in text:
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 2:
            raise typer.BadParameter(
                "expected two comma-separated fractional components, for example "
                f"0.5,0.5; got {value!r}"
            )
        try:
            parsed = (float(parts[0]), float(parts[1]))
        except ValueError:
            raise typer.BadParameter(
                f"registry components must be numbers; got {value!r}"
            ) from None
        if not all(math.isfinite(item) for item in parsed):
            raise typer.BadParameter(f"registry components must be finite; got {value!r}")
        return parsed
    if text not in REGISTRY_MODES and text != "scan":
        raise typer.BadParameter(
            "must be one of: " + ", ".join(REGISTRY_MODES) + ", none (or u,v, e.g. 0.5,0.5)"
        )
    return text


def _merged_registry(value: object) -> str | tuple[float, float] | None:
    """Re-validate a registry value merged from a config file or the command line.

    A recorded run replays its registry as whatever YAML holds -- a list
    for ``u,v``, a string for a mode -- so both shapes are accepted and
    put through the same validator the command line would have used.
    """
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise typer.BadParameter(
                f"registry (u, v) must have exactly 2 components, got {len(value)}: {value!r}"
            )
        try:
            parsed = (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            raise typer.BadParameter(
                f"registry (u, v) must be a pair of finite floats; got {value!r}"
            ) from None
        if not all(math.isfinite(item) for item in parsed):
            raise typer.BadParameter(f"registry (u, v) must be finite floats; got {value!r}")
        return parsed
    return _registry(str(value))


def _parse_oxidation(values: list[str], option: str) -> dict[str, float] | None:
    """Parse repeated Element=charge options without relying on pymatgen."""
    from ase.data import atomic_numbers

    result: dict[str, float] = {}
    for item in values:
        symbol, separator, text = item.partition("=")
        symbol, text = symbol.strip(), text.strip()
        if not separator or not symbol or not text:
            raise typer.BadParameter(
                f"expected Element=value, for example Na=1 or Cl=-1; got {item!r}",
                param_hint=option,
            )
        if symbol not in atomic_numbers or atomic_numbers[symbol] == 0:
            raise typer.BadParameter(f"unknown element symbol {symbol!r}", param_hint=option)
        if symbol in result:
            raise typer.BadParameter(f"duplicate element {symbol!r}", param_hint=option)
        try:
            value = float(text)
        except ValueError:
            raise typer.BadParameter(
                f"oxidation state for {symbol} must be a finite number; got {text!r}",
                param_hint=option,
            ) from None
        if not math.isfinite(value):
            raise typer.BadParameter(
                f"oxidation state for {symbol} must be finite", param_hint=option
            )
        result[symbol] = value
    return result or None


def _check_charge_support(ctx: typer.Context, builder_name: str, symmetric: bool | None) -> None:
    if builder_name == "coherent":
        return
    requested = [
        "--" + name.replace("_", "-")
        for name in _CHARGE_OPTIONS
        if ctx.get_parameter_source(name) not in (None, ParameterSource.DEFAULT)
    ]
    if symmetric is not None:
        requested.append("--symmetric/--no-symmetric")
    if requested:
        raise typer.BadParameter(
            f"builder {builder_name!r} does not support charge options: {', '.join(requested)}",
            param_hint="--builder",
        )


def _load_config_overrides(config_file: Path | None) -> dict[str, Any]:
    """Load a ``--config`` YAML file, rejecting keys the run never reads.

    An unknown key in a replayed configuration is almost always a typo
    (``min_thikness``) or a stale schema; silently dropping it would make
    the replay differ from the recorded run without any sign.
    """
    if config_file is None:
        return {}
    try:
        overrides = load_run_config(config_file)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from None
    unknown = sorted(set(overrides) - _RUN_CONFIG_KEYS)
    if unknown:
        raise typer.BadParameter(
            f"unknown run configuration key(s): {', '.join(unknown)}; "
            f"supported: {', '.join(sorted(_RUN_CONFIG_KEYS))}",
            param_hint="--config",
        )
    return overrides


def _resolve_run_params(
    ctx: typer.Context, overrides: dict[str, Any], cli_values: dict[str, Any]
) -> dict[str, Any]:
    """Merge a ``--config`` file with explicit CLI flags into effective parameters.

    A config-file value applies only where the corresponding flag was left
    at its default; an explicit command-line flag always wins.  Values
    sourced from the file bypass Typer's option callbacks, so every merged
    value is re-validated here with the very same validator the command
    line would have applied.  The returned mapping is exactly what the run
    records to ``run_config.yaml``, which keeps the replay file and the
    effective configuration in lockstep by construction.
    """

    def _pick(name: str):
        if ctx.get_parameter_source(name) not in (None, ParameterSource.DEFAULT):
            return cli_values[name]
        return overrides.get(name, cli_values[name])

    symmetric = _pick("symmetric")
    if symmetric is not None and not isinstance(symmetric, bool):
        raise typer.BadParameter(
            "symmetric must be true, false or null", param_hint="--symmetric/--no-symmetric"
        )
    return {
        "builder": _pick("builder"),
        "sub_miller": list(_pick("sub_miller")),
        "film_miller": list(_pick("film_miller")),
        "max_strain": float(_pick("max_strain")),
        "max_area": float(_pick("max_area")),
        "gap": _finite_nonnegative(float(_pick("gap"))),
        "min_thickness": _finite_nonnegative(float(_pick("min_thickness"))),
        "min_thickness_substrate": _optional_finite_positive(_pick("min_thickness_substrate")),
        "min_thickness_film": _optional_finite_positive(_pick("min_thickness_film")),
        "vacuum": _finite_nonnegative(float(_pick("vacuum"))),
        "max_candidates": int(_pick("max_candidates")),
        "backend": _pick("backend"),
        "committee_backend": _parse_backend_list(_pick("committee_backend")),
        "device": _validate_device(_pick("device")),
        "relax": bool(_pick("relax")),
        "fmax": _finite_nonnegative(float(_pick("fmax"))),
        "max_steps": int(_pick("max_steps")),
        "ood_threshold": _finite_nonnegative(float(_pick("ood_threshold"))),
        "charge_policy": _charge_policy(_pick("charge_policy")),
        "substrate_oxidation": list(_pick("substrate_oxidation")),
        "film_oxidation": list(_pick("film_oxidation")),
        "charge_tolerance": _finite_nonnegative(float(_pick("charge_tolerance"))),
        "dipole_tolerance": _finite_nonnegative(float(_pick("dipole_tolerance"))),
        "symmetric": symmetric,
        "registry": _merged_registry(_pick("registry")),
        "max_registry_sites": _optional_int_positive(_pick("max_registry_sites")),
    }
