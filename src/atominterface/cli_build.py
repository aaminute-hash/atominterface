"""``atominterface build`` -- generate interface candidates from bulk structures.

The command is a thin front-end over :class:`~atominterface.builders.base.InterfaceBuilder`:
it loads two bulk structures, picks the requested builder from the registry,
runs the enumeration and writes the surviving candidates plus CSV/JSON reports.

``pymatgen`` is only imported when a structure actually has to be read, so
``atominterface build --help`` keeps working on a slim installation.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import typer

try:  # typer >= 0.27 vendors click (``typer._click``) and no longer depends on it
    from typer._click.core import ParameterSource
except ImportError:  # pragma: no cover - older typer builds on the real click
    from click.core import ParameterSource  # type: ignore[no-redef]

import atominterface.builders  # noqa: F401  (import side effect: registers built-in builders)
from atominterface.builders.candidates import charge_summary
from atominterface.builders.registry import builder_registry

app = typer.Typer(
    help="Build interface candidates from two bulk structures.",
    no_args_is_help=True,
)

_DEFAULT_OUTDIR = Path("outputs/build")
_CHARGE_POLICIES = ("report", "require-neutral", "require-nonpolar")
_CHARGE_OPTIONS = (
    "charge_policy",
    "substrate_oxidation",
    "film_oxidation",
    "charge_tolerance",
    "dipole_tolerance",
)
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


def _report_no_candidates(instance, max_strain: float, max_area: float) -> None:
    """Explain charge rejection or assembly failure before geometry advice."""
    report = getattr(instance, "last_build_report", None)
    if isinstance(report, dict):
        stage = report.get("empty_stage")
        if stage == "charge_filter":
            typer.echo(
                f"\nNo candidate passed the charge/polarity policy {report.get('charge_policy')!r}."
            )
            typer.echo(
                f"Rejected {report.get('rejected', 0)} of {report.get('analyzed', 0)} "
                "analyzed candidate(s); rejection reasons may overlap:"
            )
            for reason, count in sorted(report.get("rejection_reasons", {}).items()):
                typer.echo(f"  {reason}: {count}")
            typer.echo(
                "Review the charge sources and slab geometry; unknown is not neutral/nonpolar."
            )
            return
        if stage == "assembly":
            typer.echo("\nLattice matches were found, but no interface could be assembled.")
            typer.echo(f"Assembly failed for {report.get('assembly_failed', 0)} attempt(s).")
            return
        if stage == "no_matches":
            typer.echo("\nNo lattice matches survived the geometry filters.")

    # Older builders may only offer a preview of matches before strain filtering.
    preview = getattr(instance, "preview_matches", None)
    best = preview(limit=5) if callable(preview) else []

    if not best:
        typer.echo("\nNo lattice match found within the ZSL search window.")
        typer.echo(f"All matches exceed --max-area {max_area:g}; try a larger area,")
        typer.echo("different Miller indices, or a smaller film/substrate cell.")
        return

    typer.echo("\nNo candidate satisfies the current tolerances.")
    typer.echo("\nClosest lattice matches (rejected by the strain cut-off alone):")
    typer.echo(f"{'#':>3}  {'area (A^2)':>10}  {'stretch':>8}  {'shear':>8}")
    for rank, match in enumerate(best, start=1):
        typer.echo(
            f"{rank:>3}  {match.match_area:10.2f}  {match.strain:8.4f}  {match.shear_strain:8.4f}"
        )

    needed = max(max(m.strain, m.shear_strain) for m in best)
    suggested = math.ceil(needed * 1000) / 1000
    typer.echo(
        f"\nBest achievable max(stretch, shear) = {needed:.4f}, "
        f"current --max-strain = {max_strain:g}."
    )
    typer.echo(f"Re-run with --max-strain {suggested:g} to keep these candidates.")


def _write_outputs(
    candidates: list, outdir: Path, fmt: str, build_report: dict | None = None
) -> Path:
    """Write structures, an extended CSV and strict JSON, even for an empty set.

    JSON retains the full charge analysis and geometry snapshot.  It is also
    the authoritative charge record for structure formats without metadata.
    The optional fourth argument preserves existing three-argument callers.
    """
    from ase.io import write as ase_write

    rows: list[dict[str, object]] = []
    json_rows = []
    for rank, candidate in enumerate(candidates, start=1):
        analysis = candidate.metadata.get("charge_analysis")
        row = {
            "rank": rank,
            "interface_id": candidate.interface_id,
            "builder": candidate.builder_name,
            "substrate": candidate.substrate_formula,
            "film": candidate.film_formula,
            "substrate_termination": candidate.substrate_termination,
            "film_termination": candidate.film_termination,
            "strain_substrate": f"{candidate.strain_substrate:.6f}",
            "strain_film": f"{candidate.strain_film:.6f}",
            "area": f"{candidate.area:.4f}",
            "n_atoms": candidate.total_atoms,
            **charge_summary(analysis),
        }
        rows.append(row)
        json_rows.append({**row, "charge_analysis": analysis})

    # Validate JSON before writing any files: unknowns must be null, never NaN/Inf.
    json_outputs = {
        "candidates.json": json.dumps(json_rows, indent=2, ensure_ascii=False, allow_nan=False)
    }
    if build_report is not None:
        json_outputs["build_report.json"] = json.dumps(
            build_report, indent=2, ensure_ascii=False, allow_nan=False
        )

    outdir.mkdir(parents=True, exist_ok=True)
    for rank, candidate in enumerate(candidates, start=1):
        ase_write(outdir / f"candidate_{rank:02d}.{fmt}", candidate.to_atoms())
    csv_path = outdir / "candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*_CSV_FIELDS, *charge_summary(None)])
        writer.writeheader()
        writer.writerows(rows)
    for name, content in json_outputs.items():
        (outdir / name).write_text(content + "\n", encoding="utf-8")
    return csv_path


def _print_candidates(candidates: list) -> None:
    typer.echo(f"\n{len(candidates)} candidate(s) found.")
    typer.echo("Charge/polarity columns: total | substrate | film (saved geometry snapshot).")
    typer.echo(
        f"\n{'#':>3}  {'terms':>7}  {'mismatch':>8}  {'area(A^2)':>10}  {'atoms':>5}  "
        "charge/polarity"
    )
    for rank, candidate in enumerate(candidates, start=1):
        terms = f"{candidate.substrate_termination}/{candidate.film_termination}"
        mismatch = candidate.metadata.get("mismatch", candidate.mean_strain)
        summary = charge_summary(candidate.metadata.get("charge_analysis"))
        states = " | ".join(
            f"{summary[f'{group}_neutrality']}/{summary[f'{group}_polarity']}"
            for group in ("total", "substrate", "film")
        )
        typer.echo(
            f"{rank:>3}  {terms:>7}  {float(mismatch):8.4f}  "
            f"{candidate.area:10.2f}  {candidate.total_atoms:5d}  {states}"
        )


@app.command("build")
def build(
    ctx: typer.Context,
    substrate: Path = typer.Argument(..., help="Substrate bulk structure (CIF/POSCAR)."),
    film: Path = typer.Argument(..., help="Film bulk structure (CIF/POSCAR)."),
    builder: str = typer.Option("coherent", "--builder", "-b", help="Registered builder name."),
    sub_miller: tuple[int, int, int] = typer.Option(
        (0, 0, 1), "--sub-miller", help="Substrate surface Miller indices."
    ),
    film_miller: tuple[int, int, int] = typer.Option(
        (0, 0, 1), "--film-miller", help="Film surface Miller indices."
    ),
    max_strain: float = typer.Option(0.05, "--max-strain", help="Absolute strain tolerance."),
    max_area: float = typer.Option(
        100.0,
        "--max-area",
        help="Maximum interface area in A^2. Cost grows fast with this value.",
    ),
    gap: float = typer.Option(2.0, "--gap", help="Separation gap between slabs in A."),
    max_candidates: int = typer.Option(10, "--max-candidates", help="Keep at most N candidates."),
    charge_policy: str = typer.Option(
        "report",
        "--charge-policy",
        callback=_charge_policy,
        help="Charge filter: report, require-neutral, or require-nonpolar; unknown fails strict filters.",
    ),
    substrate_oxidation: list[str] = typer.Option(
        [], "--substrate-oxidation", help="Repeat Element=value mappings, e.g. Na=1 or Cl=-1."
    ),
    film_oxidation: list[str] = typer.Option(
        [], "--film-oxidation", help="Repeat Element=value mappings, e.g. Na=1 or Cl=-1."
    ),
    charge_tolerance: float = typer.Option(
        1e-6,
        "--charge-tolerance",
        callback=_finite_nonnegative,
        help="Non-negative net-charge tolerance in e.",
    ),
    dipole_tolerance: float = typer.Option(
        1e-3,
        "--dipole-tolerance",
        callback=_finite_nonnegative,
        help="Non-negative normal dipole/area tolerance in e/A.",
    ),
    symmetric: bool | None = typer.Option(
        None,
        "--symmetric/--no-symmetric",
        # Own panel: the paired flag names are wider than the main option
        # column, and rich would otherwise shrink that column until names
        # like --substrate-oxidation truncate on an 80-column terminal.
        rich_help_panel="Symmetric slabs",
        help=(
            "Force mirror-symmetric slabs (same termination on both "
            "surfaces); default only warns about polar terminations."
        ),
    ),
    outdir: Path = typer.Option(_DEFAULT_OUTDIR, "--outdir", "-o", help="Output directory."),
    fmt: str = typer.Option("extxyz", "--format", help="Structure file format for candidates."),
) -> None:
    """Enumerate interface candidates for a substrate/film pair."""
    builder_cls = _resolve_builder(builder)
    _check_charge_support(ctx, builder, symmetric)
    config = _make_config(
        builder,
        substrate_miller=tuple(sub_miller),
        film_miller=tuple(film_miller),
        max_strain=max_strain,
        max_area=max_area,
        gap=gap,
        max_candidates=max_candidates,
        charge_policy=charge_policy,
        substrate_oxidation=_parse_oxidation(substrate_oxidation, "--substrate-oxidation"),
        film_oxidation=_parse_oxidation(film_oxidation, "--film-oxidation"),
        charge_tolerance=charge_tolerance,
        dipole_tolerance=dipole_tolerance,
        symmetric_slabs="auto" if symmetric is None else symmetric,
    )

    sub = _load_structure(substrate)
    film_struct = _load_structure(film)

    typer.echo(f"builder   : {builder}")
    typer.echo(f"substrate : {substrate} ({sub.composition.reduced_formula})")
    typer.echo(f"film      : {film} ({film_struct.composition.reduced_formula})")
    typer.echo(f"surfaces  : {tuple(sub_miller)} | {tuple(film_miller)}")

    instance = builder_cls(substrate=sub, film=film_struct, config=config)
    candidates = instance.build()
    report = getattr(instance, "last_build_report", None)
    csv_path = _write_outputs(candidates, outdir, fmt, report)

    if not candidates:
        _report_no_candidates(instance, max_strain, max_area)
    else:
        _print_candidates(candidates)
    if isinstance(report, dict) and report.get("stopped_at_limit"):
        typer.echo(
            "Enumeration stopped at --max-candidates; the report covers attempted candidates."
        )
    typer.echo(
        f"\nWrote {len(candidates)} structure file(s), {csv_path} and JSON reports to {outdir}/"
    )


if __name__ == "__main__":
    app()
