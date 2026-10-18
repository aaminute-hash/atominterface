"""``atominterface build`` -- generate interface candidates from bulk structures.

The command is a thin front-end over :class:`~atominterface.builders.base.InterfaceBuilder`:
it loads two bulk structures, picks the requested builder from the registry,
runs the enumeration and writes the surviving candidates plus a CSV summary.

``pymatgen`` is only imported when a structure actually has to be read, so
``atominterface build --help`` keeps working on a slim installation.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import typer

import atominterface.builders  # noqa: F401  (import side effect: registers built-in builders)
from atominterface.builders.registry import builder_registry

app = typer.Typer(
    help="Build interface candidates from two bulk structures.",
    no_args_is_help=True,
)

_DEFAULT_OUTDIR = Path("outputs/build")


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
    """Return the builder-specific config, or ``None`` when defaults apply.

    Only the built-in coherent builder is wired up today; unknown builders
    fall back to their own defaults instead of failing the command.
    """
    if builder_name != "coherent":
        return None

    from atominterface.builders.coherent import CoherentInterfaceConfig

    accepted = set(CoherentInterfaceConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in overrides.items() if k in accepted}
    return CoherentInterfaceConfig(**kwargs)


def _report_no_candidates(instance, max_strain: float, max_area: float) -> None:
    """Explain *why* nothing survived instead of stopping at a dead end.

    Builders may expose ``preview_matches()``, which returns the best
    lattice matches ignoring the strain cut-off.  Comparing that number
    with the requested tolerance tells the user exactly which knob to
    turn.
    """
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


def _write_outputs(candidates: list, outdir: Path, fmt: str) -> Path:
    """Write one structure file per candidate plus a CSV summary."""
    from ase.io import write as ase_write

    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for rank, candidate in enumerate(candidates, start=1):
        ase_write(outdir / f"candidate_{rank:02d}.{fmt}", candidate.to_atoms())
        rows.append(
            {
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
            }
        )

    csv_path = outdir / "candidates.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return csv_path


@app.command("build")
def build(
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
    outdir: Path = typer.Option(_DEFAULT_OUTDIR, "--outdir", "-o", help="Output directory."),
    fmt: str = typer.Option("extxyz", "--format", help="Structure file format for candidates."),
) -> None:
    """Enumerate interface candidates for a substrate/film pair."""
    builder_cls = _resolve_builder(builder)

    config = _make_config(
        builder,
        substrate_miller=tuple(sub_miller),
        film_miller=tuple(film_miller),
        max_strain=max_strain,
        max_area=max_area,
        gap=gap,
        max_candidates=max_candidates,
    )

    sub = _load_structure(substrate)
    film_struct = _load_structure(film)

    typer.echo(f"builder   : {builder}")
    typer.echo(f"substrate : {substrate} ({sub.composition.reduced_formula})")
    typer.echo(f"film      : {film} ({film_struct.composition.reduced_formula})")
    typer.echo(f"surfaces  : {tuple(sub_miller)} | {tuple(film_miller)}")

    instance = builder_cls(substrate=sub, film=film_struct, config=config)
    candidates = instance.build()

    if not candidates:
        _report_no_candidates(instance, max_strain, max_area)
        raise typer.Exit(code=0)

    typer.echo(f"\n{len(candidates)} candidate(s) found.")
    typer.echo(f"\n{'#':>3}  {'terms':>7}  {'mismatch':>8}  {'area(A^2)':>10}  {'atoms':>5}")
    for rank, candidate in enumerate(candidates, start=1):
        terms = f"{candidate.substrate_termination}/{candidate.film_termination}"
        mismatch = candidate.metadata.get("mismatch", candidate.mean_strain)
        typer.echo(
            f"{rank:>3}  {terms:>7}  {float(mismatch):8.4f}  "
            f"{candidate.area:10.2f}  {candidate.total_atoms:5d}"
        )

    csv_path = _write_outputs(candidates, outdir, fmt)
    typer.echo(f"\nWrote {len(candidates)} structure file(s) and {csv_path} to {outdir}/")


if __name__ == "__main__":
    app()
