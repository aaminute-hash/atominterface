from __future__ import annotations

import csv
import json
import math
from typing import TYPE_CHECKING, Any

import typer

from atominterface._cli_build.params import _CSV_FIELDS, _RELAX_FIELDS
from atominterface._cli_build.relax import _ood_columns
from atominterface.builders.candidates import charge_summary
from atominterface.io import canonicalize_arrays

if TYPE_CHECKING:
    from pathlib import Path


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
            for line in report.get("assembly_detail", []):
                typer.echo(f"  {line}")
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
    candidates: list,
    outdir: Path,
    fmt: str,
    build_report: dict | None = None,
    relax_results: dict[int, Any] | None = None,
) -> Path:
    """Write structures, an extended CSV and strict JSON, even for an empty set.

    JSON retains the full charge analysis and geometry snapshot.  It is also
    the authoritative charge record for structure formats without metadata.
    The optional fourth and fifth arguments (build report, relaxation
    results) preserve existing three-argument callers.
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
            **_ood_columns(relax_results.get(rank) if relax_results else None),
        }
        rows.append(row)
        json_rows.append(
            {**row, "charge_analysis": analysis, "relax": candidate.metadata.get("relax")}
        )

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
        atoms = candidate.to_atoms()
        # pymatgen site properties reach atoms.arrays in set-iteration
        # order, so the extxyz column order (and with it the file bytes)
        # would drift with PYTHONHASHSEED without this normalisation.
        canonicalize_arrays(atoms)
        ase_write(outdir / f"candidate_{rank:02d}.{fmt}", atoms)
    csv_path = outdir / "candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[*_CSV_FIELDS, *charge_summary(None), *_RELAX_FIELDS]
        )
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
