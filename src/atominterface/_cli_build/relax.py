from __future__ import annotations

from typing import TYPE_CHECKING, Any

import typer

from atominterface._cli_build.params import _RELAX_FIELDS
from atominterface.builders.coherent import measure_registry

if TYPE_CHECKING:
    from pathlib import Path


def _ood_columns(result: Any) -> dict[str, str]:
    """Relaxation-validity and OOD cells for one candidate."""
    if result is None:
        return dict.fromkeys(_RELAX_FIELDS, "")
    return {
        "valid": str(result.valid),
        "converged": str(result.converged),
        "energies_finite": str(result.energies_finite),
        "forces_finite": str(result.forces_finite),
        "exception": result.exception or "",
        "committee_enabled": str(bool(result.metadata.get("committee_enabled", False))),
        "ood_monitoring": str(result.metadata.get("ood_monitoring", "unknown")),
        "ood_score": f"{result.max_ood_score:.6f}",
        "ood_step": str(result.ood_step),
        "ood_triggered": str(result.ood_triggered),
    }


def _relax_candidates(
    candidates: list,
    *,
    backend: str,
    committee_backends: list[str],
    device: str,
    fmax: float,
    max_steps: int,
    ood_threshold: float,
    outdir: Path,
) -> dict[int, Any]:
    """Guarded-relax every candidate in place with the selected backend.

    The relaxed structure replaces ``candidate.atoms`` so the written
    ``candidate_NN`` files (with their calculator results) describe the
    relaxed geometry, and the OOD bookkeeping is stashed in
    ``candidate.metadata["relax"]`` for the JSON report.

    ``committee_backends`` names extra models whose force disagreement
    forms the OOD committee: the primary backend drives the relaxation
    and every listed backend is mounted alongside it through
    :meth:`GuardedRelaxer.from_calculators`.  An empty list keeps the
    historical single-model construction untouched.

    Returns:
        A ``{rank: RelaxResult}`` map keyed like the written candidates.
    """
    from atominterface.backends import get_calculator
    from atominterface.relax import GuardedRelaxer

    typer.echo(
        f"\nRelaxing {len(candidates)} candidate(s) with backend={backend} device={device} "
        f"(fmax={fmax:g}, max_steps={max_steps}, ood_threshold={ood_threshold:g})."
    )
    primary = get_calculator(backend, device=device)
    relaxer_kwargs: dict[str, Any] = dict(
        fmax=fmax,
        max_steps=max_steps,
        ood_threshold=ood_threshold,
        quarantine_dir=str(outdir / "flagged"),
    )
    # Committee members are shared across candidates, so each factory is
    # resolved exactly once (a heavy model must not reload per candidate).
    # They are handed over as instances because CommitteeWatcher calls
    # ``get_forces(atoms)`` on them directly.
    committee = (
        [get_calculator(name, device=device)(candidates[0].atoms) for name in committee_backends]
        if committee_backends and candidates
        else []
    )
    if committee:
        relaxer = GuardedRelaxer.from_calculators([primary, *committee], **relaxer_kwargs)
        typer.echo(f"  committee backend(s): {', '.join(committee_backends)}")
    else:
        relaxer = GuardedRelaxer(calculator=primary, **relaxer_kwargs)
    results: dict[int, Any] = {}
    for rank, candidate in enumerate(candidates, start=1):
        result = relaxer.run(candidate.atoms)
        candidate.atoms = result.atoms
        candidate.metadata["relax"] = {
            "backend": backend,
            "device": device,
            "n_steps": result.n_steps,
            "valid": result.valid,
            "converged": result.converged,
            "energies_finite": result.energies_finite,
            "forces_finite": result.forces_finite,
            "final_energy": result.final_energy,
            "exception": result.exception,
            "committee_enabled": bool(result.metadata.get("committee_enabled", False)),
            "ood_monitoring": result.metadata.get("ood_monitoring", "unknown"),
            "ood_triggered": result.ood_triggered,
            "ood_step": result.ood_step,
            "max_ood_score": result.max_ood_score,
            "reasons": result.reasons,
        }
        # Where the join ended up, measured after the relaxation rather
        # than recalled from the request: the archive holds a (110) run
        # that slid from the ridge to the trough on its own, so the site
        # a relaxation stopped on is not the site it was given. Written
        # beside the relax record and not into it -- that dictionary is
        # compared as a whole downstream.
        measured = measure_registry(candidate.atoms)
        candidate.metadata["registry_final_frac"] = None if measured is None else measured["frac"]
        candidate.metadata["registry_final_lateral_A"] = (
            None if measured is None else measured["lateral"]
        )
        results[rank] = result
        if result.valid:
            status = "valid/converged"
        elif result.ood_triggered:
            status = "invalid (OOD-stopped)"
        elif result.exception is not None:
            status = "invalid (calculator exception)"
        else:
            status = "invalid (not converged)"
        typer.echo(
            f"  candidate {rank:02d}: {status}, cycles={result.n_steps}, "
            f"max_ood={result.max_ood_score:.4f}, ood_step={result.ood_step}"
        )
    return results
