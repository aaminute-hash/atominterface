#!/usr/bin/env python
"""AtomInterface guard as a drop-in ASE Calculator/Optimizer component.

This example exists for one reader: an upstream integrator (an ASE or
pymatgen maintainer evaluating a PR) who wants to see, in a handful of
calls, how AtomInterface's relaxation guard plugs into the ordinary ASE
ecosystem instead of replacing it.  It is discussion material for that
upstream integration, not a benchmark.

Two things are demonstrated, both weight-free by default:

1. The watchers are plain, stateless callables ``(atoms, forces) ->
   WatchResult``.  :class:`~atominterface.relax.NeighborWatcher` reads
   only atomic positions and tabulated covalent radii, so the geometric
   guard runs with no model, no weights and no calculator -- fully
   offline.

2. :class:`~atominterface.relax.GuardedRelaxer` wraps any ASE-compatible
   calculator (anything exposing ``get_potential_energy`` /
   ``get_forces``) and drives ASE's own FIRE/LBFGS optimizers, returning
   a :class:`~atominterface.relax.RelaxResult` whose ``.valid`` flag is
   the fail-closed verdict.  A 0.6x-compressed fcc Cu cell is caught
   (``valid=False``) and its offending cluster is quarantined as a VASP
   single-point input; the pristine cell passes (``valid=True``).

The MACE backend is optional: its factory imports mace-torch and the
first run downloads the ~44 MB MACE-MP-0 medium checkpoint (需下载 MACE
权重).  That path is wrapped in try/except so the example still
demonstrates the geometry guard when the weights are unavailable.

Run:
    python examples/06_ase_guard_dropin.py
    python examples/06_ase_guard_dropin.py --backend mace
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    from ase.build import bulk
except Exception as exc:  # pragma: no cover - ASE is a hard dependency
    sys.exit(f"[ase-guard-dropin] ASE is required: {exc}")

from atominterface.backends import get_calculator
from atominterface.relax import GuardedRelaxer, NeighborWatcher, RelaxResult, WatchResult

if TYPE_CHECKING:
    from ase.atoms import Atoms

#: Lattice constant of fcc Cu used for the in-distribution reference (A).
FCC_CU_A = 3.615
#: Compression applied to build the out-of-distribution cell.  At 0.6x the
#: nearest-neighbour distance (a/sqrt(2) -> 1.534 A) drops below the
#: covalent-radii threshold the guard uses (0.6 * 2 * 1.32 = 1.584 A).
OOD_SCALE = 0.6
#: OOD score above which the guard stops a relaxation.
OOD_THRESHOLD = 0.8
#: Relaxation parameters.  The dummy backend emits ~0.01 eV/A forces, so
#: geometry alone -- not the force magnitude -- decides the verdict.
FMAX = 0.05
MAX_STEPS = 20
SEED = 0


def make_fcc_cu(scale: float = 1.0) -> Atoms:
    """Four-atom conventional fcc Cu cell with lattice constant ``FCC_CU_A * scale``.

    ``cubic=True`` selects ASE's conventional cell (the primitive cell's
    vector norm is ``a/sqrt(2)``, which would misreport the lattice
    constant).  ``scale < 1`` compresses the nearest-neighbour distance
    below the covalent-radii threshold, turning the cell into an
    out-of-distribution geometry the guard must reject.
    """
    return bulk("Cu", crystalstructure="fcc", a=FCC_CU_A * scale, cubic=True)


def neighbor_score(atoms: Atoms) -> tuple[float, str]:
    """Run the weight-free geometric guard on one structure.

    ``NeighborWatcher`` judges the tightest bond against
    ``min_dist_frac * (r_i + r_j)`` over the pair's tabulated covalent
    radii, so it needs no model, no weights and no calculator.  The
    all-zero forces passed here are ignored by this watcher; inside the
    full guard they are the wrapped calculator's real forces.
    """
    watcher = NeighborWatcher(min_dist_frac=0.6)
    result: WatchResult = watcher(atoms, np.zeros((len(atoms), 3), dtype=float))
    return float(result.score), result.reason


def guarded_relax(atoms: Atoms, calculator, quarantine_dir: Path) -> RelaxResult:
    """Relax ``atoms`` under the guard and return the fail-closed verdict.

    ``calculator`` may be an ASE calculator instance or a callable factory
    ``calc(atoms) -> calculator`` -- exactly what :func:`get_calculator`
    returns.  ``GuardedRelaxer`` attaches it to ``atoms`` and drives ASE's
    own optimizer, so the guard observes the very same ``get_forces`` /
    ``get_potential_energy`` calls any plain ASE run would make.
    """
    relaxer = GuardedRelaxer(
        calculator=calculator,
        optimizer="FIRE",
        fmax=FMAX,
        max_steps=MAX_STEPS,
        ood_threshold=OOD_THRESHOLD,
        quarantine_dir=str(quarantine_dir),
    )
    return relaxer.run(atoms)


def report(label: str, result: RelaxResult) -> None:
    """Print one guard verdict (mirrors the ledger's fail-closed fields)."""
    verdict = "valid=True  (guard passes)" if result.valid else "valid=False (guard intercepts)"
    print(
        f"  {label:<12} n_steps={result.n_steps:>3} "
        f"converged={str(result.converged):<5} ood_triggered={str(result.ood_triggered):<5} "
        f"max_ood_score={result.max_ood_score:.3f}  -> {verdict}"
    )
    if result.reasons:
        print(f"    reasons: {'; '.join(result.reasons)}")
    if result.quarantine_path:
        print(f"    quarantine: {result.quarantine_path}")


def main() -> int:
    """Run the weight-free geometry guard and the full guarded relaxation."""
    parser = argparse.ArgumentParser(description="AtomInterface guard as an ASE drop-in")
    parser.add_argument(
        "--backend",
        default="dummy",
        help="dummy (no weights) | mace (imports mace-torch, downloads weights on first run).",
    )
    parser.add_argument("--outdir", default="outputs/ase_guard_dropin")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    quarantine_dir = outdir / "flagged"
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    reference = make_fcc_cu(1.0)
    compressed = make_fcc_cu(OOD_SCALE)

    print("== AtomInterface guard as an ASE drop-in ==")
    print(
        f"in-distribution : fcc Cu, a={np.linalg.norm(reference.cell[0]):.3f} A, "
        f"{len(reference)} atoms"
    )
    print(
        f"OOD             : fcc Cu, a={np.linalg.norm(compressed.cell[0]):.3f} A "
        f"({OOD_SCALE:.1f}x compressed), {len(compressed)} atoms"
    )

    # 1. The watcher on its own: no calculator, no weights, no model.
    print("\n== 1. weight-free geometry guard (NeighborWatcher, no weights) ==")
    for label, atoms in (("reference", reference), ("compressed", compressed)):
        score, reason = neighbor_score(atoms)
        print(f"  {label:<12} score={score:.3f}  {reason or '(nearest neighbour above threshold)'}")

    # 2. The full guard wrapping an ASE-compatible calculator.
    print(f"\n== 2. guarded relaxation (backend={args.backend}) ==")
    # The guard itself never requires MACE weights -- it is calculator-
    # agnostic.  Only the "mace" backend does: its factory imports
    # mace-torch and the first run downloads the ~44 MB MACE-MP-0 medium
    # checkpoint (需下载 MACE 权重).  If either is missing we degrade to
    # the weight-free geometry guard above instead of crashing.
    try:
        calculator = get_calculator(args.backend, device="cpu", seed=SEED)
    except Exception as exc:  # noqa: BLE001 - optional backend/weights
        print(f"  backend unavailable: {type(exc).__name__}: {exc}")
        print("  the geometry guard above still ran without weights; install the")
        print("  backend and its weights to exercise the full GuardedRelaxer here.")
        return 0

    results: dict[str, RelaxResult] = {}
    for label, atoms in (("reference", reference), ("compressed", compressed)):
        results[label] = guarded_relax(atoms, calculator, quarantine_dir=quarantine_dir)

    for label, result in results.items():
        report(label, result)

    print(f"\nQuarantine artifacts (if any) are written under {quarantine_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
