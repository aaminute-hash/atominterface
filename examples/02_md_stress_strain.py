#!/usr/bin/env python
"""Synthetic MD stress/strain trajectory -> NumPy constitutive table.

Phase 0.4 (AutoConstitutive) data-pipeline demo, pure ase/numpy -- no network,
no MACE.  It builds a small Cu fcc supercell, applies a uniaxial stretch over
a handful of frames with stand-in thermal velocities and Cauchy stress, then
extracts the Green-Lagrange strain / Cauchy stress channels and exports them
as a ``.npz`` constitutive table for downstream fitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.build import bulk
from ase.calculators.calculator import Calculator
from ase.io import write as ase_write
from ase.md.velocitydistribution import thermalize_momenta

from atominterface.physics import (
    export_constitutive_table,
    extract,
    gpa_to_ev_per_a3,
)

if TYPE_CHECKING:
    from ase import Atoms


class _ConstantStress(Calculator):
    """Stand-in calculator reporting a fixed symmetric Cauchy stress tensor.

    A real MD run attaches its own calculator here; :func:`extract` reads
    ``atoms.get_stress(voigt=True)`` (already Cauchy stress) directly.
    """

    implemented_properties = ["stress"]

    def __init__(self, stress: np.ndarray) -> None:
        super().__init__()
        self._stress = np.asarray(stress, dtype=float)

    def get_stress(self, atoms: Atoms | None = None, voigt: bool = True) -> np.ndarray:
        return self._stress


def _build_frames(stretches) -> list[Atoms]:
    """2x2x2 Cu fcc supercell, uniaxially stretched frame by frame."""
    base = bulk("Cu", "fcc", a=3.615) * (2, 2, 2)
    ref_cell = base.cell.copy()
    rng = np.random.RandomState(0)
    frames: list[Atoms] = []
    for s in stretches:
        frame = base.copy()
        cell = ref_cell.copy()
        cell[0] *= s  # uniaxial stretch along x
        frame.set_cell(cell, scale_atoms=True)
        # Deterministic stand-in for a thermostat: a real MD integrator
        # populates velocities/momenta every step.
        thermalize_momenta(frame, 300.0, rng=rng)
        # Linear-elastic stand-in for the per-frame virial stress:
        # sigma_xx = C11 * E_xx with a Cu-like C11 (168 GPa).
        e11 = 0.5 * (s * s - 1.0)
        sigma = np.zeros((3, 3))
        sigma[0, 0] = gpa_to_ev_per_a3(168.0 * e11)
        frame.calc = _ConstantStress(sigma)
        frames.append(frame)
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traj",
        default="outputs/md_stress_strain.extxyz",
        help="trajectory snapshot to write (positions/cell/velocities)",
    )
    parser.add_argument(
        "--out",
        default="outputs/constitutive_table.npz",
        help="output .npz constitutive table",
    )
    args = parser.parse_args(argv)

    stretches = [1.0, 1.02, 1.05, 1.08, 1.12]
    frames = _build_frames(stretches)

    Path(args.traj).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Persist the trajectory for reuse.  extxyz serialises positions, cell and
    # velocities but *not* the calculator, so we extract from the in-memory
    # frames below to keep the stress channel.  For a real MD run, hand
    # extract() the trajectory object or file path directly.
    ase_write(args.traj, frames, format="extxyz")

    # 真实 MD 轨迹替换这里即可 (a real MD trajectory replaces the frames above).
    table = extract(frames)
    table_path = export_constitutive_table(table, args.out)

    last = table.n_frames - 1
    print(f"frames:              {table.n_frames}")
    print(f"trajectory written:  {args.traj}")
    print(f"constitutive table:  {table_path}")
    print(f"last-frame strain:   {np.array2string(table.strain_voigt[last], precision=6)}")
    print(
        "last-frame stress:   "
        f"{np.array2string(table.cauchy_stress_voigt_gpa[last], precision=3)} GPa"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
