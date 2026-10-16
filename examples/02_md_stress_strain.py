#!/usr/bin/env python
"""Placeholder: extract stress/strain tensors from guarded MD trajectories.

This is the data pipeline that feeds AutoConstitutive (Phase 0.4).
Not yet implemented -- kept as a stub so the downstream API surface is
visible from day one.
"""

from __future__ import annotations

import argparse


def extract(
    trajectory_path: str,
    *,
    reference_volume: float | None = None,
) -> dict:
    """Return stress/strain channel arrays from an ASE trajectory.

    Expected outputs
    ----------------
    strain_voigt : (N, 6) Green-Lagrange strain in Voigt order
    cauchy_stress_voigt : (N, 6) Cauchy stress in Voigt order, eV/A^3
    temperature : (N,) instantaneous temperature
    """
    raise NotImplementedError(
        "MD stress/strain extraction is planned for Phase 0.4 (AutoConstitutive). "
        "See docs/roadmap.md"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", help="ASE trajectory / XYZ / NetCDF path")
    parser.add_argument("--out", default="stress_strain.npz")
    args = parser.parse_args()
    channels = extract(args.trajectory)
    raise SystemExit(f"not implemented yet; would write {channels.keys()} to {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
