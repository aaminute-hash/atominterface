#!/usr/bin/env python
"""float64 vs float32 backend reference comparison.

Quantifies how much of an observed energy/force difference comes from
float32 rounding rather than physics: the same structure is evaluated
twice, once with ``dtype="float32"`` (the historical default) and once
with ``dtype="float64"``, and the residual differences are printed.

Run with the dummy backend to see the harness (identical results, since
the dummy backend is dtype-independent):

    python examples/05_float64_vs_float32.py --backend dummy

Run with a real backend for the actual comparison (requires mace-torch;
the first run downloads the ~44 MB MACE-MP-0 medium checkpoint):

    python examples/05_float64_vs_float32.py --backend mace
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _single_point(atoms, backend: str, dtype: str, device: str) -> tuple[float, np.ndarray]:
    """Return (energy, forces) for ``atoms`` at the requested dtype."""
    from atominterface.backends import get_calculator

    # A fixed seed makes the dummy backend deterministic, so its two
    # runs differ by exactly zero (it has no dtype dependence).
    extra = {"seed": 123} if backend == "dummy" else {}
    calc = get_calculator(backend, device=device, dtype=dtype, **extra)(atoms)
    energy = float(calc.get_potential_energy(atoms))
    forces = np.asarray(calc.get_forces(atoms), dtype=float)
    return energy, forces


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="dummy", help="backend name (default: dummy)")
    parser.add_argument("--device", default="cpu", help="torch device for real backends")
    args = parser.parse_args()

    try:
        from ase.build import bulk
    except Exception as exc:  # pragma: no cover
        sys.exit(f"[float64-vs-float32] ASE is required: {exc}")

    # One atom displaced off the ideal fcc site so forces are non-zero
    # and rounding differences have something to show up in.
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    atoms.positions[0] += [0.05, 0.0, 0.0]

    results = {}
    for dtype in ("float32", "float64"):
        try:
            results[dtype] = _single_point(atoms, args.backend, dtype, args.device)
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"[float64-vs-float32] {dtype} run failed: {type(exc).__name__}: {exc}")
        energy, forces = results[dtype]
        print(f"[{dtype}] energy = {energy:.12f} eV, max|F| = {abs(forces).max():.6e} eV/A")

    d_energy = results["float64"][0] - results["float32"][0]
    d_forces = results["float64"][1] - results["float32"][1]
    print(f"energy difference (float64 - float32): {d_energy:+.6e} eV")
    print(f"max force difference:                  {abs(d_forces).max():.6e} eV/A")
    print(f"rms force difference:                  {np.sqrt((d_forces**2).mean()):.6e} eV/A")
    if args.backend == "dummy":
        print("(dummy backend is dtype-independent; differences are exactly zero)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
