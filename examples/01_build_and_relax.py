#!/usr/bin/env python
"""AtomInterface quickstart: Li (bcc) | NaCl (rocksalt).

5-minute demo, no GPU / no DFT required.

Pipeline:
  1. build bulk Li and NaCl with ASE
  2. coherent interface candidates via pymatgen CoherentInterfaceBuilder
  3. strain filter
  4. guarded relax with the dummy backend
  5. write candidates + a short report

Install extras:
    pip install -e ".[pymatgen,dummy]"

Run:
    python examples/01_build_and_relax.py --backend dummy
    python examples/01_build_and_relax.py --backend dummy --relax
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from ase.build import bulk
    from ase.io import write as ase_write
except Exception as exc:  # pragma: no cover
    sys.exit(f"[quickstart] ASE is required: {exc}")


def make_bulks():
    """Return substrate (Li, bcc) and film (NaCl, rocksalt) ASE structures."""
    li = bulk("Li", crystalstructure="bcc", a=3.49)
    nacl = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
    return li, nacl


def build_coherent_interfaces(
    substrate_ase,
    film_ase,
    substrate_miller=(0, 0, 1),
    film_miller=(0, 0, 1),
    max_strain=0.08,
    max_area=600,
    gap=2.0,
    vacuum_over_film=12.0,
    film_layers=2,
    substrate_layers=3,
):
    try:
        from pymatgen.analysis.interfaces.coherent_interfaces import (
            CoherentInterfaceBuilder,
        )
        from pymatgen.analysis.interfaces.zsl import ZSLGenerator
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError:
        sys.exit(
            "[quickstart] pymatgen not installed.\n"
            "Install it with: pip install pymatgen\n"
            "Or skip interface building and use --preset-stack for the dummy path."
        )

    sub = AseAtomsAdaptor.get_structure(substrate_ase)
    film = AseAtomsAdaptor.get_structure(film_ase)

    zslgen = ZSLGenerator(
        max_area_ratio_tol=0.12,
        max_area=max_area,
        max_length_tol=0.04,
        max_angle_tol=0.02,
    )

    builder = CoherentInterfaceBuilder(
        substrate_structure=sub,
        film_structure=film,
        substrate_miller=substrate_miller,
        film_miller=film_miller,
        zslgen=zslgen,
        label_index=True,
    )

    candidates = []
    for termination in builder.terminations:
        try:
            interfaces = builder.get_interfaces(
                termination=termination,
                gap=gap,
                vacuum_over_film=vacuum_over_film,
                film_thickness=film_layers,
                substrate_thickness=substrate_layers,
                in_layers=True,
            )
        except Exception as exc:
            print(f"  ! termination {termination} failed: {exc}")
            continue

        for iface in interfaces:
            atoms = AseAtomsAdaptor.get_atoms(iface, msonable=False)
            props = getattr(iface, "interface_properties", {}) or {}
            strain = props.get("strain")
            von_mises = props.get("von_mises")

            atoms.info["termination"] = str(termination)
            atoms.info["match_area"] = float(props.get("match_area", float("nan")))
            atoms.info["strain_trace"] = (
                float(np.trace(np.array(strain))) if strain is not None else float("nan")
            )
            atoms.info["von_mises_strain"] = (
                float(von_mises) if von_mises is not None else float("nan")
            )

            candidates.append(atoms)

    kept = []
    for atoms in candidates:
        vm = atoms.info.get("von_mises_strain", float("nan"))
        if np.isnan(vm) or abs(vm) <= max_strain:
            kept.append(atoms)
    kept.sort(key=lambda a: a.info.get("von_mises_strain", 1e9))
    return kept


def preset_stack(substrate_ase, film_ase, gap=2.0, vacuum=12.0):
    """Very rough demo stack: not a real coherent interface."""
    sub = substrate_ase.copy()
    film = film_ase.copy()
    sub.cell[2, 2] += gap
    film.positions[:, 2] += sub.cell[2, 2]
    atoms = sub + film
    atoms.cell[2, 2] += vacuum
    atoms.center(axis=2)
    atoms.pbc = [True, True, True]
    atoms.info["termination"] = "manual-stack"
    atoms.info["note"] = "fallback demo only, not coherent-matched"
    return [atoms]


def relax_candidate(atoms, backend, fmax=0.2, steps=80):
    try:
        from atominterface.backends import get_calculator
        from atominterface.relax import GuardedRelaxer
    except Exception as exc:
        print(f"  ! AtomInterface relax layer unavailable: {exc}")
        return atoms, None

    calc = get_calculator(backend)
    relaxer = GuardedRelaxer(
        calculator=calc,
        optimizer="FIRE",
        fmax=fmax,
        max_steps=steps,
        ood_threshold=0.9,
        quarantine_dir="./flagged/",
    )
    result = relaxer.run(atoms)
    return result.atoms, result


def summarize(candidates, outdir: Path):
    outdir.mkdir(exist_ok=True)
    rows = []
    for i, atoms in enumerate(candidates):
        n = len(atoms)
        area = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
        rows.append(
            {
                "id": i,
                "termination": str(atoms.info.get("termination", "?")),
                "von_mises_strain": float(atoms.info.get("von_mises_strain", float("nan"))),
                "area_A2": float(area),
                "natoms": n,
            }
        )
        ase_write(outdir / f"candidate_{i:02d}.xyz", atoms)

    print("\nCandidate interfaces")
    print(f"{'id':>3}  {'termination':28} {'von_mises':>10} {'area/A^2':>10} {'natoms':>7}")
    for r in rows:
        vm_str = "  -  " if np.isnan(r["von_mises_strain"]) else f"{r['von_mises_strain']:+.4f}"
        print(
            f"{r['id']:>3}  {r['termination'][:28]:28} {vm_str:>10} {r['area_A2']:10.2f} {r['natoms']:7d}"
        )

    import csv as _csv

    with (outdir / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(
            f, fieldnames=["id", "termination", "von_mises_strain", "area_A2", "natoms"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote XYZ candidates and candidates.csv to {outdir}/")


def main():
    parser = argparse.ArgumentParser(description="AtomInterface Li | NaCl quickstart")
    parser.add_argument("--backend", default="dummy", help="dummy | mace | chgnet")
    parser.add_argument("--sub-miller", nargs=3, type=int, default=(0, 0, 1))
    parser.add_argument("--film-miller", nargs=3, type=int, default=(0, 0, 1))
    parser.add_argument("--max-strain", type=float, default=0.08)
    parser.add_argument("--relax", action="store_true", help="run guarded dummy relax")
    parser.add_argument("--outdir", default="outputs/quickstart_li_nacl")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("== AtomInterface quickstart ==")
    li, nacl = make_bulks()
    ase_write(outdir / "Li_bulk.xyz", li)
    ase_write(outdir / "NaCl_bulk.xyz", nacl)
    print(f"substrate: Li bcc, a={li.cell[0, 0]:.3f} A")
    print(f"film:      NaCl rocksalt, a={nacl.cell[0, 0]:.3f} A")

    candidates = build_coherent_interfaces(
        li,
        nacl,
        substrate_miller=tuple(args.sub_miller),
        film_miller=tuple(args.film_miller),
        max_strain=args.max_strain,
    )

    if not candidates:
        print("\nNo coherent match under current ZSL tolerances.")
        print("Falling back to a manual stacked slab for the relax demo only.")
        candidates = preset_stack(li, nacl)
    else:
        print(f"\nFound {len(candidates)} coherent candidate(s) before strain filter.")

    summarize(candidates, outdir)

    if args.relax:
        print("\n== guarded relax on first candidate ==")
        atoms = candidates[0].copy()
        relaxed, result = relax_candidate(atoms, backend=args.backend)
        ase_write(outdir / "relaxed_first.xyz", relaxed)
        if result is not None:
            print(f"steps={result.n_steps} ood_triggered={result.ood_triggered}")
            forces = relaxed.get_forces()
            print(f"max_force={np.max(np.linalg.norm(forces, axis=1)):.4f} eV/A")
        print(f"Wrote relaxed_first.xyz to {outdir}/")

    print("\nDone. Open the XYZ files in ASE/GUI or Ovito to inspect.")


if __name__ == "__main__":
    main()
