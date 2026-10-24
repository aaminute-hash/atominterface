#!/usr/bin/env python
"""End-to-end interface formation energy: Cu(001) | Ni(001).

Runs the full energy ledger behind an interface formation energy, with
one calculator setting (same backend, same device) for every term:

  1. load the two bulks (examples/data/Cu_fcc.vasp, Ni_fcc.vasp)
  2. coherent candidates via CoherentInterfaceBuilder (1x1 cell by default)
  3. bulk single points (per-atom and total)
  4. reference slabs -- same supercell, strain and termination as the
     interface, via reference_slabs()
  5. guarded relaxation of both reference slabs and the interface
  6. termination energies, then the interface formation energy

Outputs (in --outdir, default outputs/energy_cu_ni):

  candidates.csv   one-row energy ledger (bulk/slab/interface/gamma/OOD)
  metadata.yaml    provenance: backend identity, seed, artifact hash
  *.extxyz         relaxed interface, reference slabs and bulk inputs

The dummy backend runs the whole chain as a wiring self-test, but its
energies carry no physical meaning; --backend mace (the default) is the
real driver.  The seed is fixed (SEED below): the CSV ledger
reproduces to its written precision and the extxyz column order is
canonicalised, so the file layout never drifts with PYTHONHASHSEED.  A
residual ~1e-7 cross-process noise (torch CPU thread-reduction order,
machine-load dependent) can flip the last written decimal; reruns in
the same process are bit-identical.

Run:
    python examples/03_interface_energy.py --backend dummy
    python examples/03_interface_energy.py --backend mace --device cpu
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from ase.io import write as ase_write
except Exception as exc:  # pragma: no cover - ASE is a hard dependency
    sys.exit(f"[interface-energy] ASE is required: {exc}")

DATA_DIR = Path(__file__).resolve().parent / "data"
SUBSTRATE_FILE = "Cu_fcc.vasp"
FILM_FILE = "Ni_fcc.vasp"

#: Numerical reproducibility: the seed is fixed and recorded in metadata.
SEED = 0
#: OOD score that stops a relaxation early (the CLI exposes the knob).
OOD_THRESHOLD = 0.8
#: Interface area ceiling for the ZSL search; the 1x1 (001) cell is ~13 A^2,
#: so the default only admits the small commensurate cells.
MAX_AREA = 100.0
GAP = 2.0
#: Miller indices of the demonstrated Cu(001) | Ni(001) interface.
SUBSTRATE_MILLER = (0, 0, 1)
FILM_MILLER = (0, 0, 1)

CSV_FIELDS = [
    "rank",
    "interface_id",
    "substrate",
    "film",
    "sub_termination",
    "film_termination",
    "area_A2",
    "n_atoms",
    "E_bulk_sub_ev_per_atom",
    "E_bulk_film_ev_per_atom",
    "E_slab_sub_ev",
    "E_slab_film_ev",
    "E_interface_ev",
    "gamma_surf_sub_eV_A2",
    "gamma_surf_film_eV_A2",
    "gamma_eV_per_A2",
    "gamma_J_per_m2",
    "ood_score",
    "ood_step",
    "ood_triggered",
    "backend",
    "model_hash",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the demo's command line (mirrors ``atominterface build``)."""
    parser = argparse.ArgumentParser(
        description="Cu(001) | Ni(001) interface formation energy, end to end."
    )
    parser.add_argument(
        "--backend",
        default="mace",
        help="mace (default, real driver) | dummy (chain self-test, not physical) | chgnet.",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="compute device.")
    parser.add_argument(
        "--min-thickness", type=float, default=6.0, help="minimum slab thickness in A."
    )
    parser.add_argument("--vacuum", type=float, default=15.0, help="vacuum padding in A.")
    parser.add_argument("--max-candidates", type=int, default=1, help="keep at most N candidates.")
    parser.add_argument("--fmax", type=float, default=0.05, help="force tolerance in eV/A.")
    parser.add_argument(
        "--max-steps", type=int, default=200, help="relaxation step cap per structure."
    )
    parser.add_argument("--outdir", default="outputs/energy_cu_ni")
    return parser.parse_args(argv)


def _load_bulks():
    """Load the Cu/Ni bulks shipped next to this script."""
    try:
        from pymatgen.core import Structure
    except ImportError:
        sys.exit("[interface-energy] pymatgen is required: pip install pymatgen")
    paths = (DATA_DIR / SUBSTRATE_FILE, DATA_DIR / FILM_FILE)
    for path in paths:
        if not path.exists():
            sys.exit(f"[interface-energy] bulk file not found: {path}")
    return Structure.from_file(str(paths[0])), Structure.from_file(str(paths[1]))


def _to_ase(structure):
    """ASE view of a pymatgen Structure."""
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_atoms(structure)


def _make_calculator(backend: str, device: str):
    """One calculator factory shared by every energy in the ledger.

    The seed is honoured by the dummy backend (deterministic stand-in
    forces); real backends are deterministic by construction and ignore
    it.
    """
    from atominterface.backends import get_calculator

    return get_calculator(backend, device=device, seed=SEED)


def _single_point(atoms, calculator) -> tuple[float, object]:
    """Total energy of one structure, without relaxing it.

    Returns ``(energy, measured_atoms)``; the measured copy carries the
    calculator results so the written file records the single point.
    """
    measured = atoms.copy()
    measured.calc = calculator(measured)
    return float(measured.get_potential_energy()), measured


def _build_candidate(substrate, film, args: argparse.Namespace):
    """First coherent candidate plus the match its reference slabs reuse.

    ``build_all()`` consumed the match list; ``_match()`` re-derives it
    deterministically (same slabs, same ranking), so the reference slabs
    share the candidate's supercell and strain exactly.
    """
    from atominterface.builders import CoherentInterfaceBuilder, CoherentInterfaceConfig

    config = CoherentInterfaceConfig(
        substrate_miller=SUBSTRATE_MILLER,
        film_miller=FILM_MILLER,
        max_area=MAX_AREA,
        gap=GAP,
        min_thickness=args.min_thickness,
        vacuum_thickness=args.vacuum,
        max_candidates=args.max_candidates,
    )
    builder = CoherentInterfaceBuilder(substrate=substrate, film=film, config=config)
    candidates = builder.build()
    if not candidates:
        return None, None, config
    if len(candidates) > 1:
        print(f"note: {len(candidates)} candidate(s) kept; the ledger covers the first one.")
    return candidates[0], builder._match()[0], config


def _reference_slabs(substrate, film, config, match, candidate):
    """Same supercell, strain and termination as the interface sides."""
    from atominterface.builders import reference_slabs

    return reference_slabs(
        substrate,
        film,
        config,
        match,
        substrate_termination=int(candidate.substrate_termination),
        film_termination=int(candidate.film_termination),
    )


def _guarded_relax(atoms, calculator, *, fmax: float, max_steps: int, quarantine_dir: Path):
    """Relax one structure under the OOD guard; returns the RelaxResult."""
    from atominterface.relax import GuardedRelaxer

    relaxer = GuardedRelaxer(
        calculator=calculator,
        fmax=fmax,
        max_steps=max_steps,
        ood_threshold=OOD_THRESHOLD,
        quarantine_dir=str(quarantine_dir),
    )
    return relaxer.run(atoms)


def _relax_ledger(candidate, reference_sub, reference_film, calculator, args, outdir: Path):
    """Guarded relaxation of both reference slabs and the interface.

    Returns the three ``RelaxResult`` objects and the three relaxed
    total energies in ledger order.
    """
    relax_kwargs = {
        "fmax": args.fmax,
        "max_steps": args.max_steps,
        "quarantine_dir": outdir / "flagged",
    }
    relax_sub = _guarded_relax(reference_sub, calculator, **relax_kwargs)
    relax_film = _guarded_relax(reference_film, calculator, **relax_kwargs)
    relax_interface = _guarded_relax(candidate.atoms, calculator, **relax_kwargs)
    energies = (
        float(relax_sub.atoms.get_potential_energy()),
        float(relax_film.atoms.get_potential_energy()),
        float(relax_interface.atoms.get_potential_energy()),
    )
    return relax_sub, relax_film, relax_interface, energies


def _gammas(
    *,
    candidate,
    substrate_atoms,
    film_atoms,
    reference_sub,
    reference_film,
    e_bulk_sub: float,
    e_bulk_film: float,
    energies: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Termination energies and the interface formation energy.

    Each slab energy is referenced against its own bulk (the bulk terms
    cancel in the interface energy), with ``n_surfaces=2`` for the
    asymmetric slab's two exposed faces.
    """
    from atominterface.thermo import (
        interface_formation_energy,
        slab_formula_units,
        termination_energy,
    )

    e_slab_sub, e_slab_film, e_interface = energies
    area = float(candidate.area)
    gamma_sub = termination_energy(
        slab_energy=e_slab_sub,
        bulk_energy=e_bulk_sub,
        n_formula_units=slab_formula_units(reference_sub, substrate_atoms),
        area=area,
        n_surfaces=2,
    )
    gamma_film = termination_energy(
        slab_energy=e_slab_film,
        bulk_energy=e_bulk_film,
        n_formula_units=slab_formula_units(reference_film, film_atoms),
        area=area,
        n_surfaces=2,
    )
    gamma_int = interface_formation_energy(
        interface_energy_total=e_interface,
        substrate_slab_energy=e_slab_sub,
        film_slab_energy=e_slab_film,
        substrate_surface_energy=gamma_sub,
        film_surface_energy=gamma_film,
        area=area,
    )
    return area, gamma_sub, gamma_film, gamma_int


def _identity(args: argparse.Namespace) -> tuple[str, str]:
    """Backend identity string and version, for provenance and the CSV."""
    from atominterface.backends import model_identity_for_meta, resolve_model_meta

    identity = model_identity_for_meta(args.backend, device=args.device)
    backend_version = resolve_model_meta(args.backend, device=args.device)["backend_version"]
    return identity, backend_version


def _write(path: Path, atoms) -> None:
    """Write one extxyz with canonical array order (byte-stable layout).

    pymatgen site properties reach ``atoms.arrays`` in set-iteration
    order; without the normalisation the extxyz column order -- and
    with it the file bytes -- would drift between processes.
    """
    from atominterface.io import canonicalize_arrays

    canonicalize_arrays(atoms)
    ase_write(path, atoms)


def _jsonable(value):
    """argparse values -> YAML/JSON-safe primitives (Path -> str)."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    """Full SHA-256 hex digest of an artifact file."""
    from atominterface.provenance import hash_bytes

    return hash_bytes(path.read_bytes())


def _write_outputs(
    *,
    outdir: Path,
    row: dict,
    relax_sub,
    relax_film,
    relax_interface,
    bulk_sub_measured,
    bulk_film_measured,
    identity: str,
    backend_version: str,
    args: argparse.Namespace,
) -> Path:
    """Write candidates.csv, the relaxed structures and metadata.yaml."""
    from atominterface.provenance import build_meta, write_metadata

    csv_path = outdir / "candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows([row])

    _write(outdir / "interface_relaxed.extxyz", relax_interface.atoms)
    _write(outdir / "slab_sub_relaxed.extxyz", relax_sub.atoms)
    _write(outdir / "slab_film_relaxed.extxyz", relax_film.atoms)
    _write(outdir / "bulk_sub.extxyz", bulk_sub_measured)
    _write(outdir / "bulk_film.extxyz", bulk_film_measured)

    meta = build_meta(
        backend=args.backend,
        backend_version=backend_version,
        model_hash=identity,
        seed=SEED,
        cli_args={key: _jsonable(value) for key, value in vars(args).items()},
        artifacts={"candidates.csv": _sha256_file(csv_path)},
    )
    write_metadata(meta, outdir / "metadata.yaml")
    return csv_path


def main(argv: list[str] | None = None) -> int:
    """Run the Cu/Ni interface energy ledger; returns a process exit code."""
    from atominterface.physics import EV_A2_TO_J_M2

    args = _parse_args(argv)
    np.random.seed(SEED)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    substrate, film = _load_bulks()
    substrate_atoms = _to_ase(substrate)
    film_atoms = _to_ase(film)
    print("== Cu(001) | Ni(001) interface formation energy ==")
    print(f"substrate: Cu fcc, a={np.linalg.norm(substrate_atoms.cell[0]):.3f} A")
    print(f"film     : Ni fcc, a={np.linalg.norm(film_atoms.cell[0]):.3f} A")

    candidate, match, config = _build_candidate(substrate, film, args)
    if candidate is None:
        print("no coherent candidate under the current tolerances")
        return 1
    print(f"candidate: {candidate.interface_id}")
    print(
        f"          area={candidate.area:.2f} A^2, {candidate.total_atoms} atoms, "
        f"mismatch={candidate.metadata['mismatch']:.4f}"
    )

    calculator = _make_calculator(args.backend, args.device)
    print(f"backend  : {args.backend} (device {args.device})")

    # Bulk single points: per-atom values are recorded, totals feed the
    # termination energies.
    e_bulk_sub, bulk_sub_measured = _single_point(substrate_atoms, calculator)
    e_bulk_film, bulk_film_measured = _single_point(film_atoms, calculator)

    # Reference slabs: same supercell, same strain, same termination as
    # the interface side they reference.
    reference_sub, reference_film = _reference_slabs(substrate, film, config, match, candidate)
    relax_sub, relax_film, relax_interface, energies = _relax_ledger(
        candidate, reference_sub, reference_film, calculator, args, outdir
    )
    e_slab_sub, e_slab_film, e_interface = energies

    area, gamma_sub, gamma_film, gamma_int = _gammas(
        candidate=candidate,
        substrate_atoms=substrate_atoms,
        film_atoms=film_atoms,
        reference_sub=reference_sub,
        reference_film=reference_film,
        e_bulk_sub=e_bulk_sub,
        e_bulk_film=e_bulk_film,
        energies=energies,
    )
    identity, backend_version = _identity(args)

    row = {
        "rank": 1,
        "interface_id": candidate.interface_id,
        "substrate": candidate.substrate_formula,
        "film": candidate.film_formula,
        "sub_termination": candidate.substrate_termination,
        "film_termination": candidate.film_termination,
        "area_A2": f"{area:.4f}",
        "n_atoms": candidate.total_atoms,
        "E_bulk_sub_ev_per_atom": f"{e_bulk_sub / len(substrate_atoms):.6f}",
        "E_bulk_film_ev_per_atom": f"{e_bulk_film / len(film_atoms):.6f}",
        "E_slab_sub_ev": f"{e_slab_sub:.6f}",
        "E_slab_film_ev": f"{e_slab_film:.6f}",
        "E_interface_ev": f"{e_interface:.6f}",
        "gamma_surf_sub_eV_A2": f"{gamma_sub:.6f}",
        "gamma_surf_film_eV_A2": f"{gamma_film:.6f}",
        "gamma_eV_per_A2": f"{gamma_int:.6f}",
        "gamma_J_per_m2": f"{gamma_int * EV_A2_TO_J_M2:.4f}",
        "ood_score": f"{relax_interface.max_ood_score:.6f}",
        "ood_step": str(relax_interface.ood_step),
        "ood_triggered": str(relax_interface.ood_triggered),
        "backend": args.backend,
        "model_hash": identity,
    }
    csv_path = _write_outputs(
        outdir=outdir,
        row=row,
        relax_sub=relax_sub,
        relax_film=relax_film,
        relax_interface=relax_interface,
        bulk_sub_measured=bulk_sub_measured,
        bulk_film_measured=bulk_film_measured,
        identity=identity,
        backend_version=backend_version,
        args=args,
    )

    print("\n== energy ledger ==")
    print(
        f"E_bulk  sub (Cu): {e_bulk_sub:10.4f} eV total | "
        f"{e_bulk_sub / len(substrate_atoms):8.4f} eV/atom"
    )
    print(
        f"E_bulk film (Ni): {e_bulk_film:10.4f} eV total | "
        f"{e_bulk_film / len(film_atoms):8.4f} eV/atom"
    )
    # GuardedRelaxer.n_steps counts guard watch cycles, not optimizer
    # steps (one opt.run() call relaxes to fmax, so a converged relax
    # reports 1); the ledger therefore prints the convergence flag.
    print(f"E_slab  sub     : {e_slab_sub:10.4f} eV (converged={relax_sub.converged})")
    print(f"E_slab film     : {e_slab_film:10.4f} eV (converged={relax_film.converged})")
    print(f"E_interface     : {e_interface:10.4f} eV (converged={relax_interface.converged})")
    print(f"gamma_surf Cu   : {gamma_sub:10.4f} eV/A^2 = {gamma_sub * EV_A2_TO_J_M2:7.3f} J/m^2")
    print(f"gamma_surf Ni   : {gamma_film:10.4f} eV/A^2 = {gamma_film * EV_A2_TO_J_M2:7.3f} J/m^2")
    print(f"gamma_int       : {gamma_int:10.4f} eV/A^2 = {gamma_int * EV_A2_TO_J_M2:7.3f} J/m^2")
    ood = relax_interface
    print(
        f"OOD (interface) : score={ood.max_ood_score:.3f} step={ood.ood_step} "
        f"triggered={ood.ood_triggered}"
    )
    print(f"\nWrote {csv_path}, metadata.yaml and relaxed structures to {outdir}/")
    if args.backend == "dummy":
        print("note: dummy energies are not physical; the numbers only prove the chain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
