#!/usr/bin/env python
"""End-to-end interface formation energy: Cu(001) | Ni(001).

Runs the full energy ledger behind an interface formation energy, with
one calculator setting (same backend, same device) for every term:

  1. load the two bulks (examples/data/Cu_fcc.vasp, Ni_fcc.vasp)
  2. coherent candidates via CoherentInterfaceBuilder (1x1 cell by default)
  3. bulk single points: the pristine cells, plus the *coherent*
     reference (in-plane strained like the slabs, out-of-plane relaxed)
     that the termination energies must use -- a strained slab measured
     against a pristine bulk leaves n * dE_strain in gamma, which drifts
     with layer count instead of converging
  4. reference slabs -- same supercell, strain and termination as the
     interface, via reference_slabs()
  5. guarded relaxation of both reference slabs and the interface
  6. termination energies, then the interface formation energy

Outputs (in --outdir, default outputs/energy_cu_ni):

  candidates.csv   one-row ledger with energy-validity and OOD diagnostics
  metadata.yaml    provenance: backend identity, seed, validity, artifact hash
  *.extxyz         final interface, reference slabs and bulk inputs

Gamma values are emitted only when every pristine/coherent bulk reference
energy is finite and both reference slabs plus the interface pass finite
energy/force, convergence and OOD checks. Invalid runs keep their diagnostic
row and structures, leave gamma cells empty, and exit non-zero. A zero
``ood_score`` with ``committee_enabled=false`` means no committee was queried.

The dummy backend runs the whole chain as a wiring self-test, but its
energies carry no physical meaning; --backend mace (the default) is the
real driver.  The seed is fixed (SEED below): the CSV ledger
reproduces to its written precision and the extxyz column order is
canonicalised, so the file layout never drifts with PYTHONHASHSEED.  A
residual ~1e-7 cross-process noise (torch CPU thread-reduction order,
machine-load dependent) can flip the last written decimal; reruns in
the same process are bit-identical.

Run:
    python examples/03_interface_energy.py --backend dummy --fmax 0.05
    python examples/03_interface_energy.py --backend mace --device cpu

Fail-closed demonstration (non-zero exit, empty gamma cells):
    python examples/03_interface_energy.py --backend dummy --max-steps 1 --fmax 1e-6
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
    "E_bulk_sub_coh_ev_per_atom",
    "E_bulk_film_coh_ev_per_atom",
    "E_slab_sub_ev",
    "E_slab_film_ev",
    "E_interface_ev",
    "gamma_surf_sub_eV_A2",
    "gamma_surf_film_eV_A2",
    "gamma_eV_per_A2",
    "gamma_J_per_m2",
    "valid",
    "converged",
    "energies_finite",
    "slab_sub_valid",
    "slab_film_valid",
    "interface_valid",
    "invalid_reasons",
    "committee_enabled",
    "ood_monitoring",
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
        "--min-thickness",
        type=float,
        default=14.4,
        help=(
            "minimum slab thickness in A. 14.4 A cuts 8 Cu / 10 Ni layers, "
            "the plateau point of the thickness scan in "
            "docs/validation_cu_ni_001.md; the previous 6.0 A default cut "
            "only 4 Ni layers and reported gamma_Ni = 1.979 J/m^2 and "
            "gamma_int = 0.041 J/m^2 against converged values of 1.9189 and "
            "0.0590 -- 3%% and 30%% off, respectively."
        ),
    )
    parser.add_argument("--vacuum", type=float, default=15.0, help="vacuum padding in A.")
    parser.add_argument("--max-candidates", type=int, default=1, help="keep at most N candidates.")
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.01,
        help=(
            "force tolerance in eV/A. The 0.05 eV/A convention is too loose "
            "here: the force driving a Cu slab's out-of-plane expansion is "
            "itself ~0.05 eV/A, so the interior never reaches the bulk "
            "spacing the termination energy references and gamma drifts with "
            "layer count (see docs/validation_cu_ni_001.md)."
        ),
    )
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
    energy = float(measured.get_potential_energy())
    forces = np.asarray(measured.get_forces(), dtype=float)
    if not np.isfinite(energy):
        raise ValueError("single-point energy is not finite")
    if not np.all(np.isfinite(forces)):
        raise ValueError("single-point forces are not finite")
    return energy, measured


def _coherent_bulk_energy(atoms, strain: float, calculator) -> float:
    """Total energy of the bulk under the slab's in-plane strain.

    Both reference slabs share the interface's in-plane lattice, so the
    film side carries the mismatch strain (2.6% for Cu/Ni) while the
    substrate side does not.  Referencing a strained slab against the
    *pristine* bulk leaves ``n * dE_strain`` inside the termination
    energy -- a term that grows with layer count, which is why gamma_Ni
    used to drift instead of converging.  The coherent bulk (in-plane
    locked to the slab strain, out-of-plane relaxed through the
    Poisson response) is the reference that removes it.

    The out-of-plane strain is found by a parabola fitted to the three
    samples bracketing the discrete grid minimum, not by a fit over the
    whole grid: over a 10% window E(sz) is strongly anharmonic, and a
    wide least-squares parabola dips below the true curve (it reported
    +0.57 meV/atom of relaxation gain for Cu where a dense scan finds
    +0.35).  The local fit reproduces the dense-scan minimum to within
    0.01 meV/atom on both Cu and Ni.

    Args:
        atoms: Pristine bulk cell (ASE).
        strain: Isotropic in-plane strain carried by the slab
            (``candidate.strain_substrate`` / ``candidate.strain_film``).
            A general (anisotropic or sheared) match would need the full
            deformation tensor here; the (001) fcc demo is isotropic.
        calculator: Calculator factory shared by the ledger.

    Returns:
        Total energy of the strained bulk cell, in eV.
    """
    import numpy as np

    base_cell = np.array(atoms.get_cell(), dtype=float)
    grid = np.linspace(-0.08, 0.02, 9)
    energies = []
    for sz in grid:
        strained = atoms.copy()
        # Cartesian-component scaling: the pristine fcc cell is cubic with
        # the slab normal along z, so this is exactly (e, e, sz).
        strained.set_cell(
            base_cell * np.array([1.0 + strain, 1.0 + strain, 1.0 + float(sz)]),
            scale_atoms=True,
        )
        strained.calc = calculator(strained)
        energy = float(strained.get_potential_energy())
        if not np.isfinite(energy):
            raise ValueError(f"coherent-bulk energy is not finite at out-of-plane strain {sz}")
        energies.append(energy)
    profile = np.asarray(energies, dtype=float)
    index = int(np.argmin(profile))
    if index in (0, len(grid) - 1):
        # Monotonic across the sampled window: the minimum lies outside
        # it, and extrapolating is worse than taking the best sample.
        return float(profile[index])
    lower, centre, upper = (float(profile[index + offset]) for offset in (-1, 0, 1))
    curvature = lower - 2.0 * centre + upper
    if abs(curvature) < 1e-12:
        # Flat profile (a calculator whose energy does not depend on the
        # cell, e.g. the dummy wiring self-test, or a strain-free side):
        # there is no out-of-plane minimum to fit, so every grid point is
        # the answer.  Fitting a parabola through a constant would divide
        # by zero curvature.
        return float(profile[index])
    return float(centre - (lower - upper) ** 2 / (8.0 * curvature))


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
        relax_sub.final_energy,
        relax_film.final_energy,
        relax_interface.final_energy,
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
        extra={
            "valid": row["valid"] == "True",
            "invalid_reasons": row["invalid_reasons"],
            "committee_enabled": row["committee_enabled"] == "True",
            "ood_monitoring": row["ood_monitoring"],
        },
    )
    write_metadata(meta, outdir / "metadata.yaml")
    return csv_path


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912, PLR0915
    """Run the Cu/Ni interface energy ledger; returns a process exit code."""
    from atominterface.physics import EV_A2_TO_J_M2
    from atominterface.relax import RelaxResult

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

    errors: list[str] = []

    def record_error(stage: str, exc: Exception) -> None:
        message = f"{stage}: {type(exc).__name__}: {exc}"
        errors.append(message)
        print(f"invalid: {message}")

    calculator = None
    try:
        calculator = _make_calculator(args.backend, args.device)
    except Exception as exc:  # noqa: BLE001
        record_error("calculator", exc)
    print(f"backend  : {args.backend} (device {args.device})")

    e_bulk_sub = None
    e_bulk_film = None
    e_bulk_sub_coh = None
    e_bulk_film_coh = None
    bulk_sub_measured = substrate_atoms.copy()
    bulk_film_measured = film_atoms.copy()
    if calculator is not None:
        try:
            e_bulk_sub, bulk_sub_measured = _single_point(substrate_atoms, calculator)
        except Exception as exc:  # noqa: BLE001
            record_error("bulk_sub_pristine", exc)
        try:
            e_bulk_film, bulk_film_measured = _single_point(film_atoms, calculator)
        except Exception as exc:  # noqa: BLE001
            record_error("bulk_film_pristine", exc)
        try:
            e_bulk_sub_coh = _coherent_bulk_energy(
                substrate_atoms, float(candidate.strain_substrate), calculator
            )
        except Exception as exc:  # noqa: BLE001
            record_error("bulk_sub_coherent", exc)
        try:
            e_bulk_film_coh = _coherent_bulk_energy(
                film_atoms, float(candidate.strain_film), calculator
            )
        except Exception as exc:  # noqa: BLE001
            record_error("bulk_film_coherent", exc)

    references_ready = True
    try:
        reference_sub, reference_film = _reference_slabs(substrate, film, config, match, candidate)
    except Exception as exc:  # noqa: BLE001
        record_error("reference_slabs", exc)
        references_ready = False
        reference_sub = substrate_atoms.copy()
        reference_film = film_atoms.copy()

    def invalid_relax(atoms, message: str) -> RelaxResult:
        return RelaxResult(
            atoms=atoms.copy(),
            exception=message,
            reasons=[message],
            metadata={"committee_enabled": False, "ood_monitoring": "heuristics_only"},
        )

    if calculator is not None and references_ready:
        try:
            relax_sub, relax_film, relax_interface, energies = _relax_ledger(
                candidate, reference_sub, reference_film, calculator, args, outdir
            )
        except Exception as exc:  # noqa: BLE001
            # Construction-time failures (bad calculator, missing weights)
            # must not crash the driver: the ledger is written with an
            # explicit reason instead of a bare traceback.
            record_error("relaxation", exc)
            message = f"{type(exc).__name__}: {exc}"
            relax_sub = invalid_relax(reference_sub, message)
            relax_film = invalid_relax(reference_film, message)
            relax_interface = invalid_relax(candidate.atoms, message)
            energies = (None, None, None)
    else:
        message = "relaxation skipped because calculator or reference construction failed"
        relax_sub = invalid_relax(reference_sub, message)
        relax_film = invalid_relax(reference_film, message)
        relax_interface = invalid_relax(candidate.atoms, message)
        energies = (None, None, None)

    relaxations = {
        "slab_sub": relax_sub,
        "slab_film": relax_film,
        "interface": relax_interface,
    }
    for label, result in relaxations.items():
        if result.valid:
            continue
        if result.exception:
            errors.append(f"{label}: {result.exception}")
        errors.extend(f"{label}: {reason}" for reason in result.reasons if reason)

    identity = ""
    backend_version = ""
    identity_valid = True
    try:
        identity, backend_version = _identity(args)
    except Exception as exc:  # noqa: BLE001
        record_error("model_identity", exc)
        identity_valid = False

    required_energies = (e_bulk_sub, e_bulk_film, e_bulk_sub_coh, e_bulk_film_coh)
    relaxation_valid = all(result.valid for result in relaxations.values())
    chain_valid = (
        references_ready
        and identity_valid
        and all(value is not None and np.isfinite(value) for value in required_energies)
        and relaxation_valid
        and all(value is not None and np.isfinite(value) for value in energies)
    )

    area = float(candidate.area)
    gamma_sub = None
    gamma_film = None
    gamma_int = None
    if chain_valid:
        try:
            area, gamma_sub, gamma_film, gamma_int = _gammas(
                candidate=candidate,
                substrate_atoms=substrate_atoms,
                film_atoms=film_atoms,
                reference_sub=reference_sub,
                reference_film=reference_film,
                e_bulk_sub=e_bulk_sub_coh,
                e_bulk_film=e_bulk_film_coh,
                energies=energies,
            )
            if not np.all(np.isfinite([gamma_sub, gamma_film, gamma_int])):
                raise ValueError("derived gamma values are not finite")
        except Exception as exc:  # noqa: BLE001
            record_error("gamma", exc)
            gamma_sub = gamma_film = gamma_int = None
    valid = chain_valid and gamma_int is not None
    if not valid and not errors:
        errors.append("energy chain failed validity checks")
    errors = list(dict.fromkeys(errors))

    def cell(value: float | None, digits: int = 6) -> str:
        return "" if value is None else f"{value:.{digits}f}"

    ood_result = max(relaxations.values(), key=lambda result: result.max_ood_score)
    ood_triggered = any(result.ood_triggered for result in relaxations.values())
    committee_enabled = any(
        bool(result.metadata.get("committee_enabled", False)) for result in relaxations.values()
    )
    monitoring_modes = sorted(
        {str(result.metadata.get("ood_monitoring", "unknown")) for result in relaxations.values()}
    )
    all_energies_finite = all(
        value is not None and np.isfinite(value) for value in required_energies
    ) and all(result.energies_finite for result in relaxations.values())
    converged = all(result.converged for result in relaxations.values())
    e_slab_sub, e_slab_film, e_interface = energies

    row = {
        "rank": 1,
        "interface_id": candidate.interface_id,
        "substrate": candidate.substrate_formula,
        "film": candidate.film_formula,
        "sub_termination": candidate.substrate_termination,
        "film_termination": candidate.film_termination,
        "area_A2": f"{area:.4f}",
        "n_atoms": candidate.total_atoms,
        "E_bulk_sub_ev_per_atom": cell(
            None if e_bulk_sub is None else e_bulk_sub / len(substrate_atoms)
        ),
        "E_bulk_film_ev_per_atom": cell(
            None if e_bulk_film is None else e_bulk_film / len(film_atoms)
        ),
        "E_bulk_sub_coh_ev_per_atom": cell(
            None if e_bulk_sub_coh is None else e_bulk_sub_coh / len(substrate_atoms)
        ),
        "E_bulk_film_coh_ev_per_atom": cell(
            None if e_bulk_film_coh is None else e_bulk_film_coh / len(film_atoms)
        ),
        "E_slab_sub_ev": cell(e_slab_sub),
        "E_slab_film_ev": cell(e_slab_film),
        "E_interface_ev": cell(e_interface),
        "gamma_surf_sub_eV_A2": cell(gamma_sub),
        "gamma_surf_film_eV_A2": cell(gamma_film),
        "gamma_eV_per_A2": cell(gamma_int),
        "gamma_J_per_m2": cell(None if gamma_int is None else gamma_int * EV_A2_TO_J_M2, digits=4),
        "valid": str(valid),
        "converged": str(converged),
        "energies_finite": str(all_energies_finite),
        "slab_sub_valid": str(relax_sub.valid),
        "slab_film_valid": str(relax_film.valid),
        "interface_valid": str(relax_interface.valid),
        "invalid_reasons": "; ".join(errors),
        "committee_enabled": str(committee_enabled),
        "ood_monitoring": ",".join(monitoring_modes),
        "ood_score": f"{ood_result.max_ood_score:.6f}",
        "ood_step": str(ood_result.ood_step),
        "ood_triggered": str(ood_triggered),
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
        f"E_bulk  sub (Cu): {cell(e_bulk_sub, 4) or 'NA':>10} eV total | "
        f"{cell(None if e_bulk_sub is None else e_bulk_sub / len(substrate_atoms), 4) or 'NA':>8} "
        "eV/atom"
    )
    print(
        f"strain in-plane : sub {float(candidate.strain_substrate):+.4%} | "
        f"film {float(candidate.strain_film):+.4%}"
    )
    if all(value is not None for value in required_energies):
        print(
            f"E_bulk coherent : sub {e_bulk_sub_coh / len(substrate_atoms):8.4f} eV/atom "
            f"({(e_bulk_sub_coh - e_bulk_sub) / len(substrate_atoms):+.5f} vs pristine) | "
            f"film {e_bulk_film_coh / len(film_atoms):8.4f} eV/atom "
            f"({(e_bulk_film_coh - e_bulk_film) / len(film_atoms):+.5f} vs pristine)"
        )
    else:
        print("E_bulk coherent : NA (invalid bulk reference)")
    print(f"E_slab  sub     : {cell(e_slab_sub, 4) or 'NA':>10} eV (valid={relax_sub.valid})")
    print(f"E_slab film     : {cell(e_slab_film, 4) or 'NA':>10} eV (valid={relax_film.valid})")
    print(
        f"E_interface     : {cell(e_interface, 4) or 'NA':>10} eV (valid={relax_interface.valid})"
    )
    if valid:
        print(
            f"gamma_surf Cu   : {gamma_sub:10.4f} eV/A^2 = {gamma_sub * EV_A2_TO_J_M2:7.3f} J/m^2"
        )
        print(
            f"gamma_surf Ni   : {gamma_film:10.4f} eV/A^2 = {gamma_film * EV_A2_TO_J_M2:7.3f} J/m^2"
        )
        print(
            f"gamma_int       : {gamma_int:10.4f} eV/A^2 = {gamma_int * EV_A2_TO_J_M2:7.3f} J/m^2"
        )
    else:
        print("gamma_surf Cu   : NA (invalid energy chain)")
        print("gamma_surf Ni   : NA (invalid energy chain)")
        print("gamma_int       : NA (invalid energy chain)")
        print("invalid reasons : " + "; ".join(errors))
    print(
        f"OOD             : monitoring={row['ood_monitoring']} "
        f"committee_enabled={committee_enabled} score={ood_result.max_ood_score:.3f} "
        f"triggered={ood_triggered}"
    )
    print(f"\nWrote {csv_path}, metadata.yaml and structures to {outdir}/")
    if args.backend == "dummy":
        print("note: dummy energies are not physical; the numbers only prove the chain.")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
