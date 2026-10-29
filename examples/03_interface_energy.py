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

Every candidate kept by ``--max-candidates`` gets its own ledger row: the
whole chain runs per candidate and each one is judged on its own, so a
failure in the middle of the list is recorded as a failed row instead of
silently dropping the rest.

Outputs (in --outdir, default outputs/energy_cu_ni):

  candidates.csv   one row per candidate, with energy-validity and OOD
                   diagnostics; failed candidates keep their row and
                   structures but leave the gamma cells empty.
                   ``min_thickness_A`` records the shared --min-thickness,
                   while ``min_thickness_substrate_A`` and
                   ``min_thickness_film_A`` record the per-side values the
                   slabs were actually cut at (each side falls back to the
                   shared value when no override was given), so the layer
                   counts always match a recorded thickness.
  metadata.yaml    provenance: backend identity, seed, validity, artifact hash
  <rank>_*.extxyz  final interface and reference slabs, one set per candidate
  bulk_*.extxyz    bulk inputs (shared by every candidate)

Gamma values are emitted only when every pristine/coherent bulk reference
energy is finite and both reference slabs plus the interface pass finite
energy/force, convergence and OOD checks. Invalid candidates keep their
diagnostic row and structures, leave gamma cells empty, and make the run
exit non-zero while the successful candidates still ship. A zero
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
    python examples/03_interface_energy.py --backend mace --dtype float64 --fmax 0.01
    python examples/03_interface_energy.py --min-thickness-substrate 18.0 --min-thickness-film 14.4
    python examples/03_interface_energy.py --sub-miller 1 1 1 --film-miller 1 1 1

Fail-closed demonstration (non-zero exit, empty gamma cells):
    python examples/03_interface_energy.py --backend dummy --max-steps 1 --fmax 1e-6
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from ase.io import write as ase_write
except Exception as exc:  # pragma: no cover - ASE is a hard dependency
    sys.exit(f"[interface-energy] ASE is required: {exc}")

if TYPE_CHECKING:
    # Import-time only: the ledger fields below are annotated with it, and
    # ``from __future__ import annotations`` keeps the annotation a string,
    # so the demo never pays the relax-package import at module scope.
    from atominterface.relax import RelaxResult

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

#: Per-candidate ledger status. ``failed`` rows keep their diagnostics but
#: never carry a gamma -- see :meth:`EnergyLedger.to_row`.
STATUS_OK = "ok"
STATUS_FAILED = "failed"

#: Gap (A) that separates two atomic planes when counting slab layers. Far
#: below the ~1.8 A (001) interlayer spacing of Cu/Ni and far above
#: relaxation noise, so the count is stable before and after relaxation.
LAYER_GAP_A = 0.75

CSV_FIELDS = [
    "rank",
    "interface_id",
    "substrate",
    "film",
    "sub_termination",
    "film_termination",
    "area_A2",
    "area_source",
    "area_sub_A2",
    "area_film_A2",
    "area_consistent",
    "n_atoms",
    "n_layers_sub",
    "n_layers_film",
    "vacuum_A",
    "min_thickness_A",
    "min_thickness_substrate_A",
    "min_thickness_film_A",
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
    "status",
    "error_reason",
    "converged",
    "energies_finite",
    "slab_sub_valid",
    "slab_film_valid",
    "interface_valid",
    "invalid_reasons",
    "reference_state",
    "committee_enabled",
    "ood_monitoring",
    "ood_score",
    "ood_step",
    "ood_triggered",
    "backend",
    "model_hash",
]


def _cell(value: float | None, digits: int = 6) -> str:
    """One ledger number as CSV text; ``None`` stays blank (fail-closed)."""
    return "" if value is None else f"{value:.{digits}f}"


@dataclass
class EnergyLedger:
    """One candidate's complete energy ledger.

    The ledger used to be a handful of loose locals inside ``main()``,
    which is why only the first candidate could ever be reported.  As a
    record it can be produced per candidate and judged on its own:
    ``status`` is ``ok`` or ``failed``, and a failed ledger keeps every
    diagnostic while leaving all gamma fields ``None`` so :meth:`to_row`
    writes them blank rather than as a plausible-looking zero.

    ``relax_*`` carry the ``RelaxResult`` objects out of the evaluation
    purely so the relaxed structures can be written; they are not part of
    the CSV contract.
    """

    rank: int = 0
    interface_id: str = ""
    substrate: str = ""
    film: str = ""
    sub_termination: str = ""
    film_termination: str = ""

    area_a2: float = 0.0
    area_source: str = ""
    area_sub_a2: float | None = None
    area_film_a2: float | None = None
    area_consistent: bool | None = None
    n_atoms: int = 0
    n_layers_sub: int = 0
    n_layers_film: int = 0
    vacuum_a: float = 0.0
    #: Shared --min-thickness (what a run without per-side overrides uses).
    min_thickness_a: float = 0.0
    #: Resolved per-side thicknesses from ``config.slab_thicknesses()``:
    #: the override when one was given, the shared value when it was not.
    #: Recorded next to ``n_layers_sub`` / ``n_layers_film`` so a row can
    #: never show a layer count generated from a different thickness than
    #: the shared column reports.
    min_thickness_substrate_a: float = 0.0
    min_thickness_film_a: float = 0.0

    n_bulk_sub: int = 0
    n_bulk_film: int = 0
    e_bulk_sub: float | None = None
    e_bulk_film: float | None = None
    e_bulk_sub_coh: float | None = None
    e_bulk_film_coh: float | None = None
    e_slab_sub: float | None = None
    e_slab_film: float | None = None
    e_interface: float | None = None

    gamma_sub: float | None = None
    gamma_film: float | None = None
    gamma_int: float | None = None
    gamma_int_j_m2: float | None = None
    reference_state: str = ""

    converged: bool = False
    energies_finite: bool = False
    slab_sub_valid: bool = False
    slab_film_valid: bool = False
    interface_valid: bool = False
    committee_enabled: bool = False
    ood_monitoring: str = ""
    ood_score: float = 0.0
    ood_step: int = -1
    ood_triggered: bool = False

    status: str = STATUS_FAILED
    error_reason: str = ""

    backend: str = ""
    model_hash: str = ""

    relax_sub: RelaxResult | None = None
    relax_film: RelaxResult | None = None
    relax_interface: RelaxResult | None = None

    def to_row(self) -> dict[str, str]:
        """CSV row for this candidate; blank wherever a value is missing."""
        return {
            "rank": str(self.rank),
            "interface_id": self.interface_id,
            "substrate": self.substrate,
            "film": self.film,
            "sub_termination": self.sub_termination,
            "film_termination": self.film_termination,
            "area_A2": f"{self.area_a2:.4f}",
            "area_source": self.area_source,
            "area_sub_A2": _cell(self.area_sub_a2, 4),
            "area_film_A2": _cell(self.area_film_a2, 4),
            "area_consistent": "" if self.area_consistent is None else str(self.area_consistent),
            "n_atoms": str(self.n_atoms),
            "n_layers_sub": str(self.n_layers_sub),
            "n_layers_film": str(self.n_layers_film),
            "vacuum_A": f"{self.vacuum_a:.4f}",
            "min_thickness_A": f"{self.min_thickness_a:.4f}",
            "min_thickness_substrate_A": f"{self.min_thickness_substrate_a:.4f}",
            "min_thickness_film_A": f"{self.min_thickness_film_a:.4f}",
            "E_bulk_sub_ev_per_atom": _cell(
                None
                if self.e_bulk_sub is None or not self.n_bulk_sub
                else self.e_bulk_sub / self.n_bulk_sub
            ),
            "E_bulk_film_ev_per_atom": _cell(
                None
                if self.e_bulk_film is None or not self.n_bulk_film
                else self.e_bulk_film / self.n_bulk_film
            ),
            "E_bulk_sub_coh_ev_per_atom": _cell(
                None
                if self.e_bulk_sub_coh is None or not self.n_bulk_sub
                else self.e_bulk_sub_coh / self.n_bulk_sub
            ),
            "E_bulk_film_coh_ev_per_atom": _cell(
                None
                if self.e_bulk_film_coh is None or not self.n_bulk_film
                else self.e_bulk_film_coh / self.n_bulk_film
            ),
            "E_slab_sub_ev": _cell(self.e_slab_sub),
            "E_slab_film_ev": _cell(self.e_slab_film),
            "E_interface_ev": _cell(self.e_interface),
            "gamma_surf_sub_eV_A2": _cell(self.gamma_sub),
            "gamma_surf_film_eV_A2": _cell(self.gamma_film),
            "gamma_eV_per_A2": _cell(self.gamma_int),
            "gamma_J_per_m2": _cell(self.gamma_int_j_m2, digits=4),
            "valid": str(self.status == STATUS_OK),
            "status": self.status,
            "error_reason": self.error_reason,
            "converged": str(self.converged),
            "energies_finite": str(self.energies_finite),
            "slab_sub_valid": str(self.slab_sub_valid),
            "slab_film_valid": str(self.slab_film_valid),
            "interface_valid": str(self.interface_valid),
            "invalid_reasons": self.error_reason,
            "reference_state": self.reference_state,
            "committee_enabled": str(self.committee_enabled),
            "ood_monitoring": self.ood_monitoring,
            "ood_score": f"{self.ood_score:.6f}",
            "ood_step": str(self.ood_step),
            "ood_triggered": str(self.ood_triggered),
            "backend": self.backend,
            "model_hash": self.model_hash,
        }


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
        "--dtype",
        default="float32",
        choices=("float32", "float64"),
        help=(
            "calculator precision forwarded to the backend. float32 (default) "
            "matches mace-torch's own default; float64 is the double-precision "
            "reference channel that separates physics from float32 rounding in "
            "validation studies."
        ),
    )
    parser.add_argument(
        "--min-thickness",
        type=float,
        default=14.4,
        help=(
            "minimum slab thickness in A, applied to both sides unless a "
            "per-side override is given. 14.4 A cuts 8 Cu / 10 Ni layers, "
            "the plateau point of the thickness scan in "
            "docs/validation_cu_ni_001.md; the previous 6.0 A default cut "
            "only 4 Ni layers and reported gamma_Ni = 1.979 J/m^2 and "
            "gamma_int = 0.041 J/m^2 against converged values of 1.9189 and "
            "0.0590 -- 3%% and 30%% off, respectively."
        ),
    )
    parser.add_argument(
        "--min-thickness-substrate",
        type=float,
        default=None,
        help=(
            "minimum substrate (Cu) slab thickness in A. Defaults to None, "
            "which falls back to the shared --min-thickness."
        ),
    )
    parser.add_argument(
        "--min-thickness-film",
        type=float,
        default=None,
        help=(
            "minimum film (Ni) slab thickness in A. Defaults to None, which "
            "falls back to the shared --min-thickness."
        ),
    )
    parser.add_argument("--vacuum", type=float, default=15.0, help="vacuum padding in A.")
    parser.add_argument("--max-candidates", type=int, default=1, help="keep at most N candidates.")
    parser.add_argument(
        "--sub-miller",
        type=int,
        nargs=3,
        metavar=("H", "K", "L"),
        default=SUBSTRATE_MILLER,
        help=(
            "Miller indices of the substrate surface, as three integers. "
            "Default %(default)s, the documented Cu(001) reference; "
            "'--sub-miller 1 1 1' switches the substrate side to (111)."
        ),
    )
    parser.add_argument(
        "--film-miller",
        type=int,
        nargs=3,
        metavar=("H", "K", "L"),
        default=FILM_MILLER,
        help=(
            "Miller indices of the film surface, as three integers. "
            "Default %(default)s, the documented Ni(001) reference."
        ),
    )
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


def _make_calculator(backend: str, device: str, dtype: str):
    """One calculator factory shared by every energy in the ledger.

    The seed is honoured by the dummy backend (deterministic stand-in
    forces); real backends are deterministic by construction and ignore
    it.  ``dtype`` is forwarded to the backend: the mace factory validates
    it, and the dummy/chgnet factories accept and ignore it, so the default
    ``float32`` leaves every backend's own default path untouched.
    """
    from atominterface.backends import get_calculator

    return get_calculator(backend, device=device, seed=SEED, dtype=dtype)


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


def _count_layers(atoms) -> int:
    """Number of atomic planes stacked along the slab normal.

    Counted by splitting the sorted z coordinates wherever the gap
    exceeds :data:`LAYER_GAP_A`; see that constant for the margin on
    both sides.  The reference boards are counted before relaxation, so
    the number is a property of the cut and not of the optimizer.
    """
    z = np.unique(np.round(np.asarray(atoms.positions)[:, 2], 3))
    if z.size == 0:
        return 0
    return 1 + int(np.count_nonzero(np.diff(z) > LAYER_GAP_A))


def _area_fields(candidate) -> tuple[str, float, float, bool | None]:
    """Interface-area provenance for one candidate's ledger.

    ``area_source`` / ``area_sub_A2`` / ``area_film_A2`` /
    ``area_consistent`` are contributed by the ``fix/interface-area-policy``
    branch.  Until that lands the reads fall back to ``candidate.area`` and
    leave the consistency flag empty, so this branch does not depend on the
    other PR's merge order -- merging it makes the real values appear here
    with no further change.
    """

    def _read(name: str, default):
        value = getattr(candidate, name, None)
        if value is None:
            value = candidate.metadata.get(name)
        return default if value is None else value

    area_sub = float(_read("area_sub_A2", candidate.area))
    area_film = float(_read("area_film_A2", candidate.area))
    consistent = _read("area_consistent", None)
    source = str(_read("area_source", "candidate.area"))
    return source, area_sub, area_film, None if consistent is None else bool(consistent)


def _match_for(candidate):
    """Rebuild the lattice match a candidate was assembled from.

    ``build_all()`` enumerates match x substrate-termination x
    film-termination and truncates at ``max_candidates``, so the Nth
    candidate is *not* in general the Nth lattice match: with more than one
    termination per side several candidates share one match, and ``zip``
    would silently pair the rest against the wrong supercell.  The pairing
    has to come from the candidate itself, and ``_to_candidate`` already
    copies every descriptor the reference slabs need into the metadata
    (``substrate_supercell``, ``film_supercell``, ``match_area_ang2``,
    ``mismatch_stretch``, ``mismatch_shear``).
    """
    from atominterface.builders.coherent import _MatchInfo

    meta = candidate.metadata
    return _MatchInfo(
        match_area=float(meta["match_area_ang2"]),
        substrate_supercell=meta["substrate_supercell"],
        film_supercell=meta["film_supercell"],
        strain=float(meta["mismatch_stretch"]),
        shear_strain=float(meta["mismatch_shear"]),
    )


def _build_candidates(substrate, film, args: argparse.Namespace):
    """Every coherent candidate kept by ``--max-candidates``, plus the config.

    The whole point of the ledger is that it covers this whole list, not
    just its head.
    """
    from atominterface.builders import CoherentInterfaceBuilder, CoherentInterfaceConfig

    config = CoherentInterfaceConfig(
        substrate_miller=tuple(args.sub_miller),
        film_miller=tuple(args.film_miller),
        max_area=MAX_AREA,
        gap=GAP,
        min_thickness=args.min_thickness,
        min_thickness_substrate=args.min_thickness_substrate,
        min_thickness_film=args.min_thickness_film,
        vacuum_thickness=args.vacuum,
        max_candidates=args.max_candidates,
    )
    builder = CoherentInterfaceBuilder(substrate=substrate, film=film, config=config)
    return builder.build(), config


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


def _reference_state(e_bulk_sub, e_bulk_film, e_bulk_sub_coh, e_bulk_film_coh) -> str:
    """Which bulk reference the ledger's termination energies rest on.

    ``strained-coherent`` is the reference that removes the ``n *
    dE_strain`` term (see :func:`_coherent_bulk_energy`); ``pristine``
    means only the unstrained bulk energies are available, so no coherent
    correction could be applied.  Empty when neither pair survived.
    """
    if e_bulk_sub_coh is not None and e_bulk_film_coh is not None:
        return "strained-coherent"
    if e_bulk_sub is not None and e_bulk_film is not None:
        return "pristine"
    return ""


def _evaluate_candidate(  # noqa: PLR0912, PLR0915
    *,
    rank: int,
    candidate,
    match,
    config,
    substrate,
    film,
    substrate_atoms,
    film_atoms,
    calculator,
    e_bulk_sub: float | None,
    e_bulk_film: float | None,
    setup_errors: list[str],
    identity: str,
    identity_valid: bool,
    args: argparse.Namespace,
    outdir: Path,
) -> EnergyLedger:
    """Run the whole energy chain for one candidate.

    Every candidate is judged on its own: a failure anywhere in its chain
    marks *this* ledger ``failed`` with an explicit ``error_reason`` and
    leaves its gamma fields ``None``, while the other candidates still
    ship.  ``setup_errors`` holds the run-wide failures (calculator, bulk
    single points, model identity) and is folded into every candidate's
    reason, because those genuinely invalidate all of them.
    """
    from atominterface.physics import EV_A2_TO_J_M2
    from atominterface.relax import RelaxResult

    errors: list[str] = list(setup_errors)

    # Resolved per-side thicknesses: each side is the per-side override
    # when one was given and the shared --min-thickness when it was not,
    # which is exactly the value the reference slabs are cut at.
    sub_thickness, film_thickness = config.slab_thicknesses()

    def record_error(stage: str, exc: Exception) -> None:
        message = f"{stage}: {type(exc).__name__}: {exc}"
        errors.append(message)
        print(f"invalid: {message}")

    ledger = EnergyLedger(
        rank=rank,
        interface_id=candidate.interface_id,
        substrate=candidate.substrate_formula,
        film=candidate.film_formula,
        sub_termination=candidate.substrate_termination,
        film_termination=candidate.film_termination,
        area_a2=float(candidate.area),
        n_atoms=int(candidate.total_atoms),
        vacuum_a=float(config.vacuum_thickness),
        min_thickness_a=float(config.min_thickness),
        min_thickness_substrate_a=float(sub_thickness),
        min_thickness_film_a=float(film_thickness),
        n_bulk_sub=len(substrate_atoms),
        n_bulk_film=len(film_atoms),
        e_bulk_sub=e_bulk_sub,
        e_bulk_film=e_bulk_film,
        backend=args.backend,
        model_hash=identity,
    )
    ledger.area_source, ledger.area_sub_a2, ledger.area_film_a2, ledger.area_consistent = (
        _area_fields(candidate)
    )

    e_bulk_sub_coh = None
    e_bulk_film_coh = None
    if calculator is not None:
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
    ledger.e_bulk_sub_coh = e_bulk_sub_coh
    ledger.e_bulk_film_coh = e_bulk_film_coh

    references_ready = True
    try:
        reference_sub, reference_film = _reference_slabs(substrate, film, config, match, candidate)
    except Exception as exc:  # noqa: BLE001
        record_error("reference_slabs", exc)
        references_ready = False
        reference_sub = substrate_atoms.copy()
        reference_film = film_atoms.copy()
    ledger.n_layers_sub = _count_layers(reference_sub)
    ledger.n_layers_film = _count_layers(reference_film)

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

    required_energies = (e_bulk_sub, e_bulk_film, e_bulk_sub_coh, e_bulk_film_coh)
    relaxation_valid = all(result.valid for result in relaxations.values())
    chain_valid = (
        references_ready
        and identity_valid
        and all(value is not None and np.isfinite(value) for value in required_energies)
        and relaxation_valid
        and all(value is not None and np.isfinite(value) for value in energies)
    )

    gamma_sub = None
    gamma_film = None
    gamma_int = None
    # ``chain_valid`` already guarantees both coherent bulk energies are
    # present; spelling the checks out lets the type checker narrow them
    # from ``float | None`` to ``float`` for the call below.
    if chain_valid and e_bulk_sub_coh is not None and e_bulk_film_coh is not None:
        try:
            _area, gamma_sub, gamma_film, gamma_int = _gammas(
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
    valid = bool(chain_valid and gamma_int is not None)
    if not valid and not errors:
        errors.append("energy chain failed validity checks")

    ledger.e_slab_sub, ledger.e_slab_film, ledger.e_interface = energies
    ledger.gamma_sub = gamma_sub
    ledger.gamma_film = gamma_film
    ledger.gamma_int = gamma_int
    ledger.gamma_int_j_m2 = None if gamma_int is None else gamma_int * EV_A2_TO_J_M2
    ledger.reference_state = _reference_state(
        e_bulk_sub, e_bulk_film, e_bulk_sub_coh, e_bulk_film_coh
    )
    ledger.status = STATUS_OK if valid else STATUS_FAILED
    ledger.error_reason = "; ".join(dict.fromkeys(errors))

    ledger.slab_sub_valid = bool(relax_sub.valid)
    ledger.slab_film_valid = bool(relax_film.valid)
    ledger.interface_valid = bool(relax_interface.valid)
    ledger.converged = all(result.converged for result in relaxations.values())
    ledger.energies_finite = all(
        value is not None and np.isfinite(value) for value in required_energies
    ) and all(result.energies_finite for result in relaxations.values())
    ledger.relax_sub = relax_sub
    ledger.relax_film = relax_film
    ledger.relax_interface = relax_interface

    ood_worst = max(relaxations.values(), key=lambda result: result.max_ood_score)
    ledger.ood_score = float(ood_worst.max_ood_score)
    ledger.ood_step = int(ood_worst.ood_step)
    ledger.ood_triggered = any(result.ood_triggered for result in relaxations.values())
    ledger.committee_enabled = any(
        bool(result.metadata.get("committee_enabled", False)) for result in relaxations.values()
    )
    ledger.ood_monitoring = ",".join(
        sorted(
            {
                str(result.metadata.get("ood_monitoring", "unknown"))
                for result in relaxations.values()
            }
        )
    )
    return ledger


def _print_ledger(ledger: EnergyLedger, candidate) -> None:
    """Console rendering of one candidate's ledger."""
    from atominterface.physics import EV_A2_TO_J_M2

    print(f"\n== energy ledger | rank {ledger.rank} ({ledger.status}) ==")
    print(
        f"E_bulk  sub (Cu): {_cell(ledger.e_bulk_sub, 4) or 'NA':>10} eV total | "
        f"{_cell(None if ledger.e_bulk_sub is None else ledger.e_bulk_sub / ledger.n_bulk_sub, 4) or 'NA':>8} "
        "eV/atom"
    )
    print(
        f"strain in-plane : sub {float(candidate.strain_substrate):+.4%} | "
        f"film {float(candidate.strain_film):+.4%}"
    )
    # Spelled out as one conjunction (rather than ``all(...)`` over a
    # tuple) so the type checker narrows each field to ``float`` inside
    # the branch that divides by the bulk atom counts.
    if (
        ledger.e_bulk_sub is not None
        and ledger.e_bulk_film is not None
        and ledger.e_bulk_sub_coh is not None
        and ledger.e_bulk_film_coh is not None
    ):
        print(
            f"E_bulk coherent : sub {ledger.e_bulk_sub_coh / ledger.n_bulk_sub:8.4f} eV/atom "
            f"({(ledger.e_bulk_sub_coh - ledger.e_bulk_sub) / ledger.n_bulk_sub:+.5f} vs pristine) | "
            f"film {ledger.e_bulk_film_coh / ledger.n_bulk_film:8.4f} eV/atom "
            f"({(ledger.e_bulk_film_coh - ledger.e_bulk_film) / ledger.n_bulk_film:+.5f} vs pristine)"
        )
    else:
        print("E_bulk coherent : NA (invalid bulk reference)")
    print(
        f"E_slab  sub     : {_cell(ledger.e_slab_sub, 4) or 'NA':>10} eV "
        f"(valid={ledger.slab_sub_valid})"
    )
    print(
        f"E_slab film     : {_cell(ledger.e_slab_film, 4) or 'NA':>10} eV "
        f"(valid={ledger.slab_film_valid})"
    )
    print(
        f"E_interface     : {_cell(ledger.e_interface, 4) or 'NA':>10} eV "
        f"(valid={ledger.interface_valid})"
    )
    # All three gammas are set (or cleared) together; guarding on each
    # one keeps the multiplication below off a ``None`` operand.
    if (
        ledger.gamma_int is not None
        and ledger.gamma_sub is not None
        and ledger.gamma_film is not None
    ):
        print(
            f"gamma_surf Cu   : {ledger.gamma_sub:10.4f} eV/A^2 = "
            f"{ledger.gamma_sub * EV_A2_TO_J_M2:7.3f} J/m^2"
        )
        print(
            f"gamma_surf Ni   : {ledger.gamma_film:10.4f} eV/A^2 = "
            f"{ledger.gamma_film * EV_A2_TO_J_M2:7.3f} J/m^2"
        )
        print(
            f"gamma_int       : {ledger.gamma_int:10.4f} eV/A^2 = "
            f"{ledger.gamma_int * EV_A2_TO_J_M2:7.3f} J/m^2"
        )
    else:
        print("gamma_surf Cu   : NA (invalid energy chain)")
        print("gamma_surf Ni   : NA (invalid energy chain)")
        print("gamma_int       : NA (invalid energy chain)")
    if ledger.error_reason:
        print("invalid reasons : " + ledger.error_reason)
    print(
        f"OOD             : monitoring={ledger.ood_monitoring} "
        f"committee_enabled={ledger.committee_enabled} score={ledger.ood_score:.3f} "
        f"triggered={ledger.ood_triggered}"
    )


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

    # Annotated local: when this script is type-checked on its own the
    # package import resolves to ``Any``, and the annotation keeps the
    # declared ``-> str`` honest instead of leaking ``Any`` outward.
    digest: str = hash_bytes(path.read_bytes())
    return digest


def _write_outputs(
    *,
    outdir: Path,
    ledgers: list[EnergyLedger],
    bulk_sub_measured,
    bulk_film_measured,
    identity: str,
    backend_version: str,
    args: argparse.Namespace,
) -> Path:
    """Write candidates.csv, the relaxed structures and metadata.yaml.

    Every candidate gets a row and its own set of relaxed structures; the
    extxyz names carry the rank prefix so a multi-candidate run cannot
    overwrite them one another.  The bulk single points are shared by all
    candidates and stay unprefixed.
    """
    from atominterface.provenance import build_meta, write_metadata

    csv_path = outdir / "candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(ledger.to_row() for ledger in ledgers)

    for ledger in ledgers:
        prefix = f"{ledger.rank:02d}_"
        # A ledger that reaches this writer always carries its three
        # relaxations; the ``None`` default only covers a ledger that was
        # never evaluated, which has no relaxed structures to write.
        for result, label in (
            (ledger.relax_interface, "interface"),
            (ledger.relax_sub, "slab_sub"),
            (ledger.relax_film, "slab_film"),
        ):
            if result is None:
                continue
            _write(outdir / f"{prefix}{label}_relaxed.extxyz", result.atoms)
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
            "valid": all(ledger.status == STATUS_OK for ledger in ledgers),
            "invalid_reasons": "; ".join(
                ledger.error_reason for ledger in ledgers if ledger.error_reason
            ),
            "committee_enabled": any(ledger.committee_enabled for ledger in ledgers),
            "ood_monitoring": ",".join(
                sorted(
                    {
                        mode
                        for ledger in ledgers
                        for mode in ledger.ood_monitoring.split(",")
                        if mode
                    }
                )
            ),
        },
    )
    write_metadata(meta, outdir / "metadata.yaml")
    return csv_path


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912, PLR0915
    """Run the Cu/Ni interface energy ledger; returns a process exit code."""
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

    candidates, config = _build_candidates(substrate, film, args)
    if not candidates:
        print("no coherent candidate under the current tolerances")
        return 1
    print(f"candidates: {len(candidates)} kept (--max-candidates {args.max_candidates})")

    # Run-wide failures: they invalidate every candidate, so they are
    # folded into each ledger's error_reason rather than aborting the run.
    setup_errors: list[str] = []

    def record_error(stage: str, exc: Exception) -> None:
        message = f"{stage}: {type(exc).__name__}: {exc}"
        setup_errors.append(message)
        print(f"invalid: {message}")

    calculator = None
    try:
        calculator = _make_calculator(args.backend, args.device, args.dtype)
    except Exception as exc:  # noqa: BLE001
        record_error("calculator", exc)
    print(f"backend  : {args.backend} (device {args.device})")

    e_bulk_sub = None
    e_bulk_film = None
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

    identity = ""
    backend_version = ""
    identity_valid = True
    try:
        identity, backend_version = _identity(args)
    except Exception as exc:  # noqa: BLE001
        record_error("model_identity", exc)
        identity_valid = False

    ledgers: list[EnergyLedger] = []
    for rank, candidate in enumerate(candidates, start=1):
        print(f"candidate {rank}/{len(candidates)}: {candidate.interface_id}")
        print(
            f"          area={candidate.area:.2f} A^2, {candidate.total_atoms} atoms, "
            f"mismatch={candidate.metadata['mismatch']:.4f}"
        )
        ledger = _evaluate_candidate(
            rank=rank,
            candidate=candidate,
            match=_match_for(candidate),
            config=config,
            substrate=substrate,
            film=film,
            substrate_atoms=substrate_atoms,
            film_atoms=film_atoms,
            calculator=calculator,
            e_bulk_sub=e_bulk_sub,
            e_bulk_film=e_bulk_film,
            setup_errors=setup_errors,
            identity=identity,
            identity_valid=identity_valid,
            args=args,
            outdir=outdir,
        )
        ledgers.append(ledger)
        _print_ledger(ledger, candidate)

    csv_path = _write_outputs(
        outdir=outdir,
        ledgers=ledgers,
        bulk_sub_measured=bulk_sub_measured,
        bulk_film_measured=bulk_film_measured,
        identity=identity,
        backend_version=backend_version,
        args=args,
    )

    failed = [ledger.rank for ledger in ledgers if ledger.status != STATUS_OK]
    print(f"\nWrote {csv_path}, metadata.yaml and structures to {outdir}/")
    print(f"ledger   : {len(ledgers) - len(failed)}/{len(ledgers)} candidate(s) valid")
    if failed:
        print(f"invalid  : candidate rank(s) {', '.join(str(rank) for rank in failed)}")
    if args.backend == "dummy":
        print("note: dummy energies are not physical; the numbers only prove the chain.")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
