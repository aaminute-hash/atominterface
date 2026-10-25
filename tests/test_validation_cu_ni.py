"""Cu/Ni(001) instrument validation: the coherent bulk reference.

These tests pin the physics the convergence scan depends on, not just
the arithmetic.  The defect they guard against: a coherent slab is
built on the *interface* in-plane lattice, so the film side carries the
mismatch strain (2.58% for Cu/Ni) while the substrate side does not.
Referencing a strained slab against the pristine bulk leaves
``n * dE_strain`` inside the termination energy -- a term that grows
with layer count -- so gamma_Ni drifted monotonically (2.00 -> 2.10
J/m^2 from 6 to 14 Ni layers) instead of converging.
``_coherent_bulk_energy`` removes it.

The cheap cases run on synthetic calculators (or the dummy backend) so
they stay in the fast suite; the MACE-marked cases reproduce the
physical numbers recorded in ``docs/validation_cu_ni_001.md``.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator

ROOT = Path(__file__).parents[1]
EXAMPLE03 = ROOT / "examples" / "03_interface_energy.py"
DATA = ROOT / "examples" / "data"

#: MACE-MP-0 medium values recorded in docs/validation_cu_ni_001.md (J/m^2).
GAMMA_CU_RANGE = (1.40, 1.60)
GAMMA_NI_RANGE = (1.85, 2.05)
#: The film side is stretched onto Cu; the substrate side is strain-free.
STRAIN_FILM_EXPECTED = 0.025823
#: Largest tolerated gamma_Ni gap between 4 and 8 Ni layers.  Before the
#: coherent reference the same pair differed by ~0.06 J/m^2 and the series
#: kept climbing.
GAMMA_NI_DRIFT_TOLERANCE = 0.05


def _load_example03():
    """Import the example driver by path (it is not an importable package).

    Importing runs the module's top-level ``warnings.filterwarnings``
    calls, which would otherwise silence warnings for the rest of the
    session; the filters are snapshotted and restored around the import.
    """
    saved = warnings.filters[:]
    try:
        spec = importlib.util.spec_from_file_location("_example03_validation", EXAMPLE03)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        warnings.filters[:] = saved
    return module


@pytest.fixture(scope="module")
def example03():
    """The example 03 driver, imported once per test module."""
    return _load_example03()


# ------------------------------------------------------------------
# synthetic calculators: the fitting logic, without a real backend
# ------------------------------------------------------------------
class _CellOnlyCalculator(Calculator):
    """Energy that depends on the cell only, never on the positions.

    ``E = curvature * (sz - centre)**2 + floor + slope * eps_in``, where
    ``sz`` is the out-of-plane strain and ``eps_in`` the in-plane strain,
    both measured against the pristine cell handed to the constructor.
    A one-dimensional parabola in ``sz`` is exactly what
    ``_coherent_bulk_energy`` fits, so the recovered minimum is known
    analytically.
    """

    implemented_properties = ["energy"]

    def __init__(self, reference, curvature, centre, floor, slope, **kwargs):
        super().__init__(**kwargs)
        self.reference = reference
        self.curvature = curvature
        self.centre = centre
        self.floor = floor
        self.slope = slope

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes)
        cell = self.atoms.get_cell()
        sz = float(cell[2][2]) / float(self.reference[2][2]) - 1.0
        eps_in = float(cell[0][0]) / float(self.reference[0][0]) - 1.0
        self.results = {
            "energy": self.curvature * (sz - self.centre) ** 2 + self.floor + self.slope * eps_in
        }


def _factory(reference, curvature=2.5, centre=-0.03, floor=-7.5, slope=0.0):
    """Callable factory matching the ``_coherent_bulk_energy`` contract."""

    def make(atoms):
        return _CellOnlyCalculator(reference, curvature, centre, floor, slope, atoms=atoms)

    return make


def test_coherent_bulk_energy_recovers_the_parabola_minimum(example03):
    """A quadratic E(sz) profile is fitted back to its exact minimum."""
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    reference = atoms.get_cell().copy()
    floor, centre = -7.5, -0.03
    # -0.03 sits inside the -0.08 .. 0.02 sampling grid, so the 9-point
    # parabola fit interpolates the minimum instead of extrapolating it.
    grid_start, grid_stop, grid_n = -0.08, 0.02, 9
    step = (grid_stop - grid_start) / (grid_n - 1)
    nearest = min(abs(grid_start + i * step - centre) for i in range(grid_n))
    assert nearest < step
    value = example03._coherent_bulk_energy(
        atoms, 0.0, _factory(reference, centre=centre, floor=floor)
    )
    assert value == pytest.approx(floor, abs=1e-9)


def test_coherent_bulk_energy_applies_the_in_plane_strain(example03):
    """The slab's in-plane strain reaches the reference cell."""
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    reference = atoms.get_cell().copy()
    strain, slope, floor = STRAIN_FILM_EXPECTED, 4.0, -7.5
    value = example03._coherent_bulk_energy(
        atoms, strain, _factory(reference, floor=floor, slope=slope)
    )
    # The in-plane term is linear in the strain and constant along sz, so
    # the fitted minimum is the parabola floor shifted by slope * strain.
    assert value == pytest.approx(floor + slope * strain, abs=1e-9)


def test_coherent_bulk_energy_survives_a_cell_independent_calculator(example03):
    """A flat profile returns the constant instead of dividing by zero curvature.

    The dummy backend's energy does not depend on the cell, which made
    the naive parabola fit return NaN; the degenerate branch is what
    keeps the wiring self-test runnable.
    """
    from atominterface.backends import get_calculator

    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    value = example03._coherent_bulk_energy(atoms, 0.0, get_calculator("dummy"))
    assert math.isfinite(value)
    assert value == pytest.approx(0.001 * len(atoms))


def test_coherent_bulk_energy_recovers_a_minimum_off_the_grid(example03):
    """The fit interpolates: it can undercut every sampled point.

    A centre that falls between two grid samples can only be reached by
    the parabola, so a pass here proves the driver fits rather than
    picking the cheapest sampled cell.
    """
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    reference = atoms.get_cell().copy()
    curvature, centre, floor = 2.5, -0.031, -3.25
    grid_start, grid_stop, grid_n = -0.08, 0.02, 9
    step = (grid_stop - grid_start) / (grid_n - 1)
    sampled = [curvature * (grid_start + i * step - centre) ** 2 + floor for i in range(grid_n)]
    value = example03._coherent_bulk_energy(
        atoms, 0.0, _factory(reference, curvature=curvature, centre=centre, floor=floor)
    )
    assert value == pytest.approx(floor, abs=1e-9)
    assert value < min(sampled) - 1e-12


# ------------------------------------------------------------------
# cheap: the Cu/Ni mismatch the coherent reference has to absorb
# ------------------------------------------------------------------
def _cu_ni_bulks():
    """Load the Cu/Ni bulks shipped next to the example driver."""
    from pymatgen.core import Structure

    return (
        Structure.from_file(str(DATA / "Cu_fcc.vasp")),
        Structure.from_file(str(DATA / "Ni_fcc.vasp")),
    )


def _first_candidate(substrate, film, *, min_thickness: float):
    """Build the single coherent Cu/Ni(001) candidate the ledger uses."""
    from atominterface.builders import CoherentInterfaceBuilder, CoherentInterfaceConfig

    config = CoherentInterfaceConfig(
        substrate_miller=(0, 0, 1),
        film_miller=(0, 0, 1),
        max_area=100.0,
        gap=2.0,
        min_thickness=min_thickness,
        vacuum_thickness=15.0,
        max_candidates=1,
    )
    candidates = CoherentInterfaceBuilder(substrate=substrate, film=film, config=config).build()
    assert candidates, "no coherent Cu/Ni(001) candidate under the demo tolerances"
    return candidates[0]


def test_cu_ni_candidate_puts_the_mismatch_on_the_film():
    """Cu/Ni(001): substrate unstrained, film stretched by 2.58%.

    Pins the strain split that makes the coherent reference necessary --
    if both sides were strain-free the pristine bulk would be the right
    reference and the layer-count drift could not happen.
    """
    pytest.importorskip("pymatgen")
    substrate, film = _cu_ni_bulks()
    candidate = _first_candidate(substrate, film, min_thickness=7.2)
    assert float(candidate.strain_substrate) == pytest.approx(0.0, abs=1e-9)
    assert float(candidate.strain_film) == pytest.approx(STRAIN_FILM_EXPECTED, abs=5e-4)


# ------------------------------------------------------------------
# MACE: the physical reference split and the thickness plateau
# ------------------------------------------------------------------
@pytest.mark.needs_mace
def test_coherent_reference_carries_the_film_strain_energy(example03):
    """With MACE the film reference sits above pristine, the substrate at it.

    Per docs/validation_cu_ni_001.md the coherent Cu bulk is within
    ~1 meV/atom of pristine (no strain to carry) while the coherent Ni
    bulk is ~+11 meV/atom above it (2.58% tension).  That difference is
    the term whose layer-count multiple used to leak into gamma_Ni.
    """
    pytest.importorskip("mace")
    substrate, film = _cu_ni_bulks()
    calculator = example03._make_calculator("mace", "cpu")
    substrate_atoms = example03._to_ase(substrate)
    film_atoms = example03._to_ase(film)

    e_sub, _ = example03._single_point(substrate_atoms, calculator)
    e_film, _ = example03._single_point(film_atoms, calculator)
    e_sub_coh = example03._coherent_bulk_energy(substrate_atoms, 0.0, calculator)
    e_film_coh = example03._coherent_bulk_energy(film_atoms, STRAIN_FILM_EXPECTED, calculator)

    delta_sub = (e_sub_coh - e_sub) / len(substrate_atoms)
    delta_film = (e_film_coh - e_film) / len(film_atoms)
    assert delta_sub == pytest.approx(0.0, abs=2e-3)
    assert 0.005 < delta_film < 0.020


def _run_example03(outdir: Path, *, min_thickness: float) -> dict:
    """Run the driver end to end and return its single-row ledger."""
    subprocess.run(
        [
            sys.executable,
            str(EXAMPLE03),
            "--backend",
            "mace",
            "--device",
            "cpu",
            "--min-thickness",
            str(min_thickness),
            "--vacuum",
            "15.0",
            "--outdir",
            str(outdir),
        ],
        check=True,
        capture_output=True,
    )
    with (outdir / "candidates.csv").open(newline="", encoding="utf-8") as handle:
        return dict(next(csv.DictReader(handle)))


@pytest.fixture(scope="module")
def mace_runs(tmp_path_factory):
    """Two end-to-end MACE runs (thin and thick slabs), shared by the tests.

    Each is a full guarded relaxation of both reference slabs and the
    interface, so the pair is computed once for the whole module rather
    than per test.
    """
    pytest.importorskip("mace")
    base = tmp_path_factory.mktemp("cu_ni_validation")
    return {
        thickness: _run_example03(base / f"t{thickness}", min_thickness=thickness)
        for thickness in (7.2, 14.4)
    }


def _gamma_jm2(row: dict, key: str) -> float:
    """Convert a recorded eV/A^2 column to J/m^2 with the ledger's own factor."""
    scale = float(row["gamma_J_per_m2"]) / float(row["gamma_eV_per_A2"])
    return scale * float(row[key])


@pytest.mark.needs_mace
def test_gamma_values_stay_inside_the_documented_window(mace_runs):
    """Both thicknesses land inside the window reported in the validation note."""
    for thickness, row in mace_runs.items():
        gamma_cu = _gamma_jm2(row, "gamma_surf_sub_eV_A2")
        gamma_ni = _gamma_jm2(row, "gamma_surf_film_eV_A2")
        assert GAMMA_CU_RANGE[0] <= gamma_cu <= GAMMA_CU_RANGE[1], (thickness, gamma_cu)
        assert GAMMA_NI_RANGE[0] <= gamma_ni <= GAMMA_NI_RANGE[1], (thickness, gamma_ni)


@pytest.mark.needs_mace
def test_gamma_ni_no_longer_drifts_with_slab_thickness(mace_runs):
    """gamma_Ni at two slab thicknesses agrees; the pre-fix series kept climbing.

    This is the regression test for the defect itself: the termination
    energy used to retain ``n * dE_strain``, so gamma_Ni grew with the
    layer count instead of plateauing.
    """
    gamma_thin = _gamma_jm2(mace_runs[7.2], "gamma_surf_film_eV_A2")
    gamma_thick = _gamma_jm2(mace_runs[14.4], "gamma_surf_film_eV_A2")
    assert abs(gamma_thick - gamma_thin) < GAMMA_NI_DRIFT_TOLERANCE
    # The reference slabs really do differ in layer count, otherwise the
    # comparison would be vacuous.
    assert int(mace_runs[7.2]["n_atoms"]) < int(mace_runs[14.4]["n_atoms"])
