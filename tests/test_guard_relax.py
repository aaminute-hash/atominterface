"""Tests for the guard layer and GuardedRelaxer."""

import numpy as np
import pytest
from ase.atoms import Atoms

from atominterface.backends import get_calculator, list_backends
from atominterface.physics import (
    ev_per_a3_to_gpa,
    gpa_to_ev_per_a3,
    rotation_matrix,
    transform_forces,
    transform_stress_voigt,
)
from atominterface.relax.guarded_calc import GuardedRelaxer, RelaxResult
from atominterface.relax.quarantine import cut_cluster, write_dft_input
from atominterface.relax.watchers import (
    CommitteeWatcher,
    ForceNormWatcher,
    NeighborWatcher,
)

_GRID = np.stack(np.meshgrid([2.5, 5.0, 7.5], [2.5, 5.0, 7.5], [2.5, 5.0, 7.5]), axis=-1).reshape(
    -1, 3
)


def make_atoms(n=5, cell=None, seed: int = 7):
    """Well-separated atoms with reproducible positions.

    Positions used to come from an unseeded ``np.random.rand``. With six
    atoms in an 8 A box, two of them occasionally land closer than the
    0.6 A overlap threshold that `NeighborWatcher` uses, so
    `test_neighbor_watcher_passes_normal` failed roughly one CI run in
    twenty for no reason.  Picking distinct points off a 2.5 A grid
    (plus a small seeded jitter) keeps the input varied while making the
    minimum separation a property of the fixture rather than of luck.
    """
    if cell is None:
        cell = np.eye(3) * 10.0
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(_GRID), size=n, replace=False)
    positions = _GRID[picks] + rng.normal(0.0, 0.2, (n, 3))
    return Atoms(numbers=[6] * n, positions=positions, cell=cell, pbc=True)


def dummy_calc_factory():
    """Return a simple ASE-calc-like object with controllable forces."""
    forces = np.random.rand(5, 3) * 0.01

    class Calc:
        def get_potential_energy(self, atoms):
            return 0.0

        def get_forces(self, atoms):
            return forces

        def get_stress(self, atoms):
            return np.zeros(6)

    return Calc()


# ------------------------------------------------------------------
# Physics unit tests
# ------------------------------------------------------------------
def test_ev_per_a3_gpa_roundtrip():
    val = 1.0  # eV/A^3
    assert abs(ev_per_a3_to_gpa(val) - 160.21766208) < 1e-6
    back = gpa_to_ev_per_a3(ev_per_a3_to_gpa(val))
    assert abs(back - 1.0) < 1e-9


def test_rotation_matrix_is_orthogonal():
    R = rotation_matrix("z", 60.0)
    assert abs(R @ R.T - np.eye(3)).max() < 1e-12
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_stress_transforms_as_rank2():
    s = np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6])  # Voigt
    R = rotation_matrix("z", 90.0)
    sp = transform_stress_voigt(s, R)
    # Recover as matrix and compare with R S R^T
    m = np.array([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])
    expect = R @ m @ R.T
    voigt_expect = np.array(
        [expect[0, 0], expect[1, 1], expect[2, 2], expect[1, 2], expect[0, 2], expect[0, 1]]
    )
    assert np.allclose(sp, voigt_expect, atol=1e-12)


def test_forces_transform_as_vectors():
    F = np.array([[1.0, 0.0, 0.0]])
    R = rotation_matrix("z", 90.0)
    Fp = transform_forces(F, R)
    assert np.allclose(Fp, [[0.0, 1.0, 0.0]], atol=1e-12)


# ------------------------------------------------------------------
# Watchers
# ------------------------------------------------------------------
def test_neighbor_watcher_flags_overlap():
    atoms = make_atoms(4)
    atoms.positions[0] = atoms.positions[1]  # exact overlap
    w = NeighborWatcher(min_dist_frac=0.6)
    r = w(atoms, atoms.arrays.get("forces", np.zeros((4, 3))))
    assert r.score > 0.0
    assert "min_pairwise_dist" in r.reason


def test_neighbor_watcher_passes_normal():
    atoms = make_atoms(6)
    w = NeighborWatcher(min_dist_frac=0.6)
    r = w(atoms, np.zeros((6, 3)))
    assert r.score == 0.0
    assert r.reason == ""


def test_force_norm_watcher():
    atoms = make_atoms(4)
    big = np.ones((4, 3)) * 100.0
    w = ForceNormWatcher(max_norm=50.0)
    r = w(atoms, big)
    assert r.score > 0.0
    small = np.ones((4, 3)) * 1e-6
    r2 = w(atoms, small)
    assert r2.score == 0.0


def test_committee_watcher_needs_two_arrays():
    a = np.ones((5, 3))
    b = np.ones((5, 3)) * 1.1
    r = CommitteeWatcher.from_force_arrays([a, b])
    assert r.score >= 0.0
    r0 = CommitteeWatcher.from_force_arrays([a])
    assert r0.score == 0.0


# ------------------------------------------------------------------
# Backend registry
# ------------------------------------------------------------------
def test_list_backends_contains_dummy():
    caps = list_backends()
    assert "dummy" in caps
    assert caps["dummy"].supports_stress is True


def test_get_calculator_dummy_returns_callable():
    calc = get_calculator("dummy")
    atoms = make_atoms()
    e = calc(atoms).get_potential_energy(atoms)
    assert isinstance(e, float)
    f = calc(atoms).get_forces(atoms)
    assert f.shape == (5, 3)


def test_get_calculator_unknown_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        get_calculator("no-such-backend")


# ------------------------------------------------------------------
# Quarantine
# ------------------------------------------------------------------
def test_cut_cluster_returns_subset():
    atoms = make_atoms(8)
    cluster = cut_cluster(atoms, 0, radius=4.0)
    assert len(cluster) >= 1
    assert len(cluster) <= len(atoms)


def test_write_dft_input_creates_poscar(tmp_path):
    atoms = make_atoms(4)
    from atominterface.relax.quarantine import QuarantineRecord

    rec = QuarantineRecord(run_id="abc", step=3, center_index=0, ood_score=0.9)
    p = write_dft_input(atoms, rec, directory=tmp_path, fmt="vasp")
    assert p.exists()
    assert p.suffix == ".POSCAR"


# ------------------------------------------------------------------
# GuardedRelaxer
# ------------------------------------------------------------------
def test_relax_runs_with_dummy_and_records_steps():
    atoms = make_atoms()
    calc = get_calculator("dummy")
    relaxer = GuardedRelaxer(calculator=calc, max_steps=5, ood_threshold=0.99)
    result = relaxer.run(atoms)
    assert isinstance(result, RelaxResult)
    assert result.n_steps > 0


def test_relax_flags_overlapping_atoms():
    atoms = make_atoms(4)
    atoms.positions[1] = atoms.positions[0]  # force overlap
    calc = get_calculator("dummy")
    relaxer = GuardedRelaxer(calculator=calc, max_steps=50, ood_threshold=0.01)
    result = relaxer.run(atoms)
    assert result.ood_triggered is True
    assert result.quarantine_path != ""


def test_relax_with_emt_converges_or_reports():
    try:
        from ase.calculators.emt import EMT
    except ImportError:
        pytest.skip("ase[emt] not available")
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]], cell=np.eye(3) * 10, pbc=True)
    relaxer = GuardedRelaxer(calculator=EMT(), max_steps=100, fmax=0.1, ood_threshold=0.99)
    result = relaxer.run(atoms)
    assert result.n_steps > 0
