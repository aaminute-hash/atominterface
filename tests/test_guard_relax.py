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
    WatchResult,
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


# ------------------------------------------------------------------
# Committee wiring
#
# CommitteeWatcher turns multi-model force disagreement into an OOD
# signal.  The mocks below follow the Calc style of
# ``dummy_calc_factory``: an ASE-style object whose ``get_forces(atoms)``
# returns a fixed array.
# ------------------------------------------------------------------
class _CalcWith:
    """ASE-style calculator returning one fixed force array."""

    def __init__(self, forces: np.ndarray):
        self._forces = np.asarray(forces, dtype=float)

    def get_potential_energy(self, atoms):
        return 0.0

    def get_forces(self, atoms):
        return self._forces

    def get_stress(self, atoms):
        return np.zeros(6)


class _BrokenCalc:
    """Calculator whose force evaluation always fails."""

    def get_potential_energy(self, atoms):
        return 0.0

    def get_forces(self, atoms):
        raise RuntimeError("model exploded")

    def get_stress(self, atoms):
        return np.zeros(6)


class _AttachedOnlyCalc:
    """Refuse direct calls; compute only once ASE has attached us.

    Exercises the ``atoms.copy(); atoms.calc = calc`` fallback in
    ``CommitteeWatcher._committee_forces`` for non-standard members.
    """

    def __init__(self, forces: np.ndarray):
        self._forces = np.asarray(forces, dtype=float)

    def get_potential_energy(self, atoms):
        return 0.0

    def get_forces(self, atoms):
        if getattr(atoms, "calc", None) is not self:
            raise TypeError("not attached to these atoms")
        return self._forces

    def get_stress(self, atoms):
        return np.zeros(6)


def test_committee_watcher_without_calculators_scores_zero():
    """No committee registered: a lone array cannot disagree."""
    w = CommitteeWatcher()
    r = w(make_atoms(4), np.zeros((4, 3)))
    assert r.score == 0.0
    assert r.reason == ""


def test_committee_watcher_identical_members_agree():
    """Members returning the same forces: zero disagreement."""
    atoms = make_atoms(4)
    w = CommitteeWatcher([_CalcWith(np.zeros((4, 3))), _CalcWith(np.zeros((4, 3)))])
    r = w(atoms, np.zeros((4, 3)))
    assert r.score == pytest.approx(0.0)


def test_committee_watcher_extreme_disagreement_saturates():
    """Primary zeros vs a member at 10 eV/A: mean disagreement >> 1."""
    atoms = make_atoms(4)
    w = CommitteeWatcher([_CalcWith(np.ones((4, 3)) * 10.0)])
    r = w(atoms, np.zeros((4, 3)))
    assert r.score == 1.0
    assert r.reason == "committee_disagreement"
    # std of {0, 10} is 5 per component -> 5*sqrt(3) mean norm.
    assert r.meta["disagreement"] == pytest.approx(5.0 * np.sqrt(3))


def test_committee_watcher_three_members_partial_disagreement():
    """Two members agree, a third is a mild outlier: score in (0, 1)."""
    atoms = make_atoms(4)
    w = CommitteeWatcher([_CalcWith(np.zeros((4, 3))), _CalcWith(np.ones((4, 3)) * 0.5)])
    r = w(atoms, np.zeros((4, 3)))
    assert 0.0 < r.score < 1.0
    assert r.reason == "committee_disagreement"


def test_committee_watcher_skips_failing_member():
    """A member that raises is skipped; the rest still aggregate."""
    atoms = make_atoms(4)
    w = CommitteeWatcher([_CalcWith(np.ones((4, 3)) * 10.0), _BrokenCalc()])
    r = w(atoms, np.zeros((4, 3)))
    assert r.score == 1.0  # aggregated over primary + surviving member


def test_committee_watcher_all_members_failing_scores_zero():
    """Every member fails: fall back to the primary forces alone."""
    atoms = make_atoms(4)
    w = CommitteeWatcher([_BrokenCalc(), _BrokenCalc()])
    r = w(atoms, np.zeros((4, 3)))
    assert r.score == 0.0


def test_committee_watcher_skips_wrong_shape_member():
    """A member with the wrong force layout is skipped, not fatal."""
    atoms = make_atoms(4)
    w = CommitteeWatcher([_CalcWith(np.ones((3, 3)))])  # 3 atoms for a 4-atom cell
    r = w(atoms, np.zeros((4, 3)))
    assert r.score == 0.0


def test_committee_watcher_falls_back_to_ase_attachment():
    """Non-standard members are driven through an attached atoms copy."""
    atoms = make_atoms(4)
    w = CommitteeWatcher([_AttachedOnlyCalc(np.ones((4, 3)) * 10.0)])
    r = w(atoms, np.zeros((4, 3)))
    assert r.score == 1.0
    assert r.reason == "committee_disagreement"


def test_guarded_relaxer_mounts_committee_on_default_chain():
    """committee_calculators appends a CommitteeWatcher to the default chain."""
    relaxer = GuardedRelaxer(
        calculator=get_calculator("dummy"),
        committee_calculators=[_CalcWith(np.zeros((5, 3))), _CalcWith(np.zeros((5, 3)))],
    )
    assert len(relaxer.watchers) == 3
    assert isinstance(relaxer.watchers[0], NeighborWatcher)
    assert isinstance(relaxer.watchers[1], ForceNormWatcher)
    assert isinstance(relaxer.watchers[2], CommitteeWatcher)


def test_guarded_relaxer_single_committee_member_still_works():
    """One member is enough: primary forces + member = two arrays."""
    atoms = make_atoms(5)
    relaxer = GuardedRelaxer(
        calculator=get_calculator("dummy"),
        committee_calculators=[_CalcWith(np.ones((5, 3)) * 10.0)],
    )
    committee = relaxer.watchers[2]
    assert isinstance(committee, CommitteeWatcher)
    r = committee(atoms, np.zeros((5, 3)))
    assert r.score == 1.0


def test_guarded_relaxer_explicit_watchers_chain_not_extended():
    """An explicit watchers chain is respected verbatim: no committee."""
    relaxer = GuardedRelaxer(
        calculator=get_calculator("dummy"),
        watchers=[NeighborWatcher()],
        committee_calculators=[_CalcWith(np.zeros((5, 3)))] * 2,
    )
    assert len(relaxer.watchers) == 1
    assert not any(isinstance(w, CommitteeWatcher) for w in relaxer.watchers)


def test_guarded_relaxer_quarantines_on_committee_disagreement(tmp_path):
    """Full guard path: normal primary forces, dissenting committee member.

    Both heuristic watchers see nothing wrong (small forces, no
    overlaps); the guard fires purely on the model disagreement.
    ``stop_on="all"`` averages the chain, and the reason join lists
    every non-empty reason -- the two geometry watchers stay silent, so
    the reported reason is the committee's alone.
    """
    atoms = make_atoms(5)
    primary = _CalcWith(np.ones((5, 3)) * 0.01)
    dissenter = _CalcWith(np.ones((5, 3)) * 100.0)
    relaxer = GuardedRelaxer(
        calculator=primary,
        committee_calculators=[dissenter],
        max_steps=5,
        ood_threshold=0.3,  # mean of (0, 0, 1) is 1/3
        stop_on="all",
        quarantine_dir=tmp_path,
    )
    result = relaxer.run(atoms)
    assert result.ood_triggered is True
    assert result.ood_step >= 0
    assert any("committee_disagreement" in reason for reason in result.reasons)
    assert result.quarantine_path != ""
    assert list(tmp_path.iterdir())  # DFT input written for the cluster


def test_from_calculators_wires_remaining_as_committee():
    """from_calculators: primary drives, the rest form the committee."""
    c1 = _CalcWith(np.zeros((5, 3)))
    c2 = _CalcWith(np.zeros((5, 3)))
    c3 = _CalcWith(np.zeros((5, 3)))
    relaxer = GuardedRelaxer.from_calculators([c1, c2, c3])
    assert relaxer._base is c1
    committee = relaxer.watchers[2]
    assert isinstance(committee, CommitteeWatcher)
    assert committee.calculators == [c2, c3]


def test_from_calculators_honours_primary_index():
    c1 = _CalcWith(np.zeros((5, 3)))
    c2 = _CalcWith(np.zeros((5, 3)))
    relaxer = GuardedRelaxer.from_calculators([c1, c2], primary=1)
    assert relaxer._base is c2
    assert relaxer.watchers[2].calculators == [c1]


def test_from_calculators_single_calculator_has_no_committee():
    """A lone calculator leaves the default two-watcher chain alone."""
    relaxer = GuardedRelaxer.from_calculators([_CalcWith(np.zeros((5, 3)))])
    assert len(relaxer.watchers) == 2
    assert not any(isinstance(w, CommitteeWatcher) for w in relaxer.watchers)


def test_from_calculators_explicit_chain_disables_committee():
    c1 = _CalcWith(np.zeros((5, 3)))
    c2 = _CalcWith(np.zeros((5, 3)))
    relaxer = GuardedRelaxer.from_calculators([c1, c2], watchers=[NeighborWatcher()])
    assert len(relaxer.watchers) == 1


def test_from_calculators_validates_arguments():
    with pytest.raises(ValueError, match="at least one"):
        GuardedRelaxer.from_calculators([])
    with pytest.raises(ValueError, match="out of range"):
        GuardedRelaxer.from_calculators([_CalcWith(np.zeros((5, 3)))], primary=3)


# ------------------------------------------------------------------
# Aggregate reason alignment
#
# Regression tests for the reasons-index bug: _aggregate used to
# index a reasons list compacted to non-empty entries with the argmax
# watcher's position, so any empty-reason watcher ahead of the winner
# shifted the reported reason -- and an all-empty prefix raised
# IndexError, which run() swallowed as relax_exception.
# ------------------------------------------------------------------
def test_aggregate_reason_aligns_with_forcenorm_watcher():
    """scores=[0, 0.9]: the reason must come from the ForceNormWatcher."""
    atoms = make_atoms(5)
    relaxer = GuardedRelaxer(calculator=get_calculator("dummy"), stop_on="any")
    # per-atom force norm 90 -> ForceNormWatcher score 90/100 = 0.9
    forces = np.ones((5, 3)) * (90.0 / np.sqrt(3.0))
    watch = relaxer._aggregate(atoms, forces)
    assert watch.score == pytest.approx(0.9)
    assert "max_force_norm" in watch.reason


def test_aggregate_argmax_watcher_without_reason():
    """A loud low-score watcher ahead of a silent winner must not shift it."""
    atoms = make_atoms(5)

    class _LoudLowWatcher:
        def __call__(self, atoms, forces):
            return WatchResult(score=0.3, reason="low_but_loud")

    class _SilentHighWatcher:
        def __call__(self, atoms, forces):
            return WatchResult(score=0.95, reason="")

    relaxer = GuardedRelaxer(
        calculator=get_calculator("dummy"),
        watchers=[_LoudLowWatcher(), _SilentHighWatcher()],
        stop_on="any",
    )
    watch = relaxer._aggregate(atoms, np.zeros((5, 3)))
    assert watch.score == pytest.approx(0.95)
    assert watch.reason == ""


def test_guarded_relaxer_flags_high_forces_without_index_error(tmp_path):
    """End-to-end on the pre-existing crash path (no committee involved).

    ForceNormWatcher alone over threshold used to abort the relaxation
    with relax_exception=IndexError; the guard must quarantine instead.
    """
    atoms = make_atoms(5)
    relaxer = GuardedRelaxer(
        calculator=_CalcWith(np.ones((5, 3)) * 90.0),
        max_steps=5,
        ood_threshold=0.5,
        quarantine_dir=tmp_path,
    )
    result = relaxer.run(atoms)
    assert result.ood_triggered is True
    assert result.ood_step >= 0
    assert any("max_force_norm" in reason for reason in result.reasons)
    assert result.quarantine_path != ""
    assert list(tmp_path.iterdir())


def test_guarded_relaxer_committee_argmax_on_any_stop_on(tmp_path):
    """stop_on='any' (the default): committee as the winning watcher.

    The committee scores highest by design, so this is the path the
    default configuration exercises -- it must report the committee
    reason and quarantine, not die on reasons[idx].
    """
    atoms = make_atoms(5)
    relaxer = GuardedRelaxer(
        calculator=_CalcWith(np.ones((5, 3)) * 0.01),
        committee_calculators=[_CalcWith(np.ones((5, 3)) * 100.0)],
        max_steps=5,
        ood_threshold=0.8,
        quarantine_dir=tmp_path,
    )
    result = relaxer.run(atoms)
    assert result.ood_triggered is True
    assert result.ood_step >= 0
    assert result.max_ood_score == pytest.approx(1.0)
    assert result.reasons == ["committee_disagreement"]
    assert result.quarantine_path != ""
    assert list(tmp_path.iterdir())
