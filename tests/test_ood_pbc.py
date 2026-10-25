"""PBC-correct OOD signals and quarantine cuts.

Regression home for four defects that all lived in the same blind spot:
a cartesian distance where a minimum-image one was required.

* ``NeighborWatcher`` measured bare cartesian distances, so an overlap
  straddling a periodic boundary was invisible, *and* scaled its
  threshold by the mean cell-vector length -- which meant the very same
  local geometry scored differently depending on how much vacuum a slab
  carried.
* ``NeighborWatcher`` returned no ``per_atom``, so the guard had nothing
  to localise on and every quarantine cluster was anchored on index 0.
* ``CommitteeWatcher`` answered a bare ``score=0.0`` for "no committee
  configured", "every member failed" and "all members agree" alike.
* ``cut_cluster`` ignored its own ``include_images`` flag and never
  filled ``QuarantineRecord.cluster_indices``.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.atoms import Atoms

from atominterface.relax.proximity import (
    minimum_image_distances,
    nearest_image_vectors,
    nearest_neighbor_distances,
    pairwise_distance_matrix,
    pairwise_min_distance,
)
from atominterface.relax.quarantine import QuarantineRecord, cut_cluster
from atominterface.relax.watchers import (
    COMMITTEE_DEGRADED,
    COMMITTEE_DISABLED,
    COMMITTEE_OK,
    COMMITTEE_UNAVAILABLE,
    CommitteeWatcher,
    NeighborWatcher,
)

# Carbon-covalent: 0.6 * (0.76 + 0.76)
CARBON_THRESHOLD = 0.912


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
def _pair(positions, cell, pbc=True) -> Atoms:
    """Two carbon atoms at the given positions."""
    return Atoms("C2", positions=np.asarray(positions, dtype=float), cell=cell, pbc=pbc)


def _short_bond_at_3_and_4(gap: float = 0.3, depth: float = 24.0) -> Atoms:
    """Six well-separated carbons, except atoms 3 and 4, ``gap`` apart.

    The pair straddles the x boundary: their cartesian separation is
    ``12 - gap`` (11.7 A for the default), which no cartesian check can
    read as a contact.  The other four atoms sit at x >= 3, so the
    pair's location is not an accident of ``gap`` and every remaining
    distance stays above 4.1 A -- well clear of the 0.912 A carbon
    threshold -- for any ``gap`` used in these tests.

    Atoms 3 and 4 form each other's nearest neighbour, so their
    ``per_atom`` entries tie; ``argmax`` then resolves to the lower
    index, 3, which is what makes the localisation assertions
    deterministic.
    """
    positions = [
        [3.0, 2.0, 6.0],
        [3.0, 8.0, 6.0],
        [8.0, 2.0, 6.0],
        [12.0 - gap / 2.0, 5.0, 6.0],  # index 3
        [gap / 2.0, 5.0, 6.0],  # index 4, image of 3 across x
        [8.0, 8.0, 6.0],
    ]
    return Atoms("C6", positions=positions, cell=[12.0, 12.0, depth], pbc=True)


class _CalcWith:
    """ASE-style calculator returning one fixed force array."""

    def __init__(self, forces: np.ndarray) -> None:
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


# ------------------------------------------------------------------
# proximity: minimum-image distances
# ------------------------------------------------------------------
def test_minimum_image_distances_see_a_neighbour_across_a_1d_boundary():
    atoms = _pair([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]], [10.0, 20.0, 20.0], pbc=[True, False, False])
    d = minimum_image_distances(atoms, 0)
    assert d[0] == np.inf  # self-distance excluded
    assert d[1] == pytest.approx(1.0)
    assert np.linalg.norm(atoms.positions[1] - atoms.positions[0]) == pytest.approx(9.0)


def test_minimum_image_distances_see_a_neighbour_across_a_2d_boundary():
    atoms = _pair([[0.5, 0.5, 0.0], [9.5, 9.5, 0.0]], [10.0, 10.0, 20.0], pbc=[True, True, False])
    assert minimum_image_distances(atoms, 0)[1] == pytest.approx(np.sqrt(2.0))


def test_minimum_image_distances_see_a_neighbour_across_a_3d_boundary():
    atoms = _pair([[0.5, 0.5, 0.5], [9.5, 9.5, 9.5]], [10.0, 10.0, 10.0], pbc=True)
    assert minimum_image_distances(atoms, 0)[1] == pytest.approx(np.sqrt(3.0))


def test_non_periodic_directions_are_not_wrapped():
    """pbc=[False, False, True]: only the z gap is allowed to wrap."""
    atoms = _pair([[0.5, 0.0, 0.5], [0.5, 9.0, 9.5]], [10.0, 20.0, 10.0], pbc=[False, False, True])
    # z wraps 9.0 -> 1.0; the 9 A y gap is real and must survive.
    assert minimum_image_distances(atoms, 0)[1] == pytest.approx(np.sqrt(9.0**2 + 1.0**2))
    assert np.linalg.norm(atoms.positions[1] - atoms.positions[0]) == pytest.approx(
        np.sqrt(9.0**2 + 9.0**2)
    )


def test_minimum_image_distances_in_a_skewed_cell():
    """A non-orthogonal cell must wrap in lattice space, not per axis.

    Cell a2 = (2, 4, 0) tilts into x, so wrapping the cartesian x
    component by the x cell length is wrong: it reports 0.8 A where the
    true minimum image is 1.2 A (one a2 translation, then -0.3 a1).
    """
    cell = np.array([[4.0, 0.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 10.0]])
    atoms = _pair([[0.0, 0.0, 0.0], [0.8, 4.0, 0.0]], cell, pbc=True)
    assert minimum_image_distances(atoms, 0)[1] == pytest.approx(1.2)
    # The naive per-axis wrap would have said 0.8 -- the difference is
    # what makes this test discriminating rather than tautological.
    assert abs(atoms.positions[1][0] - 4.0) != pytest.approx(1.2)


def test_pairwise_min_distance_reports_the_cross_boundary_pair():
    atoms = _pair([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]], [10.0, 20.0, 20.0], pbc=[True, False, False])
    distance, i, j = pairwise_min_distance(atoms)
    assert distance == pytest.approx(1.0)
    assert (i, j) == (0, 1)


def test_pairwise_distance_matrix_is_symmetric_with_an_infinite_diagonal():
    atoms = _short_bond_at_3_and_4()
    dm = pairwise_distance_matrix(atoms)
    # Exactly symmetric: an ulp asymmetry would randomise which end of a
    # tied bond the guard localises on.
    assert np.array_equal(dm, dm.T)
    assert np.all(np.isinf(np.diag(dm)))
    assert dm[3, 4] == pytest.approx(0.3)


def test_nearest_neighbor_distances_and_single_atom_edge_cases():
    atoms = _short_bond_at_3_and_4()
    nearest = nearest_neighbor_distances(atoms)
    assert nearest[3] == pytest.approx(0.3)
    assert nearest[4] == pytest.approx(0.3)
    assert nearest[0] > 1.8

    lone = Atoms("C", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 10.0, pbc=True)
    assert nearest_neighbor_distances(lone)[0] == np.inf
    assert minimum_image_distances(lone, 0)[0] == np.inf
    assert pairwise_min_distance(lone)[0] == np.inf


# ------------------------------------------------------------------
# NeighborWatcher: PBC and vacuum independence
# ------------------------------------------------------------------
def test_neighbor_watcher_flags_an_overlap_across_the_boundary():
    """Cartesian separation 11.7 A, true contact 0.3 A -> must fire."""
    atoms = _short_bond_at_3_and_4(gap=0.3)
    cartesian = np.linalg.norm(atoms.positions[4] - atoms.positions[3])
    assert cartesian > 11.0

    watch = NeighborWatcher()(atoms, np.zeros((len(atoms), 3)))
    assert watch.score > 0.0
    assert "min_pairwise_dist" in watch.reason
    assert watch.meta["min_dist"] == pytest.approx(0.3)
    assert set(watch.meta["min_pair"]) == {3, 4}


def test_neighbor_watcher_threshold_is_a_local_length_scale():
    """The threshold is covalent-radius based, so it never sees the cell."""
    atoms = _short_bond_at_3_and_4(gap=0.3)
    watch = NeighborWatcher()(atoms, np.zeros((len(atoms), 3)))
    assert watch.meta["threshold"] == pytest.approx(CARBON_THRESHOLD)


@pytest.mark.parametrize("vacuum", [10.0, 20.0, 40.0])
def test_neighbor_watcher_score_is_invariant_to_vacuum_thickness(vacuum):
    """Same local geometry, different vacuum -> identical verdict.

    The old threshold was ``mean(|a_i|) / 10 * 0.6``, i.e. it grew with
    the c vector: a 1.0 A carbon pair was clean in a 10 A cell and
    out-of-distribution in a 40 A one.  Both are asserted here against a
    vacuum-independent threshold.
    """
    atoms = _pair([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], [10.0, 10.0, vacuum], pbc=True)
    watch = NeighborWatcher()(atoms, np.zeros((2, 3)))
    # 1.0 A is above the 0.912 A carbon threshold either way.
    assert watch.score == 0.0
    assert watch.meta["threshold"] == pytest.approx(CARBON_THRESHOLD)


def test_old_cell_scaled_threshold_really_did_drift_with_vacuum():
    """Pin that the invariance test above is not vacuous.

    Reproduces the previous formula on the two extreme cells: it moves,
    which is exactly the defect the fix removes.
    """
    thresholds = []
    for vacuum in (10.0, 40.0):
        atoms = _pair([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], [10.0, 10.0, vacuum], pbc=True)
        scale = float(np.mean(np.linalg.norm(atoms.cell, axis=1))) / 10.0
        thresholds.append(max(scale * 0.6, 0.5))
    assert thresholds[0] != thresholds[1]
    assert thresholds[1] > 1.0  # the 1.0 A pair would have been flagged


def test_neighbor_watcher_score_is_invariant_to_a_cell_translation():
    atoms = _short_bond_at_3_and_4(gap=0.3)
    shifted = atoms.copy()
    shifted.positions += np.array([3.0, -4.0, 7.0])
    base = NeighborWatcher()(atoms, np.zeros((6, 3)))
    moved = NeighborWatcher()(shifted, np.zeros((6, 3)))
    assert moved.score == pytest.approx(base.score)
    assert moved.meta["min_dist"] == pytest.approx(base.meta["min_dist"])
    assert moved.meta["min_pair"] == base.meta["min_pair"]


@pytest.mark.parametrize("vacuum", [10.0, 20.0, 40.0])
def test_neighbor_watcher_clean_structure_stays_clean_at_any_vacuum(vacuum):
    atoms = _short_bond_at_3_and_4(gap=2.4, depth=vacuum)
    watch = NeighborWatcher()(atoms, np.zeros((6, 3)))
    assert watch.score == 0.0
    assert watch.reason == ""
    assert watch.meta["global_min_dist"] == pytest.approx(2.4)


def test_neighbor_watcher_straddles_the_threshold():
    """Just below the threshold fires, just above does not."""
    below = _pair([[0.0, 0.0, 0.0], [0.95 * CARBON_THRESHOLD, 0.0, 0.0]], np.eye(3) * 30.0)
    above = _pair([[0.0, 0.0, 0.0], [1.05 * CARBON_THRESHOLD, 0.0, 0.0]], np.eye(3) * 30.0)
    watch = NeighborWatcher()
    assert watch(below, np.zeros((2, 3))).score > 0.0
    assert watch(above, np.zeros((2, 3))).score == 0.0


def test_neighbor_watcher_single_atom_scores_zero():
    lone = Atoms("C", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 10.0, pbc=True)
    watch = NeighborWatcher()(lone, np.zeros((1, 3)))
    assert watch.score == 0.0
    assert watch.reason == ""


# ------------------------------------------------------------------
# NeighborWatcher: localisation
# ------------------------------------------------------------------
def test_neighbor_watcher_per_atom_marks_the_known_worst_atom():
    """per_atom must point at index 3, not at index 0."""
    atoms = _short_bond_at_3_and_4(gap=0.3)
    watch = NeighborWatcher()(atoms, np.zeros((6, 3)))

    assert watch.per_atom is not None
    assert len(watch.per_atom) == len(atoms)
    assert int(np.argmax(watch.per_atom)) == 3
    # The innocent atoms carry no signal at all.
    assert watch.per_atom[0] == 0.0
    assert watch.per_atom[2] == 0.0
    assert watch.per_atom[3] == watch.per_atom[4] > 0.0


def test_quarantine_cluster_anchors_on_the_known_worst_atom():
    """End-to-end localisation: worst atom -> cluster cut -> indices."""
    atoms = _short_bond_at_3_and_4(gap=0.3)
    watch = NeighborWatcher()(atoms, np.zeros((6, 3)))
    worst = int(np.argmax(watch.per_atom))
    assert worst == 3

    record = QuarantineRecord(
        run_id="pbc", step=0, center_index=worst, ood_score=float(watch.score)
    )
    cluster = cut_cluster(atoms, worst, radius=4.0, record=record)

    assert record.cluster_indices, "cluster_indices must be filled"
    assert worst in record.cluster_indices
    # The partner across the boundary is inside the ball too.
    assert 4 in record.cluster_indices
    assert len(cluster) == len(record.cluster_indices)
    assert cluster.info["cluster_indices"] == record.cluster_indices


# ------------------------------------------------------------------
# CommitteeWatcher: explicit availability states
# ------------------------------------------------------------------
def test_committee_without_calculators_is_reported_as_disabled():
    atoms = _short_bond_at_3_and_4(gap=2.4)
    watch = CommitteeWatcher()(atoms, np.zeros((6, 3)))
    assert watch.score == 0.0
    assert watch.reason == ""
    assert watch.meta["committee_state"] == COMMITTEE_DISABLED
    assert watch.meta["committee_requested"] == 0
    assert watch.meta["committee_members"] == 0


def test_committee_with_all_members_failing_is_reported_as_unavailable():
    atoms = _short_bond_at_3_and_4(gap=2.4)
    watch = CommitteeWatcher([_BrokenCalc(), _BrokenCalc()])(atoms, np.zeros((6, 3)))
    assert watch.score == 0.0
    assert watch.reason == "committee_unavailable"
    assert watch.meta["committee_state"] == COMMITTEE_UNAVAILABLE
    assert watch.meta["committee_requested"] == 2
    assert watch.meta["committee_members"] == 0
    assert watch.meta["committee_failed"] == 2


def test_committee_with_an_unusable_shape_is_reported_as_unavailable():
    atoms = _short_bond_at_3_and_4(gap=2.4)
    watch = CommitteeWatcher([_CalcWith(np.ones((3, 3)))])(atoms, np.zeros((6, 3)))
    assert watch.score == 0.0
    assert watch.meta["committee_state"] == COMMITTEE_UNAVAILABLE
    assert watch.meta["committee_failed"] == 1


def test_committee_with_a_dropped_member_is_reported_as_degraded():
    atoms = _short_bond_at_3_and_4(gap=2.4)
    watch = CommitteeWatcher([_CalcWith(np.ones((6, 3)) * 10.0), _BrokenCalc()])(
        atoms, np.zeros((6, 3))
    )
    assert watch.score == 1.0
    assert watch.reason == "committee_disagreement"
    assert watch.meta["committee_state"] == COMMITTEE_DEGRADED
    assert watch.meta["committee_requested"] == 2
    assert watch.meta["committee_members"] == 1
    assert watch.meta["committee_failed"] == 1


def test_committee_with_every_member_answering_is_reported_as_ok():
    atoms = _short_bond_at_3_and_4(gap=2.4)
    watch = CommitteeWatcher([_CalcWith(np.zeros((6, 3))), _CalcWith(np.zeros((6, 3)))])(
        atoms, np.zeros((6, 3))
    )
    assert watch.score == pytest.approx(0.0)
    assert watch.meta["committee_state"] == COMMITTEE_OK
    assert watch.meta["committee_requested"] == 2
    assert watch.meta["committee_members"] == 2
    assert watch.meta["committee_failed"] == 0
    # Consensus is silence: no disagreement to report.
    assert watch.reason == ""


def test_committee_failure_is_distinguishable_from_consensus():
    """Both score 0.0; only the state (and reason) tells them apart."""
    atoms = _short_bond_at_3_and_4(gap=2.4)
    agreeing = CommitteeWatcher([_CalcWith(np.zeros((6, 3)))])(atoms, np.zeros((6, 3)))
    broken = CommitteeWatcher([_BrokenCalc()])(atoms, np.zeros((6, 3)))

    assert agreeing.score == broken.score == 0.0
    assert agreeing.meta["committee_state"] == COMMITTEE_OK
    assert broken.meta["committee_state"] == COMMITTEE_UNAVAILABLE
    assert agreeing.reason == ""
    assert broken.reason == "committee_unavailable"


def test_committee_from_force_arrays_marks_a_lone_array_unavailable():
    lone = CommitteeWatcher.from_force_arrays([np.ones((4, 3))])
    assert lone.score == 0.0
    assert lone.meta["committee_state"] == COMMITTEE_UNAVAILABLE
    assert lone.meta["committee_members"] == 0


def test_committee_disagreement_keeps_its_reason_string():
    """The guard's contract: real disagreement is still reported as such."""
    atoms = _short_bond_at_3_and_4(gap=2.4)
    watch = CommitteeWatcher([_CalcWith(np.ones((6, 3)) * 10.0)])(atoms, np.zeros((6, 3)))
    assert watch.score == 1.0
    assert watch.reason == "committee_disagreement"
    assert watch.meta["committee_state"] == COMMITTEE_OK


# ------------------------------------------------------------------
# cut_cluster: periodic images and recorded indices
# ------------------------------------------------------------------
def test_cut_cluster_includes_a_neighbour_across_the_boundary():
    atoms = _short_bond_at_3_and_4(gap=0.3)
    cluster = cut_cluster(atoms, 3, radius=2.0)
    assert cluster.info["cluster_indices"] == [3, 4]

    without = cut_cluster(atoms, 3, radius=2.0, include_images=False)
    assert without.info["cluster_indices"] == [3]


def test_cut_cluster_unwraps_the_cluster_into_one_ball():
    """Every emitted atom sits within ``radius`` of the centre."""
    atoms = _short_bond_at_3_and_4(gap=0.3)
    radius = 2.0
    center = atoms.positions[3]
    cluster = cut_cluster(atoms, 3, radius=radius)

    assert cluster.info["cluster_indices"] == [3, 4]
    for position in cluster.positions:
        assert np.linalg.norm(position - center) <= radius + 1e-9
    # The partner is emitted as its periodic image at x + 0.3, not as the
    # cartesian atom 11.7 A away on the far side of the cell.
    assert cluster.positions[1][0] == pytest.approx(center[0] + 0.3)
    assert cluster.positions[1][0] != pytest.approx(atoms.positions[4][0])


def test_cut_cluster_fills_the_record_indices():
    """``cluster_indices`` is a field that used to stay permanently empty."""
    atoms = _short_bond_at_3_and_4(gap=0.3)
    record = QuarantineRecord(run_id="r", step=7, center_index=0, ood_score=0.5)
    assert record.cluster_indices == []  # the defect: never filled

    cluster = cut_cluster(atoms, 0, radius=6.0, record=record)
    # From atom 0: 3 and 4 are ~4.1-4.3 A away once the boundary is
    # honoured, atom 2 is 5.0, atom 1 is 6.0, atom 5 is 7.8.
    assert record.cluster_indices == [0, 1, 2, 3, 4]
    assert cluster.info["cluster_indices"] == record.cluster_indices
    assert record.cluster_indices == sorted(record.cluster_indices)
    assert len(cluster) == len(record.cluster_indices)


def test_cut_cluster_indices_map_back_to_the_parent():
    atoms = _short_bond_at_3_and_4(gap=2.4)
    cluster = cut_cluster(atoms, 2, radius=7.0)
    indices = cluster.info["cluster_indices"]
    assert 2 in indices
    assert np.array_equal(cluster.numbers, atoms.numbers[indices])


def test_cut_cluster_zero_radius_keeps_only_the_centre():
    atoms = _short_bond_at_3_and_4(gap=0.3)
    cluster = cut_cluster(atoms, 3, radius=0.0)
    assert len(cluster) == 1
    assert cluster.info["cluster_indices"] == [3]


def test_cut_cluster_rejects_an_out_of_range_centre():
    atoms = _short_bond_at_3_and_4(gap=2.4)
    with pytest.raises(IndexError, match="out of range"):
        cut_cluster(atoms, len(atoms), radius=4.0)
    with pytest.raises(IndexError, match="out of range"):
        cut_cluster(atoms, -1, radius=4.0)


def test_cut_cluster_is_identical_with_or_without_images_when_not_periodic():
    atoms = _short_bond_at_3_and_4(gap=2.4)
    atoms.pbc = False
    with_images = cut_cluster(atoms, 3, radius=5.0, include_images=True)
    without = cut_cluster(atoms, 3, radius=5.0, include_images=False)
    assert np.allclose(with_images.positions, without.positions)
    assert with_images.info["cluster_indices"] == without.info["cluster_indices"]


def test_cut_cluster_mixed_periodicity_ignores_the_open_direction():
    """pbc=[True, True, False]: no wrapping through the vacuum gap."""
    atoms = Atoms(
        "C2",
        positions=[[1.0, 1.0, 0.2], [1.0, 1.0, 9.8]],
        cell=[10.0, 10.0, 10.0],
        pbc=[True, True, False],
    )
    assert minimum_image_distances(atoms, 0)[1] == pytest.approx(9.6)
    assert len(cut_cluster(atoms, 0, radius=2.0)) == 1


def test_nearest_image_vectors_return_the_shortest_displacement():
    atoms = _pair([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]], [10.0, 20.0, 20.0], pbc=[True, False, False])
    vectors, lengths = nearest_image_vectors(atoms, atoms.positions[0])
    assert lengths[0] == pytest.approx(0.0)
    assert lengths[1] == pytest.approx(1.0)
    # Shortest image of +9.0 A in a 10 A cell is -1.0 A.
    assert vectors[1] == pytest.approx([-1.0, 0.0, 0.0])
