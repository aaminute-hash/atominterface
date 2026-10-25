"""Integration of the OOD guard with per-atom signals, cluster localisation
and committee state.

These pin the wiring between the watchers (work package B) and
``GuardedRelaxer`` (work package A) that the integration branch adds:

* ``_aggregate()`` forwards ``per_atom`` / ``meta`` from the winning
  watcher, so a quarantine cluster anchors on the worst atom instead of
  always atom 0;
* ``cut_cluster`` receives the record and fills ``cluster_indices``;
* the watcher chain runs after **every optimizer step**, so a transient
  excursion inside a relaxation cannot hide behind benign final forces;
* a committee that was asked for but produced no usable member is
  reported as an outage, never as a silent score-0 consensus.
"""

from __future__ import annotations

import time

import numpy as np
from ase.atoms import Atoms
from ase.calculators.emt import EMT
from ase.io import read

from atominterface.relax.guarded_calc import GuardedRelaxer
from atominterface.relax.watchers import WatchResult


class _FlagAtom3:
    """Watcher that flags atom 3 with a per-atom signal."""

    def __call__(self, atoms, forces) -> WatchResult:
        per_atom = np.zeros(len(atoms))
        per_atom[3] = 1.0
        return WatchResult(score=1.0, reason="synthetic_flag", per_atom=per_atom)


class _TripOnEvaluation:
    """Flags only on the Nth evaluation, benign at every other one."""

    def __init__(self, evaluation: int) -> None:
        self._evaluation = evaluation
        self.calls = 0

    def __call__(self, atoms, forces) -> WatchResult:
        self.calls += 1
        tripped = self.calls == self._evaluation
        return WatchResult(
            score=1.0 if tripped else 0.0, reason="transient_excursion" if tripped else ""
        )


class _NoPerAtom:
    """A winning score without any per-atom localisation."""

    def __call__(self, atoms, forces) -> WatchResult:
        return WatchResult(score=1.0, reason="no_localisation")


class _BrokenMember:
    """A committee member whose every evaluation fails."""

    def get_forces(self, atoms):
        raise RuntimeError("member model unreadable")


def _six_carbons() -> Atoms:
    """Atom 3 isolated at the far corner; every other atom >= 10 A away."""
    positions = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (0.0, 10.0, 0.0),
        (30.0, 30.0, 30.0),
        (0.0, 0.0, 10.0),
        (10.0, 10.0, 0.0),
    ]
    return Atoms(numbers=[6] * 6, positions=positions, cell=np.eye(3) * 60.0, pbc=False)


def _h2(distance: float = 1.0) -> Atoms:
    return Atoms(
        "H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, distance]], cell=np.eye(3) * 20.0, pbc=True
    )


def test_aggregate_forwards_per_atom_and_meta():
    relaxer = GuardedRelaxer(calculator=EMT(), watchers=[_FlagAtom3()], ood_threshold=0.99)
    watch = relaxer._aggregate(_six_carbons(), np.zeros((6, 3)))
    assert watch.score == 1.0
    assert watch.reason == "synthetic_flag"
    assert watch.per_atom is not None
    assert int(np.argmax(watch.per_atom)) == 3


def test_quarantine_cluster_is_anchored_on_the_worst_atom(tmp_path):
    relaxer = GuardedRelaxer(
        calculator=EMT(),
        watchers=[_FlagAtom3()],
        ood_threshold=0.5,
        fmax=0.05,
        max_steps=50,
        quarantine_dir=tmp_path,
    )
    result = relaxer.run(_six_carbons())

    assert result.ood_triggered
    assert result.ood_step == 0
    assert not result.metadata.get("quarantine_center_degraded", False)
    # Every other atom is more than the default 6 A cut radius away from
    # atom 3, so the cluster must contain exactly atom 3, at atom 3's
    # position -- not atom 0's.
    cluster = read(result.quarantine_path, format="vasp")
    assert len(cluster) == 1
    assert np.allclose(cluster.positions[0], [30.0, 30.0, 30.0], atol=1e-6)


def test_missing_per_atom_signal_is_flagged_not_silent(tmp_path):
    relaxer = GuardedRelaxer(
        calculator=EMT(),
        watchers=[_NoPerAtom()],
        ood_threshold=0.5,
        fmax=0.05,
        max_steps=50,
        quarantine_dir=tmp_path,
    )
    result = relaxer.run(_six_carbons())

    assert result.ood_triggered
    assert result.metadata.get("quarantine_center_degraded") is True
    # Falls back to atom 0 -- flagged, not silent.
    cluster = read(result.quarantine_path, format="vasp")
    assert len(cluster) == 1
    assert np.allclose(cluster.positions[0], [0.0, 0.0, 0.0], atol=1e-6)


def test_transient_excursion_during_relaxation_is_caught(tmp_path):
    """The watcher trips only on the 3rd guard evaluation.

    Evaluations happen once before the optimizer starts and after every
    optimizer step; the 3rd is an intermediate geometry of the H2
    relaxation.  Sampling only at guard-iteration boundaries (the
    pre-integration behaviour) would check the start and the converged
    end state, never see the trip, and report a clean convergence.
    """
    trip = _TripOnEvaluation(3)
    relaxer = GuardedRelaxer(
        calculator=EMT(),
        watchers=[trip],
        ood_threshold=0.99,
        fmax=0.05,
        max_steps=50,
        quarantine_dir=tmp_path,
    )
    started = time.monotonic()
    result = relaxer.run(_h2(1.0))
    assert time.monotonic() - started < 30.0

    assert result.ood_triggered, "mid-relaxation excursion was missed"
    assert result.ood_step == 0
    assert trip.calls >= 3
    assert not result.converged
    assert result.valid is False


def test_committee_outage_is_reported_not_read_as_consensus(tmp_path):
    relaxer = GuardedRelaxer(
        calculator=EMT(),
        committee_calculators=[_BrokenMember()],
        ood_threshold=0.99,
        fmax=0.05,
        max_steps=100,
        quarantine_dir=tmp_path,
    )
    result = relaxer.run(_h2(0.9))

    # The primary model is unaffected: this is a certification gap, not
    # an energy failure.
    assert result.converged
    assert result.valid is True
    assert result.metadata["committee_state"] == "unavailable"
    assert result.metadata["ood_monitoring"] == "committee_unavailable"
    assert "committee_unavailable" in result.reasons


def test_default_chain_without_committee_reports_not_configured(tmp_path):
    relaxer = GuardedRelaxer(
        calculator=EMT(), ood_threshold=0.99, fmax=0.05, max_steps=100, quarantine_dir=tmp_path
    )
    result = relaxer.run(_h2(0.9))

    assert result.converged
    assert result.metadata["committee_state"] == "not_configured"
    assert result.metadata["committee_enabled"] is False
    assert result.metadata["ood_monitoring"] == "heuristics_only"
