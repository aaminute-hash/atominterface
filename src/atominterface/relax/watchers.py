"""OOD watchers: uncertainty signals evaluated during MD/relaxation.

A watcher is a stateless callable: given an Atoms snapshot and its
per-atom forces, return a scalar ``ood_score`` in [0, 1] and a short
reason string.  Scores across watchers are aggregated by the guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from ase.data import covalent_radii

from .proximity import pairwise_distance_matrix

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class WatchResult:
    score: float = 0.0
    reason: str = ""
    per_atom: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


# ``WatchResult.meta["committee_state"]`` values.  A committee that was
# never asked for, one that produced no usable member and one that
# produced a perfect consensus all score 0.0; the state is what tells
# them apart, so a monitoring failure can never be read as agreement.
COMMITTEE_DISABLED = "disabled"
COMMITTEE_UNAVAILABLE = "unavailable"
COMMITTEE_DEGRADED = "degraded"
COMMITTEE_OK = "ok"


class NeighborWatcher:
    """Flag overlapping atoms and abnormally short bonds.

    The "too short" threshold is ``min_dist_frac * (r_i + r_j)`` over the
    covalent radii of the bonded pair, so it depends only on the local
    chemistry.  It used to be derived from the mean cell-vector length,
    which made the very same local geometry score differently depending
    on how much vacuum a slab happened to carry: thickening the vacuum
    raised the threshold and could turn a clean structure into an OOD
    one.

    Distances are minimum-image distances, so an overlap straddling a
    periodic boundary is seen.  ``per_atom`` carries each atom's
    relative bond shortfall, which is what lets the guard anchor a
    quarantine cluster on the offending atom instead of index 0.
    """

    # Covalent radii are tabulated per element; a species outside the
    # table (or the ASE dummy Z=0) falls back to this mid-table value so
    # the threshold stays a local length scale rather than becoming 0.
    fallback_radius: float = 0.7

    def __init__(self, min_dist_frac: float = 0.6) -> None:
        self.min_dist_frac = float(min_dist_frac)

    @classmethod
    def covalent_radii_for(cls, numbers: Sequence[int]) -> np.ndarray:
        """Covalent radius per atom, with a fallback for unknown species."""
        table = np.asarray(covalent_radii, dtype=float)
        z = np.asarray(numbers, dtype=int)
        known = (z > 0) & (z < len(table))
        radii = np.full(len(z), cls.fallback_radius, dtype=float)
        radii[known] = table[z[known]]
        return radii

    def __call__(self, atoms, forces: np.ndarray) -> WatchResult:  # noqa: D401
        n = len(atoms)
        if n < 2:
            # A single atom has no pair to be too close to.
            return WatchResult()

        dm = pairwise_distance_matrix(atoms)
        nearest = dm.min(axis=1)
        partner = dm.argmin(axis=1)

        radii = self.covalent_radii_for(atoms.numbers)
        # Per-pair threshold: the tightest bond is judged against the two
        # elements that actually form it.
        pair_threshold = self.min_dist_frac * (radii[:, None] + radii[None, :])
        threshold = pair_threshold[np.arange(n), partner]
        # Guard the division below; radii are > 0, so this only bites on a
        # degenerate ``min_dist_frac`` of zero or less.
        threshold = np.maximum(threshold, 1e-12)

        # Relative shortfall in [0, 1]: 0 at or beyond the threshold, -> 1
        # as the bond collapses.  Monotone in the bond length, so argmax
        # is the worst-bound atom even when several violate at once.
        per_atom = np.clip(1.0 - nearest / threshold, 0.0, 1.0)

        worst = int(np.argmax(per_atom))
        min_d = float(nearest[worst])
        worst_threshold = float(threshold[worst])
        # Saturating, as before: once a bond is too short the signal is
        # binary for the guard's threshold comparison.
        score = 0.0 if per_atom[worst] <= 0.0 else min(1.0, worst_threshold / max(min_d, 1e-6))
        reason = "" if score == 0.0 else f"min_pairwise_dist={min_d:.3f}A < {worst_threshold:.3f}A"
        return WatchResult(
            score=float(score),
            reason=reason,
            per_atom=per_atom,
            meta={
                "min_dist": min_d,
                "threshold": worst_threshold,
                "min_pair": [worst, int(partner[worst])],
                "global_min_dist": float(nearest.min()),
            },
        )


class ForceNormWatcher:
    """High per-atom force norms often indicate OOD or a bad geometry."""

    def __init__(self, max_norm: float = 50.0) -> None:
        self.max_norm = max_norm

    def __call__(self, atoms, forces: np.ndarray) -> WatchResult:  # noqa: D401
        norms = np.linalg.norm(forces, axis=1)
        m = float(norms.max())
        score = 0.0 if m <= self.max_norm else min(1.0, m / (2 * self.max_norm))
        reason = "" if score == 0.0 else f"max_force_norm={m:.2f} eV/A"
        return WatchResult(
            score=score, reason=reason, per_atom=norms / self.max_norm, meta={"max_norm": m}
        )


class CommitteeWatcher:
    """Committee disagreement (requires multiple force arrays).

    In production this is populated by the guard with per-model forces;
    here we expose the aggregation math so it can be tested without
    loading real models.

    Instantiated with a list of *additional* ASE calculators, the
    watcher evaluates every member alongside the primary forces the
    guard already computed and scores the spread between them.  A
    member that fails (non-standard interface, broken model) is
    skipped: the guard must never crash inside its own monitoring.

    Skipping must not be silent.  A committee that was never configured,
    one whose every member failed, and one that unanimously agrees all
    produce ``score == 0.0``; only ``meta["committee_state"]`` separates
    "we checked and the models concur" from "we could not check at all".
    The states are :data:`COMMITTEE_DISABLED` (no member configured),
    :data:`COMMITTEE_UNAVAILABLE` (configured, none usable),
    :data:`COMMITTEE_DEGRADED` (some members dropped) and
    :data:`COMMITTEE_OK`.
    """

    def __init__(self, calculators: Sequence | None = None, *, scale: float = 1.0) -> None:
        self.calculators = list(calculators) if calculators else []
        # Sensitivity knob kept for callers/tuning; deliberately NOT
        # applied to the aggregation below so ``from_force_arrays``
        # keeps its documented ``min(1, mean disagreement)`` semantics.
        self.scale = float(scale)

    @staticmethod
    def from_force_arrays(force_arrays: Sequence[np.ndarray]) -> WatchResult:
        """Score the spread of a set of force arrays.

        Fewer than two arrays cannot disagree, which is a *missing
        committee*, not an agreeing one -- the result carries the
        ``unavailable`` state instead of an indistinguishable zero.
        """
        if len(force_arrays) < 2:
            return WatchResult(
                meta={
                    "committee_state": COMMITTEE_UNAVAILABLE,
                    "committee_members": 0,
                    "committee_requested": len(force_arrays),
                    "committee_failed": len(force_arrays),
                }
            )
        arr = np.asarray(force_arrays)
        per_atom = float(np.linalg.norm(arr.std(axis=0), axis=1).mean())
        score = float(min(1.0, per_atom))
        return WatchResult(
            score=score,
            reason="committee_disagreement" if score > 0.0 else "",
            meta={
                "disagreement": per_atom,
                "committee_state": COMMITTEE_OK,
                "committee_members": len(force_arrays) - 1,
                "committee_requested": len(force_arrays) - 1,
                "committee_failed": 0,
            },
        )

    @staticmethod
    def _unavailable(requested: int, failed: int) -> WatchResult:
        """No usable committee: an explicit state, never a silent zero."""
        return WatchResult(
            score=0.0,
            reason="committee_unavailable",
            meta={
                "committee_state": COMMITTEE_UNAVAILABLE,
                "committee_members": 0,
                "committee_requested": int(requested),
                "committee_failed": int(failed),
            },
        )

    @staticmethod
    def _committee_forces(calc, atoms):
        """Forces of one committee member, or ``None`` if it fails.

        Standard ASE calculators answer ``calc.get_forces(atoms)``.
        Objects with a non-standard signature (our dummy factories,
        mocks) raise ``TypeError``/``AttributeError`` there; those get
        the classic fallback of attaching the calculator to a copy of
        the atoms and letting ASE drive it.  Any failure -- in either
        path -- returns ``None`` so the member is skipped instead of
        propagating into the guard loop.
        """
        try:
            return calc.get_forces(atoms)
        except (TypeError, AttributeError):
            pass
        except Exception:  # noqa: BLE001 - a broken member must not kill the guard
            return None
        try:
            probe = atoms.copy()
            probe.calc = calc
            return probe.get_forces()
        except Exception:  # noqa: BLE001 - same: skip, never crash
            return None

    def __call__(self, atoms, forces: np.ndarray) -> WatchResult:  # noqa: D401
        if not self.calculators:
            # Not enabled.  "We did not ask" is a different statement from
            # "we asked and the models agree"; the guard must be able to
            # see which one it got.
            return WatchResult(
                meta={
                    "committee_state": COMMITTEE_DISABLED,
                    "committee_members": 0,
                    "committee_requested": 0,
                    "committee_failed": 0,
                }
            )
        primary = np.asarray(forces, dtype=np.float64)
        arrays = [primary]
        for calc in self.calculators:
            member = self._committee_forces(calc, atoms)
            if member is None:
                continue
            arr = np.asarray(member, dtype=np.float64)
            if arr.shape != primary.shape:
                continue  # unusable member (wrong atom count/layout): skip
            arrays.append(arr)
        requested = len(self.calculators)
        failed = requested - (len(arrays) - 1)
        if len(arrays) < 2:
            # Every member failed or returned an unusable array.  The
            # score is 0 because there is nothing to compare, and the
            # state says so -- a monitoring outage is not consensus.
            return self._unavailable(requested, failed)
        result = self.from_force_arrays(arrays)
        result.meta["committee_state"] = COMMITTEE_OK if failed == 0 else COMMITTEE_DEGRADED
        result.meta["committee_requested"] = requested
        result.meta["committee_members"] = len(arrays) - 1
        result.meta["committee_failed"] = failed
        return result
