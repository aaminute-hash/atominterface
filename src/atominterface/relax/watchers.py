"""OOD watchers: uncertainty signals evaluated during MD/relaxation.

A watcher is a stateless callable: given an Atoms snapshot and its
per-atom forces, return a scalar ``ood_score`` in [0, 1] and a short
reason string.  Scores across watchers are aggregated by the guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class WatchResult:
    score: float = 0.0
    reason: str = ""
    per_atom: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


class NeighborWatcher:
    """Flag overlapping atoms and abnormally short bonds."""

    def __init__(self, min_dist_frac: float = 0.6) -> None:
        self.min_dist_frac = min_dist_frac

    def __call__(self, atoms, forces: np.ndarray) -> WatchResult:  # noqa: D401
        pos = atoms.positions
        n = len(pos)
        if n < 2:
            return WatchResult()
        # crude pairwise check (demo; production should use neighbour list)
        min_d = 1e9
        for i in range(n - 1):
            d = np.linalg.norm(pos[i + 1 :] - pos[i], axis=1).min()
            min_d = min(min_d, d)
        # scale threshold by cell length as a stand-in for covalent radius
        scale = float(np.mean(np.linalg.norm(atoms.cell, axis=1))) / 10.0
        threshold = max(scale * self.min_dist_frac, 0.5)
        score = 0.0 if min_d > threshold else min(1.0, threshold / max(min_d, 1e-6))
        reason = "" if score == 0.0 else f"min_pairwise_dist={min_d:.3f}A < {threshold:.3f}A"
        return WatchResult(score=score, reason=reason, meta={"min_dist": float(min_d)})


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
    """

    @staticmethod
    def from_force_arrays(force_arrays: Sequence[np.ndarray]) -> WatchResult:
        if len(force_arrays) < 2:
            return WatchResult()
        arr = np.asarray(force_arrays)
        per_atom = float(np.linalg.norm(arr.std(axis=0), axis=1).mean())
        score = float(min(1.0, per_atom))
        return WatchResult(
            score=score, reason="committee_disagreement", meta={"disagreement": per_atom}
        )
