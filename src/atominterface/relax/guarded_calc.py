"""Guarded relaxer: wrap an ASE calculator with OOD monitoring.

This is the heart of AtomInterface.  It does **not** replace the
underlying model; it observes every energy/force/stress evaluation and
decides whether the current atomic environment is outside the model's
training distribution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from .quarantine import QuarantineRecord, cut_cluster, write_dft_input
from .watchers import ForceNormWatcher, NeighborWatcher, WatchResult

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class RelaxResult:
    atoms: Atoms  # noqa: F821
    n_steps: int = 0
    converged: bool = False
    ood_triggered: bool = False
    ood_step: int = -1
    max_ood_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    quarantine_path: str = ""
    metadata: dict = field(default_factory=dict)


class GuardedRelaxer:
    """Run a relaxation/MD while monitoring OOD signals."""

    def __init__(
        self,
        calculator,
        *,
        optimizer: str = "FIRE",
        fmax: float = 0.05,
        max_steps: int = 500,
        ood_threshold: float = 0.8,
        watchers: Sequence[Callable] | None = None,
        quarantine_dir: str | Path = "./flagged",
        dft_fmt: str = "vasp",
        stop_on: str = "any",
    ) -> None:
        self._base = calculator
        self.optimizer = optimizer
        self.fmax = fmax
        self.max_steps = max_steps
        self.ood_threshold = ood_threshold
        self.watchers = (
            list(watchers)
            if watchers is not None
            else [
                NeighborWatcher(),
                ForceNormWatcher(),
            ]
        )
        self.quarantine_dir = Path(quarantine_dir)
        self.dft_fmt = dft_fmt
        self.stop_on = stop_on

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, atoms) -> RelaxResult:  # noqa: D401, F821
        atoms = atoms.copy()
        atoms.calc = self._wrap(atoms)
        result = RelaxResult(atoms=atoms)

        try:
            opt = self._make_optimizer(atoms)
            for step in range(self.max_steps):
                result.n_steps = step + 1
                forces = atoms.get_forces()
                watch = self._aggregate(atoms, forces)
                if watch.score >= self.ood_threshold:
                    result.ood_triggered = True
                    result.ood_step = step
                    result.max_ood_score = float(watch.score)
                    result.reasons.append(watch.reason)
                    rec = QuarantineRecord(
                        run_id=os.environ.get("ATOMINTERFACE_RUN_ID", "local"),
                        step=step,
                        center_index=self._worst_atom(watch, atoms),
                        ood_score=float(watch.score),
                        reasons=[watch.reason] if watch.reason else [],
                    )
                    cluster = cut_cluster(atoms, rec.center_index)
                    rec.dft_input_path = str(
                        write_dft_input(cluster, rec, self.quarantine_dir, self.dft_fmt)
                    )
                    result.quarantine_path = rec.dft_input_path
                    break
                opt.run(fmax=self.fmax)
                if opt.converged():
                    result.converged = True
                    break
        except Exception as exc:  # noqa: BLE001
            result.reasons.append(f"relax_exception={exc!r}")
        result.atoms = atoms
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _wrap(self, atoms):  # noqa: D401
        base = self._base

        class _W(base):  # type: ignore[valid-type]
            def get_forces(self, a):  # noqa: A002
                return np.asarray(base.get_forces(a), dtype=float)

            def get_potential_energy(self, a):  # noqa: A003
                return float(base.get_potential_energy(a))

        return _W(atoms)

    def _make_optimizer(self, atoms):  # noqa: D401
        if self.optimizer.upper() == "FIRE":
            from ase.optimize import FIRE

            return FIRE(atoms)
        if self.optimizer.upper() == "LBFGS":
            from ase.optimize import LBFGS

            return LBFGS(atoms)
        raise ValueError(f"unknown optimizer {self.optimizer!r}")

    def _aggregate(self, atoms, forces: np.ndarray) -> WatchResult:  # noqa: D401
        scores, reasons = [], []
        for w in self.watchers:
            r = w(atoms, forces)
            scores.append(r.score)
            if r.reason:
                reasons.append(r.reason)
        if self.stop_on == "any":
            idx = int(np.argmax(scores))
            return WatchResult(score=float(scores[idx]), reason=reasons[idx] if reasons else "")
        score = float(np.mean(scores))
        return WatchResult(score=score, reason="; ".join(reasons))

    @staticmethod
    def _worst_atom(watch: WatchResult, atoms) -> int:  # noqa: D401
        if watch.per_atom is not None and len(watch.per_atom) == len(atoms):
            return int(np.argmax(watch.per_atom))
        return 0
