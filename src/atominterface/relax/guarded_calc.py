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
from .watchers import CommitteeWatcher, ForceNormWatcher, NeighborWatcher, WatchResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase.atoms import Atoms


@dataclass
class RelaxResult:
    atoms: Atoms
    n_steps: int = 0
    converged: bool = False
    valid: bool = False
    ood_triggered: bool = False
    ood_step: int = -1
    max_ood_score: float = 0.0
    final_energy: float | None = None
    energies_finite: bool = False
    forces_finite: bool = False
    exception: str | None = None
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
        committee_calculators: Sequence | None = None,
        quarantine_dir: str | Path = "./flagged",
        dft_fmt: str = "vasp",
        stop_on: str = "any",
    ) -> None:
        self._base = calculator
        self.optimizer = optimizer
        self.fmax = fmax
        self.max_steps = max_steps
        self.ood_threshold = ood_threshold
        # Extra calculators whose forces form the disagreement
        # committee.  Only the default chain gains a CommitteeWatcher;
        # an explicitly passed ``watchers`` chain is respected verbatim
        # (the caller owns the chain, we never append to it).
        self.committee_calculators = list(committee_calculators) if committee_calculators else []
        if watchers is not None:
            self.watchers: list[Callable[..., WatchResult]] = list(watchers)
        else:
            chain: list[Callable[..., WatchResult]] = [
                NeighborWatcher(),
                ForceNormWatcher(),
            ]
            if self.committee_calculators:
                # Multi-model force disagreement: the only watcher here
                # that is a distributional signal rather than a
                # geometry/magnitude heuristic.
                chain.append(CommitteeWatcher(self.committee_calculators))
            self.watchers = chain
        self.quarantine_dir = Path(quarantine_dir)
        self.dft_fmt = dft_fmt
        self.stop_on = stop_on

    @classmethod
    def from_calculators(cls, calculators: Sequence, *, primary: int = 0, **kw):
        """Build a relaxer whose committee is every calculator but one.

        ``calculators[primary]`` drives the relaxation; the remaining
        calculators are wired as ``committee_calculators`` so the
        default chain gains a :class:`CommitteeWatcher`.  Passing
        ``watchers`` explicitly still overrides the whole chain, in
        which case the committee is not mounted.
        """
        calcs = list(calculators)
        if not calcs:
            raise ValueError("from_calculators needs at least one calculator")
        if not 0 <= primary < len(calcs):
            raise ValueError(f"primary index {primary} out of range for {len(calcs)} calculators")
        main = calcs[primary]
        committee = [calc for idx, calc in enumerate(calcs) if idx != primary]
        kw.setdefault("committee_calculators", committee)
        return cls(main, **kw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, atoms) -> RelaxResult:  # noqa: D401, F821, PLR0915
        atoms = atoms.copy()
        committee_enabled = any(
            isinstance(watcher, CommitteeWatcher) and bool(watcher.calculators)
            for watcher in self.watchers
        )
        result = RelaxResult(
            atoms=atoms,
            metadata={
                "committee_enabled": committee_enabled,
                "ood_monitoring": "committee+heuristics"
                if committee_enabled
                else "heuristics_only",
            },
        )

        # Every committee state observed across all guard evaluations,
        # resolved worst-first at the end of the run.  A state that never
        # appears (no CommitteeWatcher in the chain) is a legitimate
        # "not configured", distinct from "unavailable" (asked for and
        # every member failed).
        committee_states: list[str] = []

        def evaluate_state() -> np.ndarray | None:
            forces = np.asarray(atoms.get_forces(), dtype=float)
            energy = float(atoms.get_potential_energy())
            result.forces_finite = bool(np.all(np.isfinite(forces)))
            result.energies_finite = bool(np.isfinite(energy))
            result.final_energy = energy if result.energies_finite else None
            if not result.energies_finite:
                result.reasons.append("non_finite_energy")
            if not result.forces_finite:
                result.reasons.append("non_finite_forces")
            if not result.energies_finite or not result.forces_finite:
                return None
            return forces

        def check_watch(forces: np.ndarray, step: int) -> bool:
            watch = self._aggregate(atoms, forces)
            state = watch.meta.get("committee_state")
            if state is not None:
                committee_states.append(str(state))
            if not np.isfinite(watch.score):
                result.reasons.append("non_finite_ood_score")
                return False
            result.max_ood_score = max(result.max_ood_score, float(watch.score))
            if watch.score < self.ood_threshold:
                return True
            result.ood_triggered = True
            result.ood_step = step
            result.reasons.append(watch.reason)
            if watch.per_atom is None or len(watch.per_atom) != len(atoms):
                # The winning watcher carries no per-atom signal: the
                # cluster falls back to atom 0 and the run says so,
                # instead of silently quarantining the wrong site.
                result.metadata["quarantine_center_degraded"] = True
            rec = QuarantineRecord(
                run_id=os.environ.get("ATOMINTERFACE_RUN_ID", "local"),
                step=step,
                center_index=self._worst_atom(watch, atoms),
                ood_score=float(watch.score),
                reasons=[watch.reason] if watch.reason else [],
            )
            cluster = cut_cluster(atoms, rec.center_index, record=rec)
            rec.dft_input_path = str(
                write_dft_input(cluster, rec, self.quarantine_dir, self.dft_fmt)
            )
            result.quarantine_path = rec.dft_input_path
            return False

        try:
            atoms.calc = self._wrap(atoms)
            opt = self._make_optimizer(atoms)
            # ``max_steps`` is a *total* optimizer-step budget shared by
            # every guard iteration, not a per-iteration cap.  ASE's
            # ``opt.run()`` stops only on convergence and defaults to a
            # 1e8-step budget, so a calculator whose residual force
            # plateaus above ``fmax`` used to spin inside one call
            # forever -- which is what the dummy backend did once the
            # demo tolerance dropped to 0.01 eV/A below its force floor.
            budget = self.max_steps
            for step in range(self.max_steps):
                result.n_steps = step + 1
                forces = evaluate_state()
                if forces is None or not check_watch(forces, step):
                    break
                done_before = opt.get_number_of_steps()
                # Walk the optimizer one step at a time: ``irun`` yields
                # once before the first step and after every optimizer
                # step, so the guard sees every intermediate geometry.
                # Sampling only at guard-iteration boundaries let a
                # transient excursion inside a single ``opt.run`` call
                # pass unnoticed whenever the final forces were benign
                # again.
                stopped = False
                for converged in opt.irun(fmax=self.fmax, steps=budget):
                    forces = evaluate_state()
                    if forces is None or not check_watch(forces, step):
                        stopped = True
                        break
                    if converged:
                        break
                # ASE accumulates its own budget across irun() calls
                # (self.max_steps = self.nsteps + steps), so the delta is
                # exactly what this call spent.
                budget -= opt.get_number_of_steps() - done_before
                if stopped:
                    break
                if opt.converged():
                    result.converged = True
                    break
                if budget <= 0:
                    # A silent converged=False is a black box for
                    # provenance, and the Cu/Ni note measured a residual
                    # force of 0.0099 eV/A against a shipped 0.01
                    # tolerance -- close enough to exhaust the budget on
                    # a real backend, which must not pass unnoticed.
                    result.reasons.append(
                        f"step_budget_exhausted after {self.max_steps} optimizer steps"
                    )
                    break
        except Exception as exc:  # noqa: BLE001
            result.exception = f"{type(exc).__name__}: {exc}"
            result.reasons.append(f"relax_exception={exc!r}")
        self._record_committee_state(result, committee_states)
        result.valid = (
            result.converged
            and not result.ood_triggered
            and result.exception is None
            and result.energies_finite
            and result.forces_finite
            and result.final_energy is not None
        )
        result.atoms = atoms
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _record_committee_state(result: RelaxResult, states: list[str]) -> None:  # noqa: D401
        """Resolve the worst committee state observed, worst-first.

        A committee that was asked for but produced no usable member is
        a monitoring outage, not a consensus: the score of an
        ``unavailable`` committee is 0.0 and must never be readable as
        "the models agree".  The outage invalidates the certification,
        not the energy, so the run stays valid and the state is carried
        in the metadata and the reasons.  A state that never appeared
        (no CommitteeWatcher in the chain) is a legitimate "not
        configured", distinct from "unavailable".
        """
        if "unavailable" in states:
            result.metadata["committee_state"] = "unavailable"
            result.metadata["ood_monitoring"] = "committee_unavailable"
            result.reasons.append("committee_unavailable")
        elif "degraded" in states:
            result.metadata["committee_state"] = "degraded"
            result.metadata["ood_monitoring"] = "committee_degraded"
        elif "ok" in states:
            result.metadata["committee_state"] = "ok"
        elif "disabled" in states:
            result.metadata["committee_state"] = "disabled"
        else:
            result.metadata["committee_state"] = "not_configured"

    def _wrap(self, atoms):  # noqa: D401
        """Resolve a calculator instance without fabricating results."""
        base = self._base

        if not isinstance(base, type) and hasattr(base, "get_forces") and callable(base.get_forces):
            calc = base
        elif callable(base):
            try:
                calc = base(atoms)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"calculator factory failed for the supplied atoms: {type(exc).__name__}: {exc}"
                ) from exc
        else:
            raise TypeError(
                "calculator must be an ASE-compatible instance or a callable factory; "
                f"got {type(base).__name__}"
            )

        if not callable(getattr(calc, "get_forces", None)) or not callable(
            getattr(calc, "get_potential_energy", None)
        ):
            raise TypeError(
                "calculator factory returned an object without get_forces() and "
                "get_potential_energy()"
            )
        atoms.calc = calc
        return calc

    def _make_optimizer(self, atoms):  # noqa: D401
        if self.optimizer.upper() == "FIRE":
            from ase.optimize import FIRE

            return FIRE(atoms)
        if self.optimizer.upper() == "LBFGS":
            from ase.optimize import LBFGS

            return LBFGS(atoms)
        raise ValueError(f"unknown optimizer {self.optimizer!r}")

    def _aggregate(self, atoms, forces: np.ndarray) -> WatchResult:  # noqa: D401
        # Keep every watcher's full result: the reason reported for the
        # winning watcher must come from that same watcher.  The old
        # code indexed a reasons list compacted to non-empty entries,
        # so any empty-reason watcher ahead of the argmax shifted the
        # index (and an all-empty prefix raised IndexError, aborting
        # the relaxation as relax_exception).
        #
        # ``per_atom`` and ``meta`` are forwarded from the winning
        # watcher.  The previous rebuild dropped both, which anchored
        # every quarantine cluster on atom 0 (no localisation signal)
        # and made a committee outage indistinguishable from agreement
        # (no state in the metadata).
        results = [w(atoms, forces) for w in self.watchers]
        scores = [r.score for r in results]
        if self.stop_on == "any":
            idx = int(np.argmax(scores))
            best = results[idx]
            # The winner's keys take precedence, but every watcher's meta
            # is merged in: committee state is a property of the committee
            # watcher, not of whoever wins the argmax, and an outage
            # scores 0.0 -- it would never win and never surface if only
            # the winner were forwarded.
            meta = dict(best.meta)
            for res in results:
                for key, value in res.meta.items():
                    meta.setdefault(key, value)
            return WatchResult(
                score=float(scores[idx]),
                reason=best.reason,
                per_atom=best.per_atom,
                meta=meta,
            )
        score = float(np.mean(scores))
        # No single winner in mean mode: merge the metas and keep the
        # strongest per-atom signal on offer.
        merged: dict = {}
        for res in results:
            merged.update(res.meta)
        best_signal = max(
            (r for r in results if r.per_atom is not None),
            key=lambda r: r.score,
            default=None,
        )
        return WatchResult(
            score=score,
            reason="; ".join(r.reason for r in results if r.reason),
            per_atom=best_signal.per_atom if best_signal is not None else None,
            meta=merged,
        )

    @staticmethod
    def _worst_atom(watch: WatchResult, atoms) -> int:  # noqa: D401
        if watch.per_atom is not None and len(watch.per_atom) == len(atoms):
            return int(np.argmax(watch.per_atom))
        return 0
