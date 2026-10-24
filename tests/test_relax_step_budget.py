"""Step-budget regression for ``GuardedRelaxer``.

``max_steps`` is documented as the relaxation step cap, but ``run()``
used to call ``opt.run(fmax=...)`` without ``steps`` -- ASE's default
budget is 100,000,000 steps, so a calculator whose residual force never
falls below ``fmax`` relaxed *forever* inside a single call, and the
guard loop never got to see a second iteration.

Two things made this reachable rather than theoretical:

* the dummy backend's force floor is ~0.018 eV/A, so the demo's 0.01
  eV/A tolerance is unreachable -- ``examples/03_interface_energy.py
  --backend dummy`` span past 265,000 FIRE steps before this fix;
* the Cu/Ni validation note measured a real MACE residual of 0.0099
  eV/A against that same 0.01 eV/A tolerance, i.e. a production run
  sits 1 % under the threshold.  Any backend that plateaus just above
  it hangs the same way.

The budget is now a *total* optimizer-step allowance shared by all
guard iterations, so a non-converging run returns with
``converged=False`` and an explicit reason instead of spinning.
"""

from __future__ import annotations

import time

import numpy as np
from ase.atoms import Atoms

from atominterface.relax.guarded_calc import GuardedRelaxer

MAX_STEPS = 20
WALL_CLOCK_LIMIT_S = 10.0


def _atoms(n: int = 3) -> Atoms:
    """A few atoms far apart in a box large enough that drift cannot
    reach them: these runs must stop on the budget, not on geometry."""
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])[:n]
    return Atoms(numbers=[6] * n, positions=positions, cell=np.eye(3) * 20.0, pbc=True)


class _PlateauCalc:
    """Constant non-zero force: the residual never reaches ``fmax``.

    Counts ``get_forces`` calls, the observable proxy for optimizer
    steps -- ASE evaluates forces once per step, plus once per guard
    iteration before the optimizer is handed the atoms.
    """

    def __init__(self, n_atoms: int, magnitude: float = 0.05) -> None:
        self._forces = np.zeros((n_atoms, 3))
        self._forces[:, 0] = magnitude
        self.calls = 0

    def get_potential_energy(self, atoms: Atoms, force_consistent: bool = False) -> float:
        # ASE 3.29's optimizer probes get_potential_energy(
        # force_consistent=True) once per relaxation; a stub that
        # rejects the kwarg dies as relax_exception before the budget
        # ever gets a chance to run out.
        return -0.001 * self.calls

    def get_forces(self, atoms: Atoms) -> np.ndarray:
        self.calls += 1
        return self._forces

    def get_stress(self, atoms: Atoms) -> np.ndarray:
        return np.zeros(6)


class _ZeroForceCalc:
    """Already at the minimum: the optimizer must return immediately."""

    def __init__(self, n_atoms: int) -> None:
        self._forces = np.zeros((n_atoms, 3))
        self.calls = 0

    def get_potential_energy(self, atoms: Atoms, force_consistent: bool = False) -> float:
        return 0.0

    def get_forces(self, atoms: Atoms) -> np.ndarray:
        self.calls += 1
        return self._forces

    def get_stress(self, atoms: Atoms) -> np.ndarray:
        return np.zeros(6)


def test_non_converging_calculator_terminates_within_budget():
    """``max_steps`` caps the total optimizer work, not the guard loop."""
    atoms = _atoms()
    calc = _PlateauCalc(len(atoms))
    relaxer = GuardedRelaxer(
        calculator=calc,
        max_steps=MAX_STEPS,
        fmax=1e-6,
        ood_threshold=0.99,
    )
    started = time.monotonic()
    result = relaxer.run(atoms)
    elapsed = time.monotonic() - started

    # A run that ignores the budget does not fail an assertion, it just
    # never gets here; the wall clock keeps this test honest even then.
    assert elapsed < WALL_CLOCK_LIMIT_S, f"relaxation did not terminate: {elapsed:.1f}s"
    assert result.converged is False
    assert result.ood_triggered is False  # stopped by the budget, not the guard
    # The invariant is O(max_steps) force evaluations, not an exact
    # count: ASE asks for forces a few times per optimizer step (step(),
    # converged(), loop entry), measured at ~3.2 calls/step on ASE
    # 3.29 -- 64 calls for these 20 steps.  The regression this pins is
    # unbounded: the same stub used to reach 265,000 steps in 90 s.
    assert calc.calls <= 5 * MAX_STEPS
    assert any(reason.startswith("step_budget_exhausted") for reason in result.reasons)


def test_converging_calculator_is_unaffected_by_the_budget():
    """The budget must not slow down or reject a converged relaxation."""
    atoms = _atoms()
    calc = _ZeroForceCalc(len(atoms))
    relaxer = GuardedRelaxer(
        calculator=calc,
        max_steps=MAX_STEPS,
        fmax=0.05,
        ood_threshold=0.99,
    )
    result = relaxer.run(atoms)
    assert result.converged is True
    assert result.n_steps == 1
    assert result.reasons == []
    assert calc.calls <= 2 * MAX_STEPS
