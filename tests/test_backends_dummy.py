"""Regression tests for the dummy backend's ASE-facing surface.

The dummy backend is the zero-dependency stand-in CI runs against, so
if it stops answering ASE's calculator protocol every ``--relax``
demo silently degrades into a no-op: ``GuardedRelaxer.run`` swallows
the optimizer constructor failure as ``relax_exception``.  These tests
pin the protocol explicitly instead of trusting a green exit code.
"""

from __future__ import annotations

from ase.atoms import Atoms

from atominterface.backends import get_calculator
from atominterface.relax import GuardedRelaxer


def _dummy(atoms: Atoms):
    """Materialise the calculator instance behind the registry entry."""
    return get_calculator("dummy")(atoms)


def _cu2() -> Atoms:
    """Small periodic cell; content is irrelevant to the protocol."""
    return Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [1.8, 1.8, 1.8]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )


def test_dummy_accepts_force_consistent_kwarg():
    """ASE forwards force_consistent= into get_potential_energy.

    ``Optimizer._use_force_consistent_energy`` (ase/optimize/optimize.py)
    probes ``atoms.get_potential_energy(force_consistent=True)`` before
    the first step; a calculator that rejects the kwarg aborts every
    relaxation.  The dummy has one energy surface, so both calls must
    return the same value.
    """
    atoms = _cu2()
    calc = _dummy(atoms)
    plain = calc.get_potential_energy(atoms)
    consistent = calc.get_potential_energy(atoms, force_consistent=True)
    assert plain == consistent


def test_guarded_relax_dummy_runs_without_relax_exception():
    """The guard must run the optimizer, not swallow its constructor."""
    atoms = _cu2()
    relaxer = GuardedRelaxer(
        calculator=get_calculator("dummy"),
        fmax=0.05,
        max_steps=5,
    )
    result = relaxer.run(atoms)
    assert not any("relax_exception" in reason for reason in result.reasons)
    assert result.n_steps >= 1
