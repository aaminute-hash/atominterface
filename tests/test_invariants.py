"""Tests for the physics invariants module (check_force_consistency)."""

import numpy as np
import pytest

from atominterface.physics import check_force_consistency


def _harmonic_energy(k: float = 1.0):
    """Toy potential E = (k/2) sum(r^2) whose gradient is exactly -k*r."""

    def energy(r: np.ndarray) -> float:
        return 0.5 * k * float(np.sum(np.asarray(r, dtype=float) ** 2))

    return energy


def test_check_force_consistency_harmonic_oscillator_exact():
    """Analytic forces of a harmonic well pass with near-zero residual."""
    k = 1.7
    rng = np.random.default_rng(5)
    positions = rng.normal(0.0, 0.5, (3, 3))
    exact_forces = -k * positions
    residual = check_force_consistency(_harmonic_energy(k), exact_forces, positions, delta=1e-4)
    assert 0.0 <= residual < 1e-6


def test_check_force_consistency_reports_wrong_force_nonzero():
    """Forces that are not the gradient of the energy are reported."""
    k = 1.0
    positions = np.array([[0.1, -0.2, 0.3]])
    wrong_forces = -1.5 * k * positions  # 1.5x too steep
    residual = check_force_consistency(_harmonic_energy(k), wrong_forces, positions, delta=1e-4)
    assert residual > 0.0
    # The planted force error is 0.5*k*|r| per component; the finite
    # difference itself contributes only ~delta^2 of noise.
    assert residual == pytest.approx(0.5 * k * np.max(np.abs(positions)), rel=1e-4)


def test_check_force_consistency_zero_atoms_returns_zero():
    """Empty input is a defined zero, not an error."""
    energy_fn = _harmonic_energy()
    assert check_force_consistency(energy_fn, np.zeros((0, 3)), np.zeros((0, 3))) == 0.0


def test_check_force_consistency_does_not_mutate_positions():
    """The probe shifts must never leak into the caller's array."""
    positions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    snapshot = positions.copy()
    check_force_consistency(_harmonic_energy(), -positions, positions, delta=1e-4)
    assert np.array_equal(positions, snapshot)
