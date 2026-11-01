"""Physical tests for the trajectory stress/strain extraction pipeline.

These are hand-computed, pure numpy/ase checks -- no network, no MACE, no
pymatgen.  The Voigt order ``[xx, yy, zz, yz, xz, xy]`` with tensor (not
engineering, i.e. no factor of two) shear components is pinned here so a
future sign/order regression fails loudly.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write as ase_write

from atominterface.physics import (
    EV_A3_TO_GPA,
    cauchy_to_voigt,
    export_constitutive_table,
    extract,
    green_lagrange_strain,
    rotate_stress,
    rotation_matrix,
    strain_to_voigt,
    virial_to_cauchy,
)


def _matrix_from_voigt(v) -> np.ndarray:
    """Inverse of the package's Voigt convention (tensor shear, no 2x)."""
    v = np.asarray(v, dtype=float)
    return np.array(
        [
            [v[0], v[5], v[4]],
            [v[5], v[1], v[3]],
            [v[4], v[3], v[2]],
        ]
    )


# ── Green-Lagrange strain ─────────────────────────────────────


def test_uniaxial_stretch_green_lagrange() -> None:
    cell_ref = np.eye(3)
    cell = np.diag([1.1, 1.0, 1.0])
    E = green_lagrange_strain(cell, cell_ref)
    expected = np.zeros((3, 3))
    expected[0, 0] = 0.5 * (1.1**2 - 1.0)  # = 0.105
    np.testing.assert_allclose(E, expected, atol=1e-12)
    assert E[0, 0] == pytest.approx(0.105, abs=1e-12)


def test_pure_shear_green_lagrange() -> None:
    g = 0.05
    cell_ref = np.eye(3)
    cell = np.array([[1.0, g, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    E = green_lagrange_strain(cell, cell_ref)
    # F^T F = [[1, g, 0], [g, 1+g^2, 0], [0, 0, 1]], so
    # E_xx = 0, E_yy = g^2/2, E_xy = E_yx = g/2 (tensor component, no 2x).
    assert E[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert E[1, 1] == pytest.approx(g**2 / 2.0, abs=1e-12)
    assert E[0, 1] == pytest.approx(g / 2.0, abs=1e-12)
    assert E[1, 0] == pytest.approx(g / 2.0, abs=1e-12)
    assert E[2, 2] == pytest.approx(0.0, abs=1e-12)


def test_reference_equal_cell_is_zero_strain() -> None:
    cell = np.array([[4.0, 0.1, 0.2], [0.0, 3.9, 0.1], [0.0, 0.0, 4.1]])
    E = green_lagrange_strain(cell, cell.copy())
    np.testing.assert_allclose(E, np.zeros((3, 3)), atol=1e-12)


# ── Cauchy stress ─────────────────────────────────────────────


def test_virial_to_cauchy_sign_and_normalization() -> None:
    virial = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    volume = 2.0
    sigma = virial_to_cauchy(virial, volume)
    np.testing.assert_allclose(sigma, -virial / volume, atol=1e-15)


def test_rigid_rotation_preserves_principal_stresses() -> None:
    sigma = np.array([[3.0, 0.5, 0.2], [0.5, 2.0, 0.1], [0.2, 0.1, 1.0]])
    R = rotation_matrix([1.0, 2.0, 3.0], 37.5)
    sigma_rot = rotate_stress(sigma, R)
    # rotate_stress is exactly R @ s @ R.T by contract.
    np.testing.assert_allclose(sigma_rot, R @ sigma @ R.T, atol=1e-12)
    eig_orig = np.linalg.eigvalsh(sigma)
    eig_rot = np.linalg.eigvalsh(sigma_rot)
    np.testing.assert_allclose(np.sort(eig_orig), np.sort(eig_rot), atol=1e-12)


def test_voigt_order_and_roundtrip() -> None:
    E = np.array([[0.11, 0.03, 0.02], [0.03, 0.04, 0.01], [0.02, 0.01, 0.06]])
    v = strain_to_voigt(E)
    np.testing.assert_allclose(v, [0.11, 0.04, 0.06, 0.01, 0.02, 0.03], atol=1e-15)
    np.testing.assert_allclose(_matrix_from_voigt(v), E, atol=1e-15)

    sigma = np.array([[1.0, 0.5, 0.3], [0.5, 2.0, 0.4], [0.3, 0.4, 3.0]])
    sv = cauchy_to_voigt(sigma)
    np.testing.assert_allclose(sv, [1.0, 2.0, 3.0, 0.4, 0.3, 0.5], atol=1e-15)
    np.testing.assert_allclose(_matrix_from_voigt(sv), sigma, atol=1e-15)
    # voigt -> matrix -> voigt is the identity.
    np.testing.assert_allclose(cauchy_to_voigt(_matrix_from_voigt(sv)), sv, atol=1e-15)


# ── extract() on Atoms sequences ──────────────────────────────


class _StressCalc(Calculator):
    """Calculator that always reports the same (3, 3) stress tensor."""

    implemented_properties = ["stress"]

    def __init__(self, stress: np.ndarray) -> None:
        super().__init__()
        self._stress = np.asarray(stress, dtype=float)

    def get_stress(self, atoms: Atoms | None = None, voigt: bool = True) -> np.ndarray:
        return self._stress


def _make_frame(cell_diag, *, velocities=None, stress=None) -> Atoms:
    atoms = Atoms(
        "Cu4",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.0, 1.8, 0.0], [1.8, 1.8, 1.8]],
        cell=np.diag(cell_diag),
        pbc=True,
    )
    if velocities is not None:
        atoms.set_velocities(np.asarray(velocities, dtype=float))
    if stress is not None:
        atoms.calc = _StressCalc(stress)
    return atoms


SIGMA = np.array([[1.0, 0.5, 0.3], [0.5, 2.0, 0.4], [0.3, 0.4, 3.0]])
SIGMA_VOIGT = np.array([1.0, 2.0, 3.0, 0.4, 0.3, 0.5])
VELS = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6], [-0.1, 0.2, 0.7], [0.3, -0.3, 0.0]])


def test_extract_atoms_sequence_channels_and_shapes() -> None:
    frames = [
        _make_frame([3.6, 3.6, 3.6]),  # reference: no velocity, no stress
        _make_frame([3.96, 3.6, 3.6], velocities=VELS, stress=SIGMA),
        _make_frame([4.32, 3.6, 3.6], velocities=VELS, stress=SIGMA),
    ]
    table = extract(frames)

    assert table.n_frames == 3
    assert table.strain_voigt.shape == (3, 6)
    assert table.cauchy_stress_voigt.shape == (3, 6)
    assert table.cauchy_stress_voigt_gpa.shape == (3, 6)
    assert table.temperature.shape == (3,)
    assert table.cells.shape == (3, 3, 3)
    assert table.volumes.shape == (3,)
    assert table.has_stress.shape == (3,)

    # First frame is the reference configuration -> zero strain.
    np.testing.assert_allclose(table.strain_voigt[0], np.zeros(6), atol=1e-12)
    # Frame 1: 1.1x along x -> E_xx = 0.105; frame 2: 1.2x -> E_xx = 0.22.
    assert table.strain_voigt[1, 0] == pytest.approx(0.105, abs=1e-12)
    assert table.strain_voigt[2, 0] == pytest.approx(0.22, abs=1e-12)

    # Stress: frame 0 missing (NaN), frames 1/2 carry SIGMA in Voigt order.
    assert not table.has_stress[0]
    assert bool(table.has_stress[1]) and bool(table.has_stress[2])
    np.testing.assert_allclose(table.cauchy_stress_voigt[1], SIGMA_VOIGT, atol=1e-12)
    np.testing.assert_allclose(table.cauchy_stress_voigt[2], SIGMA_VOIGT, atol=1e-12)
    assert np.isnan(table.cauchy_stress_voigt[0]).all()

    # GPa channel is the eV/A^3 channel scaled by EV_A3_TO_GPA.
    np.testing.assert_allclose(
        table.cauchy_stress_voigt_gpa[1:],
        table.cauchy_stress_voigt[1:] * EV_A3_TO_GPA,
        atol=1e-12,
    )

    # Temperature: frame 0 missing (NaN); frames 1/2 match ASE's own value.
    assert np.isnan(table.temperature[0])
    assert table.temperature[1] == pytest.approx(frames[1].get_temperature(), rel=1e-12)
    assert table.temperature[2] == pytest.approx(frames[2].get_temperature(), rel=1e-12)

    # Units are explicit.
    assert table.units["cauchy_stress_voigt"] == "eV/A^3"
    assert table.units["cauchy_stress_voigt_gpa"] == "GPa"
    assert table.units["strain_voigt"].startswith("dimensionless")


def test_extract_reference_cell_override() -> None:
    ref = np.diag([3.6, 3.6, 3.6])
    frame = _make_frame([3.96, 3.6, 3.6])
    table = extract([frame], reference_cell=ref)
    assert table.strain_voigt[0, 0] == pytest.approx(0.105, abs=1e-12)
    np.testing.assert_allclose(table.reference_cell, ref, atol=1e-15)


def test_extract_no_velocities_temperature_none() -> None:
    frames = [_make_frame([3.6, 3.6, 3.6]), _make_frame([3.96, 3.6, 3.6])]
    table = extract(frames)
    assert table.temperature is None


def test_extract_no_calc_stress_none_and_has_stress_false() -> None:
    frames = [_make_frame([3.6, 3.6, 3.6]), _make_frame([3.96, 3.6, 3.6])]
    table = extract(frames)
    assert table.cauchy_stress_voigt is None
    assert table.cauchy_stress_voigt_gpa is None
    assert not table.has_stress.any()


def test_extract_empty_trajectory_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        extract([])


# ── extract() from a file path ────────────────────────────────


def test_extract_from_file_path(tmp_path) -> None:
    path = tmp_path / "frames.extxyz"
    frames = [
        _make_frame([3.6, 3.6, 3.6]),
        _make_frame([3.96, 3.6, 3.6]),
        _make_frame([4.32, 3.6, 3.6]),
    ]
    ase_write(path, frames, format="extxyz")
    table = extract(path)

    assert table.n_frames == 3
    assert table.strain_voigt.shape == (3, 6)
    # The cell round-trips through an extxyz file, so allow for float formatting
    # noise; the in-memory tests above pin the physics at 1e-12.
    np.testing.assert_allclose(table.strain_voigt[0], np.zeros(6), atol=1e-8)
    assert table.strain_voigt[1, 0] == pytest.approx(0.105, abs=1e-8)
    assert table.strain_voigt[2, 0] == pytest.approx(0.22, abs=1e-8)


# ── export_constitutive_table() round-trip ─────────────────────


def test_export_constitutive_table_roundtrip(tmp_path) -> None:
    frames = [
        _make_frame([3.6, 3.6, 3.6]),
        _make_frame([3.96, 3.6, 3.6], velocities=VELS, stress=SIGMA),
        _make_frame([4.32, 3.6, 3.6], velocities=VELS, stress=SIGMA),
    ]
    table = extract(frames)
    path = export_constitutive_table(table, tmp_path / "table")
    assert path == tmp_path / "table.npz"
    assert path.exists()

    data = np.load(path)
    assert set(data.files) == {
        "strain_voigt",
        "cauchy_stress_voigt",
        "cauchy_stress_voigt_gpa",
        "temperature",
        "cells",
        "volumes",
        "reference_cell",
        "has_stress",
        "units",
    }

    np.testing.assert_allclose(data["strain_voigt"], table.strain_voigt, atol=1e-12)
    np.testing.assert_allclose(data["cauchy_stress_voigt"], table.cauchy_stress_voigt, atol=1e-12)
    np.testing.assert_allclose(
        data["cauchy_stress_voigt_gpa"], table.cauchy_stress_voigt_gpa, atol=1e-12
    )
    np.testing.assert_allclose(data["temperature"], table.temperature, atol=1e-12)
    np.testing.assert_allclose(data["cells"], table.cells, atol=1e-12)
    np.testing.assert_allclose(data["volumes"], table.volumes, atol=1e-12)
    np.testing.assert_allclose(data["reference_cell"], table.reference_cell, atol=1e-12)
    np.testing.assert_array_equal(data["has_stress"], table.has_stress)
    assert table.n_frames == 3

    assert json.loads(str(data["units"])) == table.units
    assert json.loads(str(data["units"]))["cauchy_stress_voigt"] == "eV/A^3"
    assert json.loads(str(data["units"]))["cauchy_stress_voigt_gpa"] == "GPa"


def test_export_constitutive_table_omits_absent_channels(tmp_path) -> None:
    table = extract([_make_frame([3.6, 3.6, 3.6])])
    path = export_constitutive_table(table, tmp_path / "table.npz")
    data = np.load(path)
    assert "cauchy_stress_voigt" not in data.files
    assert "cauchy_stress_voigt_gpa" not in data.files
    assert "temperature" not in data.files
