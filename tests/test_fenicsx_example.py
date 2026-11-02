"""Tests for the FEniCSx unit-cell example's NumPy core (no dolfinx needed).

The FEniCSx solve itself requires a dolfinx/ufl installation and is not
exercised here; the pure-NumPy pieces it depends on (loading a table, fitting
the Voigt stiffness, reassembling a 3x3 tensor) are unit-tested so the example
is verifiable on any environment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
_SPEC = importlib.util.spec_from_file_location(
    "_fenicsx_example", _EXAMPLES / "04_fenicsx_unit_cell.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)


def _isotropic_c(lam: float, mu: float) -> np.ndarray:
    """Isotropic Voigt stiffness with tensor shear components (GPa)."""
    c = np.zeros((6, 6))
    for i in range(3):
        c[i, i] = lam + 2.0 * mu
    for i in range(3):
        for j in range(3):
            if i != j:
                c[i, j] = lam
    for i in range(3, 6):
        c[i, i] = 2.0 * mu
    return c


def test_fit_stiffness_recovers_isotropic() -> None:
    lam, mu = 100.0, 50.0
    c0 = _isotropic_c(lam, mu)
    rng = np.random.RandomState(0)
    strain = rng.uniform(-0.005, 0.005, size=(24, 6))
    strain[:, 0] = np.linspace(-0.01, 0.02, 24)
    stress = strain @ c0.T
    c = _mod.fit_stiffness(strain, stress)
    np.testing.assert_allclose(c, c0, atol=1e-8)


def test_fit_stiffness_is_symmetric() -> None:
    rng = np.random.RandomState(1)
    strain = rng.uniform(-0.01, 0.01, size=(20, 6))
    stress = strain @ _isotropic_c(100.0, 50.0).T
    c = _mod.fit_stiffness(strain, stress)
    np.testing.assert_allclose(c, c.T, atol=1e-12)


def test_voigt_matrix_to_tensor_mapping() -> None:
    sv = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    s = _mod.voigt_matrix_to_tensor(sv)
    np.testing.assert_allclose(
        s,
        np.array(
            [
                [1.0, 6.0, 5.0],
                [6.0, 2.0, 4.0],
                [5.0, 4.0, 3.0],
            ]
        ),
    )


def test_load_table_roundtrip(tmp_path: Path) -> None:
    strain = np.linspace(0.0, 0.02, 5)[:, None] * np.ones((5, 6))
    stress = strain @ _isotropic_c(100.0, 50.0).T
    units = {"strain_voigt": "dimensionless", "cauchy_stress_voigt_gpa": "GPa"}
    path = tmp_path / "t.npz"
    np.savez(
        path,
        strain_voigt=strain,
        cauchy_stress_voigt_gpa=stress,
        units=np.asarray(__import__("json").dumps(units)),
        allow_pickle=False,
    )
    got_strain, got_stress, got_units = _mod.load_table(path)
    np.testing.assert_allclose(got_strain, strain)
    np.testing.assert_allclose(got_stress, stress)
    assert got_units == units
