"""Physical invariants for the strain metrics.

The in-plane metric is what decides whether an interface candidate
survives, so these tests pin the textbook cases rather than any
particular implementation detail.
"""

from __future__ import annotations

import numpy as np
import pytest

from atominterface.builders.strain import (
    compute_mismatch,
    in_plane_mismatch,
    max_shear_strain,
    max_stretch_strain,
)

A = 4.0
SUB = np.diag([A, A, 1.0])


def test_identical_cells_have_zero_strain() -> None:
    result = in_plane_mismatch(SUB, SUB.copy())
    assert result["tensile"] == pytest.approx(0.0, abs=1e-12)
    assert result["shear"] == pytest.approx(0.0, abs=1e-12)


def test_isotropic_in_plane_stretch_is_pure_dilation() -> None:
    """5 % isotropic stretch must report 5 % tensile and **zero** shear.

    Under the old 3D decomposition this came out as 3.4 % tensile plus
    3.4 % of shear that does not exist in the crystal.
    """
    film = np.diag([A * 1.05, A * 1.05, 1.0])
    result = in_plane_mismatch(SUB, film)
    assert result["tensile"] == pytest.approx(0.05, rel=1e-12)
    assert result["shear"] == pytest.approx(0.0, abs=1e-12)


def test_uniaxial_in_plane_stretch() -> None:
    film = np.diag([A * 1.05, A, 1.0])
    result = in_plane_mismatch(SUB, film)
    assert result["tensile"] == pytest.approx(0.025, rel=1e-12)
    assert result["shear"] == pytest.approx(0.05, rel=1e-12)
    assert result["max_principal"] == pytest.approx(0.05, rel=1e-12)
    assert result["min_principal"] == pytest.approx(0.0, abs=1e-12)


def test_pure_in_plane_shear_reports_gamma() -> None:
    gamma = 0.05
    film = np.array([[A, A * gamma, 0.0], [0.0, A, 0.0], [0.0, 0.0, 1.0]])
    result = in_plane_mismatch(SUB, film)
    assert result["shear"] == pytest.approx(gamma, rel=1e-12)
    assert result["tensile"] == pytest.approx(0.0, abs=1e-12)


def test_vacuum_axis_does_not_leak_into_the_metric() -> None:
    """Stretching the vacuum axis must not register as strain."""
    stretched = np.diag([A, A, 3.7])
    assert in_plane_mismatch(SUB, stretched)["shear"] == pytest.approx(0.0, abs=1e-12)
    assert max_stretch_strain(SUB, stretched) == pytest.approx(0.0, abs=1e-12)


def test_accepts_two_by_two_and_two_by_three_inputs() -> None:
    film = np.diag([A * 1.05, A * 1.05, 1.0])
    from_2x3 = in_plane_mismatch(SUB[:2], film[:2])
    from_2x2 = in_plane_mismatch(SUB[:2, :2], film[:2, :2])
    assert from_2x3["tensile"] == pytest.approx(from_2x2["tensile"])
    assert from_2x3["shear"] == pytest.approx(from_2x2["shear"])


def test_degenerate_in_plane_cell_is_infinite_not_fatal() -> None:
    collapsed = np.array([[A, 0.0, 0.0], [A, 0.0, 0.0], [0.0, 0.0, 1.0]])
    result = in_plane_mismatch(SUB, collapsed)
    assert np.isinf(result["shear"])
    assert np.isinf(max_stretch_strain(SUB, collapsed))


def test_mixed_sign_mismatch_uses_absolute_stretch() -> None:
    """A 5 % compression must not slip past a stretch filter."""
    film = np.diag([A * 0.95, A * 0.95, 1.0])
    assert max_stretch_strain(SUB, film) == pytest.approx(0.05, rel=1e-12)


def test_full_3d_metric_is_kept_for_bulk_cells() -> None:
    """``compute_mismatch`` still decomposes in 3D -- documented, not used by builders."""
    bulk_sub = np.diag([A, A, A])
    bulk_film = np.diag([A * 1.05, A * 1.05, A * 1.05])
    assert compute_mismatch(bulk_sub, bulk_film)["tensile"] == pytest.approx(0.05125, rel=1e-9)


def test_shared_helpers_agree_with_the_dict() -> None:
    film = np.diag([A * 1.02, A * 1.01, 1.0])
    result = in_plane_mismatch(SUB, film)
    assert max_shear_strain(SUB, film) == pytest.approx(result["shear"])
    assert max_stretch_strain(SUB, film) == pytest.approx(
        max(abs(result["max_principal"]), abs(result["min_principal"]))
    )
