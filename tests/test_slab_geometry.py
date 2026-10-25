"""Slab-geometry gates: real ``SlabGenerator`` output, no mocks.

These tests pin the contract the builders rely on: a generated slab is a
finite board whose atomic planes are spaced by the theoretical
interplanar distance ``d_hkl`` of its Miller indices, with real vacuum
above the top plane.  pymatgen re-orients the cell so the surface normal
is ``cross(a1, a2)`` -- for anything but (001) that is *not* the
cartesian z axis, so every measurement here projects onto the true
normal.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from atominterface.builders.charge import finite_slab_geometry
from atominterface.builders.coherent import _flatten_terminations, _get_terminations


def _fcc(species: str, a: float):
    """Conventional fcc cell, as a bulk input for slab generation."""
    try:
        from pymatgen.core import Lattice, Structure  # noqa: WPS433
    except ImportError:
        pytest.skip("pymatgen not installed")

    coords: list[list[float]] = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    return Structure(Lattice.cubic(a), [species] * 4, coords)


def _plane_levels(structure) -> np.ndarray:
    """Sorted atomic-plane levels along the slab's own surface normal."""
    matrix = np.asarray(structure.lattice.matrix, dtype=float)
    normal = np.cross(matrix[0], matrix[1])
    normal = normal / np.linalg.norm(normal)
    projected = np.asarray(structure.cart_coords, dtype=float) @ normal
    return np.unique(np.round(projected, 4))


@pytest.mark.parametrize(
    ("species", "a"),
    [("Cu", 3.615), ("Ni", 3.524)],
)
@pytest.mark.parametrize("miller", [(0, 0, 1), (1, 1, 1)])
def test_generated_slab_planes_match_the_interplanar_spacing(species, a, miller):
    """Plane spacing must equal d_hkl; no gap may exceed 1.5 x d_hkl.

    ``min_slab_size`` is spent in whole oriented-unit-cell repeats
    (``in_unit_planes=False``), so the guaranteed quantity is the plane
    count: every repeat contributes ``height / d_hkl`` planes, hence the
    slab carries at least ``ceil(min_slab_size / d_hkl)`` of them.
    """
    min_slab_size = 8.0
    d_hkl = a / 2.0 if miller == (0, 0, 1) else a / np.sqrt(3.0)

    terms = _flatten_terminations(_get_terminations(_fcc(species, a), miller, min_slab_size, 15.0))
    assert terms, "termination enumeration returned no slab"
    slab = terms[0][1]

    levels = _plane_levels(slab)
    gaps = np.diff(levels)
    assert len(levels) >= math.ceil(min_slab_size / d_hkl)
    assert np.allclose(gaps, d_hkl, atol=1e-3)
    assert gaps.max() <= 1.5 * d_hkl
    # The board really is as thick as the plane budget buys.
    assert levels[-1] - levels[0] >= (len(levels) - 1) * d_hkl - 1e-6


@pytest.mark.parametrize(
    ("species", "a"),
    [("Cu", 3.615), ("Ni", 3.524)],
)
def test_generated_slab_carries_real_vacuum_above_the_top_plane(species, a):
    terms = _flatten_terminations(_get_terminations(_fcc(species, a), (0, 0, 1), 8.0, 15.0))
    slab = terms[0][1]

    matrix = np.asarray(slab.lattice.matrix, dtype=float)
    normal = np.cross(matrix[0], matrix[1])
    normal = normal / np.linalg.norm(normal)
    projected = np.asarray(slab.cart_coords, dtype=float) @ normal
    height = float(matrix[2] @ normal)

    assert height - projected.max() > 5.0  # usable vacuum above the board
    assert slab.properties.get("ai_slab_generated") is True
    geometry = finite_slab_geometry(slab)
    assert geometry["finite_slab"] is True
    assert geometry["vacuum_reference"] is not None


def test_111_slab_spacing_uses_the_oriented_normal_not_cartesian_z():
    """A skewed (111) cell must be measured along ``cross(a1, a2)``.

    pymatgen re-orients the cell so the surface normal is the cross
    product of the in-plane vectors; for fcc (111) that axis is 54.7
    degrees away from cartesian z.  Sorting by z reports the in-plane
    lattice constant (3.615 A) instead of the interplanar spacing
    (a/sqrt(3) = 2.087 A).
    """
    terms = _flatten_terminations(_get_terminations(_fcc("Cu", 3.615), (1, 1, 1), 8.0, 15.0))
    slab = terms[0][1]

    matrix = np.asarray(slab.lattice.matrix, dtype=float)
    normal = np.cross(matrix[0], matrix[1])
    normal = normal / np.linalg.norm(normal)
    assert not np.allclose(np.abs(normal), [0.0, 0.0, 1.0], atol=1e-6)

    cartesian_z = np.asarray(slab.cart_coords, dtype=float)[:, 2]
    assert np.allclose(np.diff(_plane_levels(slab)), 3.615 / np.sqrt(3.0), atol=1e-3)
    # The naive z ordering is measurably different -- that is the bug the
    # projection guards against.
    assert not np.allclose(
        np.diff(np.unique(np.round(cartesian_z, 4))), 3.615 / np.sqrt(3.0), atol=1e-3
    )


def test_get_terminations_raises_naming_the_miller_indices_on_failure():
    bulk = _fcc("Cu", 3.615)
    with (
        patch("pymatgen.core.surface.SlabGenerator", side_effect=RuntimeError("bad slab")),
        pytest.raises(ValueError, match=r"\(1, 1, 1\)"),
    ):
        _get_terminations(bulk, (1, 1, 1), 8.0, 15.0)


def test_get_terminations_raises_when_every_cut_fails():
    bulk = _fcc("Cu", 3.615)
    with (
        patch("pymatgen.core.surface.SlabGenerator.get_slabs", return_value=[]),
        pytest.raises(ValueError, match="no termination slab"),
    ):
        _get_terminations(bulk, (0, 0, 1), 8.0, 15.0)


def test_generated_slab_is_a_finite_board_not_the_bulk_cell():
    """A real cut is a finite board with vacuum, not the bulk handed back."""
    bulk = _fcc("Cu", 3.615)
    terms = _flatten_terminations(_get_terminations(bulk, (0, 0, 1), 8.0, 15.0))
    slab = terms[0][1]

    matrix = np.asarray(slab.lattice.matrix, dtype=float)
    normal = np.cross(matrix[0], matrix[1])
    normal = normal / np.linalg.norm(normal)
    projected = np.asarray(slab.cart_coords, dtype=float) @ normal
    height = float(matrix[2] @ normal)

    # The board occupies only part of its cell: real vacuum above it.
    assert projected.max() - projected.min() < height
    assert height - projected.max() > 5.0
    assert slab.properties.get("ai_slab_generated") is True
    geometry = finite_slab_geometry(slab)
    assert geometry["finite_slab"] is True
    # The bulk reference carries no slab provenance, so it is never
    # certified as a finite board.
    assert finite_slab_geometry(bulk)["finite_slab"] is False
