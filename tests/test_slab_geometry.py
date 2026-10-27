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
from atominterface.builders.coherent import (
    CoherentInterfaceBuilder,
    CoherentInterfaceConfig,
    _flatten_terminations,
    _get_terminations,
    _make_slabs,
)


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


def _normal_metrics(structure) -> tuple[float, float]:
    """``(board extent, total vacuum)`` measured along the slab's surface normal."""
    matrix = np.asarray(structure.lattice.matrix, dtype=float)
    normal = np.cross(matrix[0], matrix[1])
    normal = normal / np.linalg.norm(normal)
    projected = np.asarray(structure.cart_coords, dtype=float) @ normal
    height = float(matrix[2] @ normal)
    extent = float(projected.max() - projected.min())
    return extent, height - extent


def test_slab_thicknesses_fall_back_to_the_shared_min_thickness():
    """No per-side override must behave exactly like the pre-override code."""
    config = CoherentInterfaceConfig(min_thickness=8.0)
    assert config.slab_thicknesses() == (8.0, 8.0)

    sub_terms, film_terms = _make_slabs(_fcc("Cu", 3.615), _fcc("Ni", 3.524), config)
    expected_sub = _flatten_terminations(
        _get_terminations(_fcc("Cu", 3.615), (0, 0, 1), 8.0, config.vacuum_thickness)
    )
    expected_film = _flatten_terminations(
        _get_terminations(_fcc("Ni", 3.524), (0, 0, 1), 8.0, config.vacuum_thickness)
    )
    assert [len(slab) for _, slab in sub_terms] == [len(slab) for _, slab in expected_sub]
    assert [len(slab) for _, slab in film_terms] == [len(slab) for _, slab in expected_film]


def test_per_side_min_thickness_overrides_apply_independently():
    """Each side is cut at its own resolved thickness, not the shared one."""
    config = CoherentInterfaceConfig(
        min_thickness=4.0,
        min_thickness_substrate=12.0,
        min_thickness_film=6.0,
        relax_terminations=False,
    )
    assert config.slab_thicknesses() == (12.0, 6.0)

    sub_terms, film_terms = _make_slabs(_fcc("Cu", 3.615), _fcc("Ni", 3.524), config)
    sub_slab = sub_terms[0][1]
    film_slab = film_terms[0][1]

    # Each side equals a direct cut at its own override...
    expected_sub = _flatten_terminations(
        _get_terminations(_fcc("Cu", 3.615), (0, 0, 1), 12.0, 15.0)
    )
    expected_film = _flatten_terminations(
        _get_terminations(_fcc("Ni", 3.524), (0, 0, 1), 6.0, 15.0)
    )
    assert len(sub_slab) == len(expected_sub[0][1])
    assert len(film_slab) == len(expected_film[0][1])
    # ...and the substrate really is thicker than the film, which a single
    # shared min_thickness could never express.
    sub_extent, _ = _normal_metrics(sub_slab)
    film_extent, _ = _normal_metrics(film_slab)
    assert sub_extent >= 12.0 - 3.615  # within one oriented-cell repeat of the request
    assert sub_extent > film_extent


def test_vacuum_is_a_fixed_amount_independent_of_slab_thickness():
    """Thicker slabs must not eat into (or inflate) the vacuum budget.

    ``SlabGenerator`` spends ``min_vacuum_size`` in whole
    oriented-unit-cell repeats of its own -- the repeat count never
    involves ``min_slab_size`` -- so the vacuum width is bit-identical
    for any slab thickness.  This test pins that contract.
    """
    bulk = _fcc("Cu", 3.615)
    vacuums = []
    extents = []
    for min_thickness in (6.0, 12.0):
        slab = _flatten_terminations(_get_terminations(bulk, (0, 0, 1), min_thickness, 15.0))[0][1]
        extent, vacuum = _normal_metrics(slab)
        extents.append(extent)
        vacuums.append(vacuum)
    # Sanity: the two cuts really differ in board thickness.
    assert extents[1] > extents[0] + 3.0
    # The fixed part: identical vacuum, and at least the requested 15 A.
    assert vacuums[0] == pytest.approx(vacuums[1], abs=1e-6)
    assert vacuums[0] >= 15.0


def test_assembled_interface_vacuum_not_swallowed_by_thicker_substrate():
    """``_assemble_interface`` pads exactly ``vacuum_thickness`` above the
    topmost atom; a thicker substrate changes the cell height, not the pad.
    """
    cu = _fcc("Cu", 3.615)
    ni = _fcc("Ni", 3.524)
    top_pads = []
    for overrides in ({}, {"min_thickness_substrate": 12.0}):
        config = CoherentInterfaceConfig(
            min_thickness=6.0, vacuum_thickness=15.0, relax_terminations=False, **overrides
        )
        builder = CoherentInterfaceBuilder(substrate=cu, film=ni, config=config)
        candidates = builder.build_all()
        assert candidates
        atoms = candidates[0].atoms
        cell_height = float(np.asarray(atoms.cell.array)[2, 2])
        top_pads.append(cell_height - float(atoms.get_positions()[:, 2].max()))
    assert top_pads[0] == pytest.approx(15.0, abs=1e-8)
    assert top_pads[1] == pytest.approx(15.0, abs=1e-8)


def test_symmetric_slabs_respect_per_side_thickness():
    """``symmetric_slabs=True`` must use each side's resolved thickness too.

    The assertion compares against boards built with the shared
    ``min_thickness`` set to the same value, so the test does not depend
    on exactly how many layers the mirroring loop produces.
    """
    cu = _fcc("Cu", 3.615)
    overridden = CoherentInterfaceConfig(
        min_thickness=6.0,
        min_thickness_substrate=12.0,
        symmetric_slabs=True,
        relax_terminations=False,
    )
    plain = CoherentInterfaceConfig(
        min_thickness=6.0, symmetric_slabs=True, relax_terminations=False
    )
    thick_shared = CoherentInterfaceConfig(
        min_thickness=12.0, symmetric_slabs=True, relax_terminations=False
    )

    sub_terms, film_terms = _make_slabs(cu, cu, overridden)
    plain_terms, _ = _make_slabs(cu, cu, plain)
    thick_terms, _ = _make_slabs(cu, cu, thick_shared)

    # Substrate side matches a shared-12 A build; film side matches the
    # plain shared-6 A build.
    assert [len(s) for _, s in sub_terms] == [len(s) for _, s in thick_terms]
    assert [len(s) for _, s in film_terms] == [len(s) for _, s in plain_terms]
    assert len(sub_terms[0][1]) > len(film_terms[0][1])


@pytest.mark.parametrize("field", ["min_thickness_substrate", "min_thickness_film"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_per_side_thickness_must_be_positive_when_provided(field, value):
    config = CoherentInterfaceConfig(**{field: value})
    assert any(field in error for error in config.validate())


def test_per_side_thickness_none_or_positive_passes_validation():
    assert not CoherentInterfaceConfig().validate()
    assert not CoherentInterfaceConfig(
        min_thickness_substrate=3.0, min_thickness_film=9.0
    ).validate()
