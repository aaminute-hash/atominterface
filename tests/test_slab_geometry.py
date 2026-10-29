"""Slab-geometry gates: real ``SlabGenerator`` output, no mocks.

These tests pin the contract the builders rely on: a generated slab is a
finite board whose atomic planes are spaced by the theoretical
interplanar distance ``d_hkl`` of its Miller indices, with real vacuum
above the top plane.  pymatgen hands back cells whose in-plane vectors
carry a z component for every surface but (001), so :func:`_orient_slab`
rotates them level first -- after which ``cross(a1, a2)`` *is* the
cartesian z axis and the plain ``[:2]`` slices the builders live by are
exact rather than a silent projection.  The mirror-symmetric route cuts
and mirrors its own half-board, so it gets the same leveling.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from atominterface.builders.charge import finite_slab_geometry, make_symmetric_slab
from atominterface.builders.coherent import (
    CoherentInterfaceBuilder,
    CoherentInterfaceConfig,
    _flatten_terminations,
    _get_terminations,
    _make_slabs,
)

#: Tolerance (A) for "this cell component is zero".  Cell lengths are a few
#: A, so this is ~1e-10 relative -- far above float64 round-off, far below
#: any distance the builders care about.
LEVEL_TOL = 1e-9

#: (miller, interplanar spacing / a, in-plane area / a^2, Cu layers per 14.4 A).
#: Areas are written analytically rather than as the 4-dp table values
#: (6.5341 / 5.6587 / 9.2406): comparing against the rounded literal at a
#: tight relative tolerance is a tautology, not a gate.
ORIENTATIONS = (
    ((0, 0, 1), 1 / 2.0, 1 / 2.0, 8),  # a/2 = 1.8075 A, a^2/2 = 6.5341 A^2
    ((1, 1, 1), 1 / math.sqrt(3.0), math.sqrt(3.0) / 4.0, 7),  # 2.0871 A, 5.6587 A^2
    ((1, 1, 0), 1 / (2.0 * math.sqrt(2.0)), 1 / math.sqrt(2.0), 12),  # 1.2781 A, 9.2406 A^2
)


def _shortest_inplane_period(matrix: np.ndarray) -> float:
    """Shortest ``|m*a + n*b|`` over ``m, n`` in ``[-4, 4]``.

    Brute force rather than ``min(|a|, |b|)``: a reduced-but-not-minimal
    basis (``((1, 1), (0, 2))``, say) can have both basis vectors longer
    than a short combination of them, so taking ``|a|`` alone would read
    too long a period and let a degenerate cell pass.
    """
    best = float("inf")
    for m in range(-4, 5):
        for n in range(-4, 5):
            if m == 0 and n == 0:
                continue
            best = min(best, float(np.linalg.norm(m * matrix[0] + n * matrix[1])))
    return best


def _level_violation(matrix: np.ndarray) -> float:
    """Largest in-plane part of ``c`` / out-of-plane part of ``a`` or ``b``."""
    return float(
        max(
            abs(matrix[0][2]),
            abs(matrix[1][2]),
            abs(matrix[2][0]),
            abs(matrix[2][1]),
        )
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


def test_slabs_are_level_so_cartesian_z_is_the_surface_normal():
    """Every generated slab comes back with ``cross(a, b)`` along ``+z``.

    Before :func:`_orient_slab` re-levels the board, an fcc (111) slab was
    handed back as ``a = (1.81, 0, -1.81)`` -- the normals sat at 54.7
    degrees to cartesian z, so slicing ``[:2]`` projected the in-plane
    vectors onto xy (area 3.2671 instead of 5.6587) and reading heights
    off ``cart_coords[:, 2]`` reported the in-plane lattice constant
    (3.615 A) instead of the interplanar spacing (2.087 A).  Every
    measurement below is done the naive cartesian way on purpose: after
    the re-leveling those have to agree with the true d_hkl.
    """
    terms = _flatten_terminations(_get_terminations(_fcc("Cu", 3.615), (1, 1, 1), 8.0, 15.0))
    slab = terms[0][1]

    matrix = np.asarray(slab.lattice.matrix, dtype=float)
    normal = np.cross(matrix[0], matrix[1])
    normal = normal / np.linalg.norm(normal)
    assert np.allclose(np.abs(normal), [0.0, 0.0, 1.0], atol=1e-9)

    # The naive z ordering now measures the interplanar spacing.
    cartesian_z = np.asarray(slab.cart_coords, dtype=float)[:, 2]
    assert np.allclose(
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


# ---------------------------------------------------------------------------
# Orientation gates.  Everything below asserts the contract ``_orient_slab``
# establishes: an oriented slab's normal is exactly +z, so the ``[:2]`` slices
# every builder makes are exact rather than a projection onto xy.
# ---------------------------------------------------------------------------

MIN_THICKNESS = 14.4
#: fcc Cu nearest-neighbour distance, a/sqrt(2) = 2.556191 A -- the 2.5562 in
#: the acceptance note is this rounded to 4 dp.
NN_LIMIT = 3.615 / math.sqrt(2.0)


@pytest.mark.parametrize(
    ("miller", "spacing_factor", "area_factor", "expect_layers"),
    ORIENTATIONS,
    ids=["001", "111", "110"],
)
def test_generated_slab_is_level_along_z(miller, spacing_factor, area_factor, expect_layers):
    """``cross(a, b)`` must be the cartesian z axis, whatever the surface.

    This is the invariant the rest of the builder chain is built on: the
    in-plane vectors carry no z component and the vacuum column carries no
    in-plane part, so ``lattice[:2, :2]`` is the truth and reading heights
    off ``cart_coords[:, 2]`` is legitimate.
    """
    terms = _flatten_terminations(_get_terminations(_fcc("Cu", 3.615), miller, MIN_THICKNESS, 15.0))
    slab = terms[0][1]
    matrix = np.asarray(slab.lattice.matrix, dtype=float)

    assert _level_violation(matrix) < LEVEL_TOL
    normal = np.cross(matrix[0], matrix[1])
    normal = normal / np.linalg.norm(normal)
    assert np.allclose(normal, [0.0, 0.0, 1.0], atol=LEVEL_TOL)


@pytest.mark.parametrize(
    ("miller", "spacing_factor", "area_factor", "expect_layers"),
    ORIENTATIONS,
    ids=["001", "111", "110"],
)
def test_inplane_area_and_spacing_match_the_orientation(
    miller, spacing_factor, area_factor, expect_layers
):
    """True area ``|a x b|`` and true layer spacing ``d_hkl``, per surface.

    The expected values are the textbook fcc ones: the (111) board has
    ``sqrt(3)/4 a^2 = 5.6587 A^2`` of interface and planes every
    ``a/sqrt(3) = 2.0871 A`` -- the numbers the projected cell reported as
    3.2671 A^2 and 3.615 A.
    """
    terms = _flatten_terminations(_get_terminations(_fcc("Cu", 3.615), miller, MIN_THICKNESS, 15.0))
    slab = terms[0][1]
    matrix = np.asarray(slab.lattice.matrix, dtype=float)

    area = float(np.linalg.norm(np.cross(matrix[0], matrix[1])))
    assert area == pytest.approx(area_factor * 3.615**2, rel=1e-9)

    levels = np.unique(np.round(np.asarray(slab.cart_coords, dtype=float)[:, 2], 4))
    gaps = np.diff(levels)
    assert len(levels) == expect_layers
    assert np.allclose(gaps, spacing_factor * 3.615, atol=1e-3)


@pytest.mark.parametrize(
    ("miller", "spacing_factor", "area_factor", "expect_layers"),
    ORIENTATIONS,
    ids=["001", "111", "110"],
)
def test_assembled_interface_keeps_the_orientations_geometry(
    miller, spacing_factor, area_factor, expect_layers
):
    """End to end: the candidate the builder ships carries the same numbers.

    A degraded rebuild would show up here rather than only in the raw
    slab, because it is ``_assemble_interface`` that re-lays the cell from
    the in-plane block and pads the vacuum from a z coordinate.
    """
    config = CoherentInterfaceConfig(
        substrate_miller=miller, film_miller=miller, relax_terminations=False
    )
    builder = CoherentInterfaceBuilder(
        substrate=_fcc("Cu", 3.615), film=_fcc("Ni", 3.524), config=config
    )
    candidates = builder.build_all()
    assert candidates

    for candidate in candidates[:2]:
        cell = np.asarray(candidate.atoms.cell.array, dtype=float)
        assert _level_violation(cell) < LEVEL_TOL

        area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
        expected_area = (
            area_factor * 3.615**2 * candidate.metadata["substrate_supercell_multiplicity"]
        )
        assert area == pytest.approx(expected_area, rel=1e-9)
        assert candidate.area == pytest.approx(area, rel=1e-9)

        # No board here is denser than an fcc nearest-neighbour shell: the
        # shortest lattice period bounds every in-plane atom separation.
        assert _shortest_inplane_period(cell) >= NN_LIMIT - 1e-6


def test_001_area_is_pinned_not_merely_consistent():
    """Regression lock: the published gamma values are normalised by 6.5341 A^2.

    The area is asserted at its full precision (6.5341125 A^2, i.e.
    ``a^2/2``) rather than against the rounded 6.5341, because a relative
    tolerance of 1e-12 against the rounded literal could not be met even
    before the re-orientation -- it would be a tautology, not a gate.
    Any drift here moves gamma_Cu, gamma_Ni and gamma_int together.
    """
    config = CoherentInterfaceConfig(substrate_miller=(0, 0, 1), film_miller=(0, 0, 1))
    builder = CoherentInterfaceBuilder(
        substrate=_fcc("Cu", 3.615), film=_fcc("Ni", 3.524), config=config
    )
    candidates = builder.build_all()
    assert candidates

    candidate = candidates[0]
    assert candidate.metadata["substrate_supercell"] == ((1, 0), (0, 1))
    assert candidate.area == pytest.approx(6.534112500000001, rel=1e-12)
    cell = np.asarray(candidate.atoms.cell.array, dtype=float)
    assert float(np.linalg.norm(np.cross(cell[0], cell[1]))) == pytest.approx(
        6.534112500000001, rel=1e-12
    )


def test_oriented_slab_keeps_the_finite_slab_provenance():
    """Re-orienting must not strip the properties the charge layer reads."""
    terms = _flatten_terminations(_get_terminations(_fcc("Cu", 3.615), (1, 1, 1), 8.0, 15.0))
    slab = terms[0][1]

    assert slab.properties.get("ai_slab_generated") is True
    geometry = finite_slab_geometry(slab)
    assert geometry["finite_slab"] is True
    assert geometry["vacuum_reference"] is not None
    assert set(slab.site_properties) >= {"bulk_equivalent", "bulk_wyckoff"}


# ---------------------------------------------------------------------------
# Mirror-symmetric boards.  ``make_symmetric_slab`` cuts and mirrors its own
# half-board, so it needs the same leveling the plain route gets -- and it has
# to keep doing the charge bookkeeping correctly afterwards.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("miller", ORIENTATIONS, ids=lambda row: f"{row[0]}")
def test_symmetric_slab_is_level_along_z(miller):
    """A mirrored board must come back with its true normal along +z too.

    The mirroring helper itself projects onto ``cross(a, b)``, so it is
    orientation-agnostic -- which means nothing downstream of it will notice
    if the board it produced is skewed.  Before the leveling was added here
    an fcc (111) symmetric board carried a 1.807 A skew and a squared-up
    -3.2671 A^2- cell, and the builder assembled that silently.
    """
    result = make_symmetric_slab(_fcc("Cu", 3.615), miller[0], min_thickness=14.4, vacuum=15.0)
    board = result.structure
    matrix = np.asarray(board.lattice.matrix, dtype=float)

    assert _level_violation(matrix) < LEVEL_TOL
    area = float(np.linalg.norm(np.cross(matrix[0], matrix[1])))
    assert area == pytest.approx(result.area, rel=1e-12)
    # Layer counting off cartesian z now works, which it did not before:
    # the skewed cell read 8 planes at 2.0657 A instead of 9 at 2.0871 A.
    levels = np.unique(np.round(np.asarray(board.cart_coords, dtype=float)[:, 2], 4))
    gaps = np.diff(levels)
    assert np.allclose(gaps, miller[1] * 3.615, atol=1e-3)


def test_symmetric_111_board_keeps_its_charge_bookkeeping():
    """Leveling the board must leave the charge layer's verdicts alone.

    ``analyze_charge_polarity`` projects onto ``cross(a, b)``, so the
    polarity bookkeeping is invariant under the rotation -- this pins it,
    because the naive way to place the leveling call (after the analysis)
    would have rebuilt the board with its provenance property stripped.
    """
    board = make_symmetric_slab(_fcc("Cu", 3.615), (1, 1, 1), min_thickness=14.4, vacuum=15.0)

    assert board.structure.properties.get("ai_slab_generated") is True
    geometry = finite_slab_geometry(board.structure)
    assert geometry["finite_slab"] is True
    assert geometry["vacuum_reference"] is not None
    assert board.n_formula_units == pytest.approx(2.25, rel=1e-12)


def test_symmetric_interface_candidates_carry_the_true_area():
    """``symmetric_slabs=True`` through the builder, end to end."""
    config = CoherentInterfaceConfig(
        substrate_miller=(1, 1, 1),
        film_miller=(1, 1, 1),
        relax_terminations=False,
        symmetric_slabs=True,
    )
    builder = CoherentInterfaceBuilder(
        substrate=_fcc("Cu", 3.615), film=_fcc("Ni", 3.524), config=config
    )
    candidates = builder.build_all()
    assert candidates

    area_factor = ORIENTATIONS[1][2]
    for candidate in candidates[:2]:
        cell = np.asarray(candidate.atoms.cell.array, dtype=float)
        assert _level_violation(cell) < LEVEL_TOL
        expected = area_factor * 3.615**2 * candidate.metadata["substrate_supercell_multiplicity"]
        assert candidate.area == pytest.approx(expected, rel=1e-9)
        assert _shortest_inplane_period(cell) >= NN_LIMIT - 1e-6
