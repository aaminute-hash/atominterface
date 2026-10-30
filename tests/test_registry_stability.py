"""The lateral registry of a non-(001) interface must not depend on thickness.

Stacking the two boards as they come puts the film's join layer on the
sublattice pymatgen's cut left it on and the substrate's join layer on the
one its own layer count implies.  On fcc (111) that is sublattice
``(n_sub - 1) mod 3``, so every third ``--min-thickness`` step lands the
two exactly on top of each other -- and an exactly opposed stack sits on a
mirror-symmetric stationary point whose in-plane gradient vanishes by
symmetry, so the relaxation converges where it started and stays.  Measured
on Cu/Ni(111) before the fix: 10.8 / 14.4 / 18.0 / 21.6 A gave gamma_int
0.081 / 0.898 / 0.081 / 0.081 J/m^2, an 11x step from the thickness knob
alone, with nothing sliding during relaxation.

These tests pin three things:

* the (111) registry is off the symmetric point and the same number at
  every thickness, including the ``n_sub = 1 (mod 3)`` family that used to
  collapse to zero;
* (001) and (110) come out **bit-identical** to the build the fix does not
  touch -- compared against that build directly, so the guarantee is
  structural rather than a tolerance;
* the shift helper itself fires on exactly one layer-count family.

No MLIP is involved: everything here is decided during construction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atominterface.builders.coherent import (
    ON_TOP_LATERAL_TOL,
    CoherentInterfaceBuilder,
    CoherentInterfaceConfig,
    _flatten_terminations,
    _get_terminations,
    _off_on_top_shift,
    _strain_film_onto,
)

#: The four thicknesses the validation matrix documents.
THICKNESSES = (10.8, 14.4, 18.0, 21.6)

#: Extra (111) points whose Cu slab is 1 (mod 3) layers -- 4, 7 and 10
#: planes.  The four documented thicknesses happen to hit that family
#: exactly once (14.4 A, 7 planes), so on their own they cannot tell
#: "fixed" apart from "the one bad point was special-cased".
EXTRA_111_ON_TOP_FAMILY = (8.0, 13.0, 19.0)

#: Lateral offset of a film atom sitting on a hollow site of the Cu (111)
#: surface: a_nn / sqrt(3) = (3.615 / sqrt(2)) / sqrt(3).
HOLLOW_LATERAL_111 = 1.4758175700268625

#: Lateral offset of the (001) and (110) joins, which the fix must leave
#: alone.  Both are half an in-plane lattice vector of their own cell.
AS_BUILT_LATERAL_001_110 = 1.8075


def _fcc(species: str, a: float):
    """Conventional fcc cell, as a bulk input for slab generation."""
    try:
        from pymatgen.core import Lattice, Structure  # noqa: WPS433
    except ImportError:
        pytest.skip("pymatgen not installed")

    coords: list[list[float]] = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    return Structure(Lattice.cubic(a), [species] * 4, coords)


def _build(miller, min_thickness):
    """First candidate of a Cu/Ni interface at ``miller`` and ``min_thickness``."""
    config = CoherentInterfaceConfig(
        substrate_miller=miller, film_miller=miller, min_thickness=min_thickness
    )
    builder = CoherentInterfaceBuilder(
        substrate=_fcc("Cu", 3.615), film=_fcc("Ni", 3.524), config=config
    )
    candidates = builder.build()
    assert candidates
    return candidates[0]


def _registry(miller, min_thickness) -> float:
    return float(_build(miller, min_thickness).metadata["registry_lateral_offset_ang"])


@pytest.mark.parametrize("min_thickness", THICKNESSES + EXTRA_111_ON_TOP_FAMILY)
def test_111_registry_is_off_the_symmetric_point_at_every_thickness(min_thickness):
    """No (111) thickness may ship an exactly opposed join.

    ``> 0.5`` is the load-bearing bound: it is far below the hollow-site
    offset (1.4758 A) and far above :data:`ON_TOP_LATERAL_TOL`, so it
    fails both on a collapse to zero and on a nudge that only moved the
    film by round-off.
    """
    registry = _registry((1, 1, 1), min_thickness)
    assert registry > 0.5
    assert registry == pytest.approx(HOLLOW_LATERAL_111, abs=1e-6)


def test_111_registry_is_the_same_number_at_every_thickness():
    """The property the defect violated, asserted directly.

    A thickness scan is supposed to vary one thing.  Comparing the
    registry across the scan is what makes that checkable, so it is
    asserted here rather than left to seven separate approx() calls --
    those would pass on any value that happens to sit near one number.
    """
    registries = [_registry((1, 1, 1), t) for t in THICKNESSES + EXTRA_111_ON_TOP_FAMILY]
    assert max(registries) - min(registries) < 1e-6


def test_111_on_top_family_no_longer_warns_about_being_exactly_opposed():
    """Criterion D used to append its on-top warning at 14.4 A and nowhere else.

    The warning was right and the geometry was wrong, so it is the
    geometry that moved.  Asserted on the warning's absence so a future
    regression cannot quietly bring the symmetric point back.
    """
    import warnings

    config = CoherentInterfaceConfig(
        substrate_miller=(1, 1, 1), film_miller=(1, 1, 1), min_thickness=14.4
    )
    builder = CoherentInterfaceBuilder(
        substrate=_fcc("Cu", 3.615), film=_fcc("Ni", 3.524), config=config
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        candidates = builder.build()

    assert candidates
    opposed = [w for w in caught if "exactly opposed" in str(w.message)]
    assert opposed == []


@pytest.mark.parametrize("min_thickness", THICKNESSES)
def test_001_positions_are_bit_identical_without_the_shift(min_thickness, monkeypatch):
    """(001) must not move by even the last bit.

    Compared against a build that has the shift switched off, which is the
    pre-fix geometry exactly.  ``array_equal`` and not ``allclose``: the
    claim is that the shift path is a no-op here, and a tolerance would
    turn "we changed (001)" into a passing test as long as the change was
    small.
    """
    from atominterface.builders import coherent

    monkeypatch.setattr(coherent, "_OFF_ON_TOP_MIN_SHIFT", math.inf)
    without = _build((0, 0, 1), min_thickness)
    monkeypatch.undo()
    with_shift = _build((0, 0, 1), min_thickness)

    assert np.array_equal(without.atoms.get_positions(), with_shift.atoms.get_positions())
    assert np.array_equal(np.asarray(without.atoms.cell), np.asarray(with_shift.atoms.cell))
    assert float(with_shift.metadata["registry_lateral_offset_ang"]) == pytest.approx(
        AS_BUILT_LATERAL_001_110, abs=1e-6
    )


@pytest.mark.parametrize("min_thickness", THICKNESSES)
def test_110_positions_are_bit_identical_without_the_shift(min_thickness, monkeypatch):
    """(110) is stable across thickness already; the fix must not disturb it.

    Pinned explicitly because its as-built registry is *biased* (the ridge
    site reads 0.5953 J/m^2 against a scanned 0.088) but it is not
    *thickness-dependent* -- a different defect, with a different fix, and
    moving it here would conflate the two.
    """
    from atominterface.builders import coherent

    monkeypatch.setattr(coherent, "_OFF_ON_TOP_MIN_SHIFT", math.inf)
    without = _build((1, 1, 0), min_thickness)
    monkeypatch.undo()
    with_shift = _build((1, 1, 0), min_thickness)

    assert np.array_equal(without.atoms.get_positions(), with_shift.atoms.get_positions())
    assert float(with_shift.metadata["registry_lateral_offset_ang"]) == pytest.approx(
        AS_BUILT_LATERAL_001_110, abs=1e-6
    )


#: Cartesian positions of Cu/Ni(001) at 10.8 A, as recorded by the
#: validation matrix before the fix.  A literal, so the file carries one
#: hard external anchor instead of only self-referential comparisons.
PINNED_001_10P8_POSITIONS = [
    [0.0, 0.0, 11.748750000000001],
    [1.2780955069946844, 1.2780955069946847, 9.941249999999998],
    [0.0, 0.0, 15.363750000000001],
    [1.2780955069946844, 1.2780955069946847, 13.55625],
    [0.0, 0.0, 18.97875],
    [1.2780955069946844, 1.2780955069946847, 17.171249999999997],
    [0.0, 0.0, 22.74075],
    [-1.2780955069946842, 1.2780955069946847, 20.97875],
    [0.0, 0.0, 26.26475],
    [-1.2780955069946842, 1.2780955069946847, 24.502750000000006],
    [0.0, 0.0, 29.78875],
    [-1.2780955069946842, 1.2780955069946847, 28.02675],
    [0.0, 0.0, 33.31275],
    [-1.2780955069946842, 1.2780955069946847, 31.55075],
]


def test_001_positions_match_the_recorded_baseline():
    """The (001) geometry the published numbers were measured on, verbatim."""
    candidate = _build((0, 0, 1), 10.8)
    positions = candidate.atoms.get_positions()
    assert len(positions) == len(PINNED_001_10P8_POSITIONS)
    assert np.array_equal(np.asarray(positions, dtype=float), np.asarray(PINNED_001_10P8_POSITIONS))


def _slab(species: str, a: float, miller, min_thickness):
    return _flatten_terminations(_get_terminations(_fcc(species, a), miller, min_thickness, 15.0))[
        0
    ][1]


@pytest.mark.parametrize(
    ("min_thickness", "n_cu_planes", "expected_shift"),
    [
        (10.8, 6, 0.0),
        (14.4, 7, HOLLOW_LATERAL_111),
        (15.0, 8, 0.0),
        (17.0, 9, 0.0),
    ],
)
def test_off_on_top_shift_fires_on_one_layer_count_family(
    min_thickness, n_cu_planes, expected_shift
):
    """The helper is a function of ``n_sub mod 3`` and of nothing else.

    ``n_cu_planes`` is asserted so the parametrization stays honest: the
    whole point is which residue each thickness lands on, and a thickness
    that stopped cutting 7 planes would silently turn this into a test of
    the wrong family.
    """
    substrate = _slab("Cu", 3.615, (1, 1, 1), min_thickness)
    film = _slab("Ni", 3.524, (1, 1, 1), min_thickness)
    sub_2d = np.asarray(substrate.lattice.matrix, dtype=np.float64)[:2, :2]

    # Same frames ``_prepare_stack`` hands ``_assemble_interface``: the
    # film strained onto the substrate cell, both boards levelled.
    strained = _strain_film_onto(film, sub_2d)
    shift = _off_on_top_shift(
        np.asarray(substrate.cart_coords, dtype=np.float64),
        np.asarray(strained.cart_coords, dtype=np.float64),
        sub_2d,
    )

    # 7 planes puts the substrate's join layer on the film's sublattice;
    # 6, 8 and 9 do not.
    z = np.asarray(substrate.cart_coords, dtype=np.float64)[:, 2]
    steps = np.diff(np.sort(z))
    assert 1 + int(np.count_nonzero(steps > 1e-3)) == n_cu_planes

    assert float(np.linalg.norm(shift)) == pytest.approx(expected_shift, abs=1e-9)
    if expected_shift == 0.0:
        assert np.array_equal(shift, np.zeros(2))


def test_off_on_top_tol_is_the_trigger_the_helper_uses():
    """The trigger is the same tolerance the criterion-D message claims.

    Two thresholds for "on top" would be two different answers to the same
    question, so the helper borrows the published one rather than growing
    a private copy.
    """
    assert ON_TOP_LATERAL_TOL == 1e-6
