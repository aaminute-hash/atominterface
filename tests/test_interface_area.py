"""Bilateral interface area resolution.

The energy ledger has to know *which* side's area a normalisation came
from, so the builder records both sides and a verdict instead of handing
downstream a single scalar that quietly assumes the two agree.  The one
thing this module must never allow is a geometric mean standing in for an
unresolved mismatch: that would make a real disagreement look handled.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atominterface.builders.base import Candidate
from atominterface.builders.coherent import (
    AREA_POLICIES,
    CoherentInterfaceConfig,
    InterfaceAreas,
    interface_areas,
)


def _ortho(a: float, b: float, c: float = 10.0) -> np.ndarray:
    """Orthogonal cell with the given in-plane edges (Å)."""
    return np.array([[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]], dtype=np.float64)


def _skew(a: float, b: float, angle_deg: float, c: float = 10.0) -> np.ndarray:
    """Non-orthogonal in-plane cell: ``a`` along x, ``b`` at ``angle_deg``."""
    ang = math.radians(angle_deg)
    return np.array(
        [[a, 0.0, 0.0], [b * math.cos(ang), b * math.sin(ang), 0.0], [0.0, 0.0, c]],
        dtype=np.float64,
    )


def _expected_area(lattice: np.ndarray) -> float:
    """Independent ``|a x b|`` reference, deliberately not the code under test."""
    mat = np.asarray(lattice, dtype=np.float64)
    return float(np.linalg.norm(np.cross(mat[0], mat[1])))


# --------------------------------------------------------------------------
# Consistent stacks
# --------------------------------------------------------------------------


def test_equal_sides_are_consistent_and_share_one_area():
    lattice = _ortho(2.5, 3.0)
    areas = interface_areas(lattice, lattice)

    assert areas.consistent is True
    assert areas.area_substrate == pytest.approx(7.5)
    assert areas.area_film == pytest.approx(7.5)
    assert areas.area_used == pytest.approx(7.5)
    # One number, three names: there is nothing to choose between.
    assert areas.area_used == areas.area_substrate == areas.area_film
    assert areas.policy == "strict"


def test_equal_sides_agree_under_every_policy():
    lattice = _ortho(2.5, 3.0)
    for policy in AREA_POLICIES:
        areas = interface_areas(lattice, lattice, policy=policy)
        assert areas.consistent is True, policy
        assert areas.area_used == pytest.approx(7.5), policy
        assert areas.policy == policy


# --------------------------------------------------------------------------
# Inconsistent stacks: record both, never average
# --------------------------------------------------------------------------


def test_mismatch_records_both_real_areas():
    substrate = _ortho(2.5, 3.0)  # 7.5 Å²
    film = _ortho(2.0, 3.0)  # 6.0 Å²

    areas = interface_areas(substrate, film)

    assert areas.consistent is False
    # Both honest numbers survive; neither is folded into the other.
    assert areas.area_substrate == pytest.approx(7.5)
    assert areas.area_film == pytest.approx(6.0)
    assert areas.area_substrate == pytest.approx(_expected_area(substrate))
    assert areas.area_film == pytest.approx(_expected_area(film))
    # The note names both numbers, so a human reading the ledger sees why.
    assert "7.5" in areas.note and "6" in areas.note


def test_strict_policy_refuses_to_pick_a_side():
    substrate = _ortho(2.5, 3.0)
    film = _ortho(2.0, 3.0)

    areas = interface_areas(substrate, film, policy="strict")

    assert areas.consistent is False
    assert math.isnan(areas.area_used)


@pytest.mark.parametrize("policy", list(AREA_POLICIES))
def test_no_geometric_mean_is_ever_substituted(policy):
    """The core prohibition: a mismatch must never be averaged away."""
    substrate = _ortho(2.5, 3.0)  # 7.5 Å²
    film = _ortho(2.0, 3.0)  # 6.0 Å²
    geometric_mean = math.sqrt(7.5 * 6.0)

    areas = interface_areas(substrate, film, policy=policy)

    assert areas.consistent is False
    assert areas.area_used != pytest.approx(geometric_mean)
    assert not math.isclose(areas.area_used, geometric_mean)
    assert areas.area_used != geometric_mean


def test_policy_substrate_takes_the_substrate_side():
    substrate = _ortho(2.5, 3.0)
    film = _ortho(2.0, 3.0)

    areas = interface_areas(substrate, film, policy="substrate")

    assert areas.consistent is False
    assert areas.area_used == pytest.approx(7.5)
    assert areas.area_used == areas.area_substrate
    assert areas.policy == "substrate"
    assert "substrate" in areas.note


def test_policy_film_takes_the_film_side():
    substrate = _ortho(2.5, 3.0)
    film = _ortho(2.0, 3.0)

    areas = interface_areas(substrate, film, policy="film")

    assert areas.consistent is False
    assert areas.area_used == pytest.approx(6.0)
    assert areas.area_used == areas.area_film
    assert areas.policy == "film"
    assert "film" in areas.note


def test_unknown_policy_is_rejected_by_the_resolver():
    with pytest.raises(ValueError, match="area_policy"):
        interface_areas(_ortho(2.0, 2.0), _ortho(2.0, 2.0), policy="average")


# --------------------------------------------------------------------------
# Geometry: non-orthogonal cells
# --------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [30.0, 60.0, 75.0, 100.0, 120.0])
def test_non_orthogonal_cell_area_uses_the_cross_product(angle):
    lattice = _skew(2.0, 3.0, angle)
    expected = 2.0 * 3.0 * math.sin(math.radians(angle))

    areas = interface_areas(lattice, lattice)

    assert areas.consistent is True
    assert areas.area_substrate == pytest.approx(expected)
    assert areas.area_film == pytest.approx(expected)
    assert areas.area_used == pytest.approx(expected)
    # A 2x2 |det| would be wrong for any of these; |a x b| is not.
    assert areas.area_used == pytest.approx(_expected_area(lattice))


def test_non_orthogonal_mismatch_is_detected():
    substrate = _skew(2.0, 3.0, 60.0)
    film = _skew(2.0, 3.0, 80.0)

    areas = interface_areas(substrate, film)

    assert areas.consistent is False
    assert areas.area_substrate == pytest.approx(_expected_area(substrate))
    assert areas.area_film == pytest.approx(_expected_area(film))
    assert areas.area_substrate > 0 and areas.area_film > 0
    assert areas.area_substrate != pytest.approx(areas.area_film)


def test_rotated_but_equal_cells_stay_consistent():
    """Same area, different shape: still one area, still consistent."""
    substrate = _ortho(2.0, 3.0)
    film = _skew(3.0, 2.0, 90.0)

    areas = interface_areas(substrate, film)

    assert areas.area_substrate == pytest.approx(6.0)
    assert areas.area_film == pytest.approx(6.0)
    assert areas.consistent is True


def test_two_by_two_in_plane_slice_is_accepted():
    """Legacy callers hand over a bare 2x2 in-plane slice."""
    areas = interface_areas(np.array([[2.0, 0.0], [0.0, 3.0]]), np.array([[2.0, 0.0], [0.0, 3.0]]))

    assert areas.consistent is True
    assert areas.area_used == pytest.approx(6.0)


# --------------------------------------------------------------------------
# Tolerance boundary
# --------------------------------------------------------------------------


def test_relative_difference_just_inside_the_tolerance_is_consistent():
    tolerance = 1e-6
    substrate = _ortho(1.0, 10.0)  # 10 Å²
    film = _ortho(1.0, 10.0 * (1.0 - 0.5 * tolerance))  # 5e-6 Å² smaller

    areas = interface_areas(substrate, film, tolerance=tolerance)

    assert areas.consistent is True
    assert areas.area_used == pytest.approx(10.0)


def test_relative_difference_just_outside_the_tolerance_is_inconsistent():
    tolerance = 1e-6
    substrate = _ortho(1.0, 10.0)  # 10 Å²
    film = _ortho(1.0, 10.0 * (1.0 - 2.0 * tolerance))  # 2e-5 Å² smaller

    areas = interface_areas(substrate, film, tolerance=tolerance)

    assert areas.consistent is False
    assert math.isnan(areas.area_used)


def test_tolerance_is_relative_not_absolute():
    """A 1e-6 Å² gap is fatal on a 1 Å² cell and invisible on a 1e6 Å² one."""
    small = interface_areas(_ortho(1.0, 1.0), _ortho(1.0, 1.0 - 1e-3))
    large = interface_areas(_ortho(1e3, 1e3), _ortho(1e3, 1e3 - 1e-3))

    assert small.consistent is False
    assert large.consistent is True


def test_a_wider_tolerance_absorbs_the_same_mismatch():
    substrate = _ortho(2.5, 3.0)
    film = _ortho(2.0, 3.0)

    assert interface_areas(substrate, film).consistent is False
    assert interface_areas(substrate, film, tolerance=0.5).consistent is True
    assert interface_areas(substrate, film, tolerance=0.5).area_used == pytest.approx(7.5)


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("policy", list(AREA_POLICIES))
def test_config_accepts_every_declared_policy(policy):
    errors = CoherentInterfaceConfig(area_policy=policy).validate()
    assert not [e for e in errors if "area_policy" in e]


@pytest.mark.parametrize("policy", ["average", "geometric", "", "STRICT", "mean"])
def test_config_rejects_unknown_area_policy(policy):
    errors = CoherentInterfaceConfig(area_policy=policy).validate()
    assert any("area_policy" in e for e in errors)


def test_default_area_policy_is_strict():
    assert CoherentInterfaceConfig().area_policy == "strict"
    assert "strict" in AREA_POLICIES


def test_interface_areas_is_a_frozen_record():
    areas = interface_areas(_ortho(2.0, 2.0), _ortho(2.0, 2.0))
    assert isinstance(areas, InterfaceAreas)
    with pytest.raises((AttributeError, TypeError)):
        areas.area_used = 1.0  # type: ignore[misc]


# --------------------------------------------------------------------------
# Candidate plumbing
# --------------------------------------------------------------------------


def _candidate(**overrides) -> Candidate:
    values = dict(
        atoms=None,
        interface_id="test:0-0",
        builder_name="coherent",
        substrate_formula="Cu",
        film_formula="Ni",
        substrate_miller=(0, 0, 1),
        film_miller=(0, 0, 1),
        area=12.5,
    )
    values.update(overrides)
    return Candidate(**values)


def test_candidate_falls_back_to_area_for_unmeasured_sides():
    """Hand-built candidates only ever set ``area``; they must not read NaN."""
    cand = _candidate(area=12.5)

    assert cand.area == pytest.approx(12.5)
    assert cand.area_substrate == pytest.approx(12.5)
    assert cand.area_film == pytest.approx(12.5)
    assert cand.area_consistent is True
    assert cand.area_policy == "strict"


def test_candidate_keeps_explicitly_measured_sides():
    cand = _candidate(
        area=12.5,
        area_substrate=12.5,
        area_film=11.0,
        area_consistent=False,
        area_policy="film",
    )

    assert cand.area == pytest.approx(12.5)
    assert cand.area_substrate == pytest.approx(12.5)
    assert cand.area_film == pytest.approx(11.0)
    assert cand.area_consistent is False
    assert cand.area_policy == "film"


def test_candidate_default_area_falls_back_to_zero_not_nan():
    cand = _candidate(area=0.0)

    assert cand.area_substrate == 0.0
    assert cand.area_film == 0.0


# --------------------------------------------------------------------------
# Through the builder: a matched stack versus a legacy unstrained one
# --------------------------------------------------------------------------


def _bulk(species: str, a: float):
    """Conventional fcc cell, as a bulk input for the coherent builder."""
    try:
        from pymatgen.core import Lattice, Structure  # noqa: WPS433
    except ImportError:
        pytest.skip("pymatgen not installed")

    coords: list[list[float]] = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    return Structure(Lattice.cubic(a), [species] * 4, coords)


def _config(**overrides):
    values = dict(
        substrate_miller=(0, 0, 1),
        film_miller=(0, 0, 1),
        max_strain=0.05,
        max_area=100.0,
        gap=2.0,
        min_thickness=6.0,
        vacuum_thickness=15.0,
        relax_terminations=False,
    )
    values.update(overrides)
    return CoherentInterfaceConfig(**values)


def test_matched_stack_records_one_shared_area():
    """A coherent match strains the film onto the substrate: one area, recorded."""
    from atominterface.builders.coherent import CoherentInterfaceBuilder

    builder = CoherentInterfaceBuilder(
        substrate=_bulk("Cu", 3.61), film=_bulk("Ni", 3.52), config=_config()
    )
    candidates = builder.build_all()
    assert candidates, "builder returned nothing; the area record cannot be checked"

    for cand in candidates:
        assert cand.area_consistent is True
        assert cand.area_substrate == pytest.approx(cand.area)
        assert cand.area_film == pytest.approx(cand.area)
        assert cand.area_policy == "strict"
        assert cand.metadata["area_consistent"] is True
        assert cand.metadata["area_used"] == pytest.approx(cand.area)
        assert cand.metadata["area_substrate"] == pytest.approx(cand.metadata["area_film"])
        assert cand.metadata["area_policy"] == "strict"


def test_legacy_unstrained_stack_is_marked_and_not_raised():
    """No match means no strain: the sides disagree, so they are recorded as such.

    This is the reachable asymmetric case the area policy exists for.  It
    must stay constructible -- turning it into an exception would break
    the legacy stacking path outright.
    """
    from atominterface.builders.coherent import (
        _assemble_interface,
        _get_terminations,
        _MatchInfo,
        _to_candidate,
    )

    config = _config()
    substrate_slab = _get_terminations(_bulk("Cu", 3.61), (0, 0, 1), 6.0, 15.0)[0][1]
    film_slab = _get_terminations(_bulk("Ni", 3.52), (0, 0, 1), 6.0, 15.0)[0][1]
    match = _MatchInfo(
        match_area=0.0,
        substrate_supercell=((1, 0), (0, 1)),
        film_supercell=((1, 0), (0, 1)),
        strain=0.0,
        shear_strain=0.0,
    )

    interface = _assemble_interface(substrate_slab, film_slab, None, config)
    cand = _to_candidate(interface, substrate_slab, film_slab, match, 0, 0, config)

    # Primitive fcc (001) surface cells: a^2 / 2, and the two differ.
    assert cand.area_substrate == pytest.approx(3.61**2 / 2)
    assert cand.area_film == pytest.approx(3.52**2 / 2)
    assert cand.area_consistent is False
    assert cand.area_policy == "strict"
    # Both real numbers survive in the metadata; neither is an average.
    assert cand.metadata["area_substrate"] == pytest.approx(3.61**2 / 2)
    assert cand.metadata["area_film"] == pytest.approx(3.52**2 / 2)
    assert math.isnan(cand.metadata["area_used"])
    assert cand.metadata["area_used"] != math.sqrt((3.61**2 / 2) * (3.52**2 / 2))
    # The stack's own cell is still the substrate cell, unchanged.
    assert cand.area == pytest.approx(3.61**2 / 2)


def test_legacy_stack_under_a_committed_policy_names_its_side():
    """``area_policy`` decides which side the ledger divides by."""
    from atominterface.builders.coherent import (
        _assemble_interface,
        _get_terminations,
        _MatchInfo,
        _to_candidate,
    )

    substrate_slab = _get_terminations(_bulk("Cu", 3.61), (0, 0, 1), 6.0, 15.0)[0][1]
    film_slab = _get_terminations(_bulk("Ni", 3.52), (0, 0, 1), 6.0, 15.0)[0][1]
    match = _MatchInfo(
        match_area=0.0,
        substrate_supercell=((1, 0), (0, 1)),
        film_supercell=((1, 0), (0, 1)),
        strain=0.0,
        shear_strain=0.0,
    )

    for policy, expected in (("substrate", 3.61**2 / 2), ("film", 3.52**2 / 2)):
        config = _config(area_policy=policy)
        interface = _assemble_interface(substrate_slab, film_slab, None, config)
        cand = _to_candidate(interface, substrate_slab, film_slab, match, 0, 0, config)

        assert cand.area_consistent is False
        assert cand.area_policy == policy
        assert cand.metadata["area_used"] == pytest.approx(expected)
