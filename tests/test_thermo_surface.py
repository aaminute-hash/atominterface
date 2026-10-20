"""Tests for bulk-referenced termination energies and deterministic ranking."""

from __future__ import annotations

import warnings

import pytest
from ase import Atoms

from atominterface.thermo import (
    TerminationRecord,
    rank_terminations,
    slab_formula_units,
    termination_energy,
)
from atominterface.thermo.surface import _EV_A2_TO_UNIT  # noqa: WPS433


def _record(label, **overrides):
    values = dict(
        label=label,
        formula="NaCl",
        stoichiometric=True,
        tasker_type=1,
        nominal_charge=0.0,
        area=10.0,
        n_atoms=8,
        slab_energy=None,
        bulk_energy=None,
        n_formula_units=None,
        n_surfaces=2,
        stretch=0.01,
        shear=0.02,
        symmetric=True,
    )
    values.update(overrides)
    return TerminationRecord(**values)


def test_termination_energy_matches_hand_calculation():
    # (12.0 - 4 * 2.5) / (2 * 10.0) = 0.1 eV/A^2
    assert termination_energy(
        slab_energy=12.0, bulk_energy=2.5, n_formula_units=4.0, area=10.0
    ) == pytest.approx(0.1)


def test_termination_energy_n_surfaces_divides_the_total():
    energy = dict(slab_energy=12.0, bulk_energy=2.5, n_formula_units=4.0, area=10.0)
    symmetric = termination_energy(n_surfaces=2, **energy)
    asymmetric = termination_energy(n_surfaces=1, **energy)
    assert asymmetric == pytest.approx(2.0 * symmetric)


def test_termination_energy_unit_conversion():
    value = termination_energy(
        slab_energy=12.0, bulk_energy=2.5, n_formula_units=4.0, area=10.0, unit="J_m2"
    )
    assert value == pytest.approx(0.1 * 16.02176634, rel=1e-12)


def test_termination_energy_rejects_bad_unit_and_area():
    with pytest.raises(ValueError, match="unsupported unit"):
        termination_energy(
            slab_energy=1.0, bulk_energy=1.0, n_formula_units=1.0, area=1.0, unit="J/m2"
        )
    for area in (0.0, -1.0):
        with pytest.raises(ValueError, match="area must be positive"):
            termination_energy(slab_energy=1.0, bulk_energy=1.0, n_formula_units=1.0, area=area)


@pytest.mark.parametrize("n_surfaces", [0, -1, 2.0, True])
def test_termination_energy_rejects_bad_surface_count(n_surfaces):
    with pytest.raises(ValueError, match="n_surfaces"):
        termination_energy(
            slab_energy=1.0, bulk_energy=1.0, n_formula_units=1.0, area=1.0, n_surfaces=n_surfaces
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("slab_energy", float("nan")),
        ("bulk_energy", float("inf")),
        ("n_formula_units", True),
        ("area", "10"),
    ],
)
def test_termination_energy_rejects_nonfinite_and_non_numeric(name, value):
    kwargs = dict(slab_energy=1.0, bulk_energy=1.0, n_formula_units=1.0, area=1.0)
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        termination_energy(**kwargs)


def test_non_stoichiometric_board_warns_but_still_references_the_bulk():
    with pytest.warns(UserWarning, match="non-stoichiometric"):
        value = termination_energy(slab_energy=5.0, bulk_energy=2.0, n_formula_units=2.5, area=10.0)
    assert value == pytest.approx(0.0)


def test_slab_formula_units_counts_stoichiometric_multiples():
    bulk = Atoms("NaCl", positions=[[0, 0, 0], [0, 0, 2]], cell=[4, 4, 8])
    assert slab_formula_units(bulk, bulk) == pytest.approx(1.0)
    doubled = bulk * (2, 1, 1)
    assert slab_formula_units(doubled, bulk) == pytest.approx(2.0)


def test_slab_formula_units_weights_non_stoichiometric_boards():
    bulk = Atoms("NaCl", positions=[[0, 0, 0], [0, 0, 2]], cell=[4, 4, 8])
    # Na3Cl2: (3/1 + 2/1) / 2 = 2.5 formula units.
    slab = Atoms("Na3Cl2", positions=[[0, 0, 0]] * 5, cell=[4, 4, 8])
    assert slab_formula_units(slab, bulk) == pytest.approx(2.5)


def test_slab_formula_units_rejects_foreign_elements_and_empty_bulk():
    bulk = Atoms("NaCl", positions=[[0, 0, 0], [0, 0, 2]], cell=[4, 4, 8])
    with pytest.raises(ValueError, match="absent from the bulk"):
        slab_formula_units(Atoms("NaClK"), bulk)
    with pytest.raises(ValueError, match="no atoms"):
        slab_formula_units(bulk, Atoms())


def test_rank_orders_by_energy_then_geometry_then_label():
    cheapest = _record("a", slab_energy=9.0, bulk_energy=2.0, n_formula_units=4.0)  # 0.05
    # Same energy, smaller stretch wins.
    low_stretch = _record(
        "b", stretch=0.001, slab_energy=12.0, bulk_energy=2.0, n_formula_units=5.0
    )
    high_stretch = _record(
        "c", stretch=0.02, slab_energy=12.0, bulk_energy=2.0, n_formula_units=5.0
    )
    # Same energy and strain, larger area wins (-area).
    large_area = _record("d", area=20.0, slab_energy=18.0, bulk_energy=2.0, n_formula_units=6.0)
    small_area = _record("e", area=10.0, slab_energy=15.0, bulk_energy=2.0, n_formula_units=6.0)
    ranked = rank_terminations([high_stretch, small_area, cheapest, large_area, low_stretch])
    assert [record.label for record in ranked] == ["a", "b", "c", "d", "e"]


def test_rank_keeps_energy_less_records_in_input_order_at_the_end():
    first = _record("first")
    second = _record("second")
    with_energy = _record("priced", slab_energy=99.0, bulk_energy=2.0, n_formula_units=4.0)
    ranked = rank_terminations([second, first, with_energy])
    assert [record.label for record in ranked] == ["priced", "second", "first"]


def test_rank_is_deterministic_for_identical_geometry():
    left = _record("x", slab_energy=10.0, bulk_energy=2.0, n_formula_units=4.0)
    right = _record("y", slab_energy=10.0, bulk_energy=2.0, n_formula_units=4.0)
    assert [r.label for r in rank_terminations([left, right])] == ["x", "y"]
    assert [r.label for r in rank_terminations([right, left])] == ["x", "y"]


def test_rank_propagates_invalid_area():
    broken = _record("broken", area=0.0, slab_energy=1.0, bulk_energy=1.0, n_formula_units=1.0)
    with pytest.raises(ValueError, match="area must be positive"):
        rank_terminations([broken])


def test_record_energy_is_none_until_all_inputs_are_supplied():
    record = _record("partial", slab_energy=10.0, bulk_energy=2.0)
    assert record.termination_energy_ev_a2 is None
    priced = _record("full", slab_energy=10.0, bulk_energy=2.0, n_formula_units=4.0)
    assert priced.termination_energy_ev_a2 == pytest.approx((10.0 - 8.0) / (2.0 * 10.0))


def test_unit_table_only_contains_supported_units():
    assert set(_EV_A2_TO_UNIT) == {"eV_A2", "J_m2"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # A stoichiometric board must not warn.
        assert termination_energy(
            slab_energy=12.0, bulk_energy=2.5, n_formula_units=4.0, area=10.0
        ) == pytest.approx(0.1)
