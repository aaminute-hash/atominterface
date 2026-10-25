"""Tests for the bulk-referenced interface formation energy."""

from __future__ import annotations

import pytest

from atominterface.thermo import (
    interface_formation_energy,
    termination_energy,
    work_of_separation,
)


def test_interface_formation_energy_matches_hand_calculation():
    # gamma_int = (E_int - E_A_slab - E_B_slab) / A + gamma_A + gamma_B
    #           = (-10.0 + 6.0 + 8.0) / 10.0 + 0.5 + 0.3 = 1.2 eV/A^2
    assert interface_formation_energy(
        interface_energy_total=-10.0,
        substrate_slab_energy=-6.0,
        film_slab_energy=-8.0,
        substrate_surface_energy=0.5,
        film_surface_energy=0.3,
        area=10.0,
    ) == pytest.approx(1.2)


def test_interface_formation_energy_unit_conversion():
    value = interface_formation_energy(
        interface_energy_total=-10.0,
        substrate_slab_energy=-6.0,
        film_slab_energy=-8.0,
        substrate_surface_energy=0.5,
        film_surface_energy=0.3,
        area=10.0,
        unit="J_m2",
    )
    assert value == pytest.approx(1.2 * 16.02176634, rel=1e-12)


@pytest.mark.parametrize(
    ("n_a", "bulk_a", "gamma_a", "n_b", "bulk_b", "gamma_b", "gamma_int"),
    [
        (4.0, -5.0, 0.5, 3.0, -4.0, 0.3, 0.25),
        (1.0, -10.0, 1.25, 2.0, -7.5, 0.05, -0.4),
        (6.0, -3.25, 0.0, 6.0, -3.25, 0.0, 0.75),
    ],
)
def test_formation_energy_is_self_consistent_for_any_bulk_reference(
    n_a, bulk_a, gamma_a, n_b, bulk_b, gamma_b, gamma_int
):
    """E_slab = n * E_bulk + 2 * A * gamma must reproduce gamma_int exactly."""
    area = 12.5
    slab_a = n_a * bulk_a + 2.0 * area * gamma_a
    slab_b = n_b * bulk_b + 2.0 * area * gamma_b
    interface = n_a * bulk_a + n_b * bulk_b + area * (gamma_a + gamma_b + gamma_int)
    assert interface_formation_energy(
        interface_energy_total=interface,
        substrate_slab_energy=slab_a,
        film_slab_energy=slab_b,
        substrate_surface_energy=gamma_a,
        film_surface_energy=gamma_b,
        area=area,
    ) == pytest.approx(gamma_int, abs=1e-12)


def test_roundtrip_with_termination_energy_reproduces_the_target():
    """termination_energy(n_surfaces=2) in, interface_formation_energy out."""
    area = 10.0
    n_a, bulk_a, gamma_a = 4.0, -5.0, 0.5
    n_b, bulk_b, gamma_b = 3.0, -4.0, 0.3
    gamma_int = 0.25

    slab_a = n_a * bulk_a + 2.0 * area * gamma_a
    slab_b = n_b * bulk_b + 2.0 * area * gamma_b
    interface = n_a * bulk_a + n_b * bulk_b + area * (gamma_a + gamma_b + gamma_int)

    recovered_a = termination_energy(
        slab_energy=slab_a, bulk_energy=bulk_a, n_formula_units=n_a, area=area, n_surfaces=2
    )
    recovered_b = termination_energy(
        slab_energy=slab_b, bulk_energy=bulk_b, n_formula_units=n_b, area=area, n_surfaces=2
    )
    assert recovered_a == pytest.approx(gamma_a, abs=1e-12)
    assert recovered_b == pytest.approx(gamma_b, abs=1e-12)

    assert interface_formation_energy(
        interface_energy_total=interface,
        substrate_slab_energy=slab_a,
        film_slab_energy=slab_b,
        substrate_surface_energy=recovered_a,
        film_surface_energy=recovered_b,
        area=area,
    ) == pytest.approx(gamma_int, abs=1e-12)


def test_formation_energy_equals_surface_energies_minus_work_of_separation():
    """gamma_int = gamma_A + gamma_B - W_ad, with W_ad from work_of_separation."""
    area = 10.0
    interface, slab_a, slab_b = -21.5, -10.0, -6.0
    gamma_a, gamma_b = 0.5, 0.3
    work = work_of_separation(
        interface_energy_total=interface, separated_energy_total=slab_a + slab_b
    )
    gamma_int = interface_formation_energy(
        interface_energy_total=interface,
        substrate_slab_energy=slab_a,
        film_slab_energy=slab_b,
        substrate_surface_energy=gamma_a,
        film_surface_energy=gamma_b,
        area=area,
    )
    assert gamma_int == pytest.approx(gamma_a + gamma_b - work / area, abs=1e-12)


def test_interface_formation_energy_rejects_bad_unit_and_area():
    kwargs = dict(
        interface_energy_total=-10.0,
        substrate_slab_energy=-6.0,
        film_slab_energy=-8.0,
        substrate_surface_energy=0.5,
        film_surface_energy=0.3,
        area=10.0,
    )
    with pytest.raises(ValueError, match="unsupported unit"):
        interface_formation_energy(unit="J/m2", **kwargs)
    for area in (0.0, -1.0):
        bad = dict(kwargs, area=area)
        with pytest.raises(ValueError, match="area must be positive"):
            interface_formation_energy(**bad)


@pytest.mark.parametrize(
    "name",
    [
        "interface_energy_total",
        "substrate_slab_energy",
        "film_slab_energy",
        "substrate_surface_energy",
        "film_surface_energy",
        "area",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "10"])
def test_interface_formation_energy_rejects_nonfinite_and_non_numeric(name, value):
    kwargs = dict(
        interface_energy_total=-10.0,
        substrate_slab_energy=-6.0,
        film_slab_energy=-8.0,
        substrate_surface_energy=0.5,
        film_surface_energy=0.3,
        area=10.0,
    )
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        interface_formation_energy(**kwargs)
