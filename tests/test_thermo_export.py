"""Tests for thermodynamics and FEM export."""

import numpy as np
import pytest

from atominterface.export.czm_fem import export_czm_moose
from atominterface.thermo import (
    fit_cohesive_zone,
    interface_energy,
    work_of_separation,
)


def test_interface_energy_zero_for_matching_bulk():
    # E_interface == E_sub + E_film -> gamma = 0
    gamma = interface_energy(
        interface_energy_total=10.0,
        substrate_energy_per_area=4.0,
        film_energy_per_area=6.0,
        area=1.0,
    )
    assert abs(gamma - 0.0) < 1e-12


def test_interface_energy_positive_for_higher_interface():
    gamma = interface_energy(
        interface_energy_total=12.0,
        substrate_energy_per_area=4.0,
        film_energy_per_area=6.0,
        area=1.0,
    )
    assert gamma == 2.0


def test_interface_energy_rejects_nonpositive_area():
    with pytest.raises(ValueError, match="positive"):
        interface_energy(
            interface_energy_total=1.0,
            substrate_energy_per_area=0.0,
            film_energy_per_area=0.0,
            area=0.0,
        )


def test_interface_energy_unit_check():
    with pytest.raises(ValueError, match="unsupported unit"):
        interface_energy(
            interface_energy_total=1.0,
            substrate_energy_per_area=0.0,
            film_energy_per_area=0.0,
            area=1.0,
            unit="J_m2",
        )


def test_work_of_separation():
    w = work_of_separation(interface_energy_total=10.0, separated_energy_total=12.5)
    assert w == 2.5


def test_fit_cohesive_zone_bilinear():
    disp = np.linspace(0, 2.0, 200)
    # bilinear: rises to 1.0 at delta=1, then drops to 0 at delta=2
    trac = np.where(disp <= 1.0, disp, 2.0 - disp)
    params = fit_cohesive_zone(disp, trac)
    assert abs(params["max_traction"] - 1.0) < 1e-3
    assert abs(params["critical_separation"] - 1.0) < 1e-2
    # area under triangle = 0.5 * base * height = 0.5 * 2 * 1 = 1.0
    assert abs(params["fracture_energy"] - 1.0) < 1e-2


def test_fit_cohesive_zone_requires_sorted_inputs():
    disp = np.array([1.0, 0.0, 2.0])
    trac = np.array([0.5, 0.0, 1.0])
    # should not raise; function sorts internally
    params = fit_cohesive_zone(disp, trac)
    assert params["max_traction"] == 1.0


def test_export_czm_moose_writes_file(tmp_path):
    params = {"max_traction": 1.2e9, "critical_separation": 2.5e-9, "fracture_energy": 1.5}
    p = export_czm_moose(params, interface_id="li_llzo", out_path=tmp_path / "czm.i")
    assert p.exists()
    text = p.read_text()
    assert "CohesiveZoneMasterMaterial" in text
    assert "1.200000e+09" in text
