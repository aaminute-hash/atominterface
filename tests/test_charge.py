"""Charge-source, finite-slab geometry, and compatibility regression tests."""

from __future__ import annotations

import builtins
import copy
import json

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from atominterface.builders.charge import (
    ANALYSIS_CHARGE,
    CHARGE_KNOWN,
    CHARGE_SOURCE,
    SLAB_ID,
    SymmetricSlab,
    _normal_extent,
    analyze_charge_polarity,
    charge_rejection_reasons,
    check_neutral_slabs,
    is_potentially_polar,
    make_symmetric_slab,
    resolve_charge_data,
    suggest_dipole_correction,
    validate_charge_options,
)


@pytest.fixture
def slab():
    atoms = Atoms(
        "NaClNaCl",
        positions=[[0, 0, 2], [1, 0, 2], [0, 0, 6], [1, 0, 6]],
        cell=[2, 2, 10],
        pbc=[True, True, False],
    )
    atoms.set_array(SLAB_ID, np.array([0, 0, 1, 1]))
    return atoms


def _store(atoms, data):
    for name, field in (
        (ANALYSIS_CHARGE, "values"),
        (CHARGE_KNOWN, "known"),
        (CHARGE_SOURCE, "sources"),
    ):
        atoms.set_array(name, data[field])


def _analyze(atoms, **kwargs):
    options = {"oxidation_states": {"Na": 1, "Cl": -1}, "finite_slab": True}
    options.update(kwargs)
    return analyze_charge_polarity(atoms, **options)


def test_absent_charges_are_not_ase_default_zeros(slab):
    assert np.all(slab.get_initial_charges() == 0)
    result = analyze_charge_polarity(slab, finite_slab=True)
    for group in result["groups"].values():
        assert group["neutrality"] == group["polarity"] == "unknown"
        assert group["net_charge_e"] is None
        assert group["n_known"] == 0
        assert group["sources"] == ["unknown"]
    json.dumps(result, allow_nan=False)


def test_explicit_zeros_and_set_initial_zeros_are_known(slab):
    explicit = analyze_charge_polarity(slab, charges=[0, 0, 0, 0], finite_slab=True)
    slab.set_initial_charges(np.zeros(4))
    initial = analyze_charge_polarity(slab, finite_slab=True)
    for result, source in ((explicit, "explicit_charges"), (initial, "initial_charges")):
        for group in result["groups"].values():
            assert group["neutrality"] == "neutral"
            assert group["polarity"] == "nonpolar"
            assert group["net_charge_e"] == group["dipole_normal_eA"] == 0
            assert group["sources"] == [source]


def test_resolution_priority_records_overridden_sources():
    stored = resolve_charge_data(["Na", "Cl"], charges=[3, -3])
    result = resolve_charge_data(
        ["Na", "Cl"],
        charges=[0, 0],
        oxidation_states={"Na": 1, "Cl": -1},
        stored=stored,
        initial_charges=[2, -2],
        oxi_states=[4, -4],
    )
    np.testing.assert_array_equal(result["values"], [0, 0])
    assert result["sources"].tolist() == ["explicit_charges"] * 2
    assert result["overridden_sources"] == [
        "oxidation_mapping",
        "stored",
        "initial_charges",
        "oxidation_states",
    ]
    result = resolve_charge_data(["Na", "Cl"], oxidation_states={"Na": 1}, stored=stored)
    np.testing.assert_array_equal(result["known"], [True, False])
    np.testing.assert_array_equal(result["values"], [1, 0])
    assert result["sources"].tolist() == ["oxidation_mapping", "unknown"]
    assert result["overridden_sources"] == ["stored"]


def test_empty_mapping_is_explicit_unknown_not_missing():
    result = resolve_charge_data(["Na", "Cl"], oxidation_states={}, initial_charges=[1, -1])
    assert result["known"].tolist() == [False, False]
    assert result["values"].tolist() == [0.0, 0.0]
    assert result["overridden_sources"] == ["initial_charges"]


def test_stored_arrays_take_priority_and_are_independent_copies():
    original = resolve_charge_data(["Na", "Cl"], oxidation_states={"Na": 1})
    result = resolve_charge_data(["Na", "Cl"], stored=original, initial_charges=[0, 0])
    assert result["known"].tolist() == [True, False]
    assert result["sources"].tolist() == ["oxidation_mapping", "unknown"]
    assert result["overridden_sources"] == ["initial_charges"]
    result["values"][0] = 7
    result["known"][1] = True
    result["sources"][0] = "unknown"
    assert original["values"].tolist() == [1, 0]
    assert original["known"].tolist() == [True, False]
    assert original["sources"].tolist() == ["oxidation_mapping", "unknown"]


def test_site_oxidation_states_allow_none_without_filling():
    result = resolve_charge_data(["Fe", "Fe", "O"], oxi_states=[2, None, -2])
    assert result["values"].tolist() == [2, 0, -2]
    assert result["known"].tolist() == [True, False, True]
    assert result["sources"].tolist() == ["oxidation_states", "unknown", "oxidation_states"]


@pytest.mark.parametrize("argument", ["charges", "initial_charges"])
@pytest.mark.parametrize(
    "values",
    [
        [1],
        [[1, -1]],
        [None, 1],
        [True, -1],
        [1, np.nan],
        [1, np.inf],
        [1, -np.inf],
        [1, 2j],
        ["1", "-1"],
    ],
)
def test_explicit_charge_arrays_must_be_complete_finite_real(argument, values):
    with pytest.raises(ValueError):
        resolve_charge_data(["Na", "Cl"], **{argument: values})


@pytest.mark.parametrize(
    "mapping",
    [
        {"NotAnElement": 1},
        {"X": 0},
        {"na": 1},
        {"Na": True},
        {"Na": None},
        {"Na": np.nan},
        {"Na": np.inf},
        {"Na": 1j},
        {"Na": "1"},
        [1, -1],
    ],
)
def test_mapping_validation(mapping):
    with pytest.raises(ValueError):
        resolve_charge_data(["Na", "Cl"], oxidation_states=mapping)


@pytest.mark.parametrize(
    "stored",
    [
        {},
        {"values": [1, -1], "known": [True, False]},
        {"values": [1, -1], "known": [1, 0], "sources": ["explicit_charges"] * 2},
        {"values": [1, -1], "known": [True, True], "sources": ["guess"] * 2},
        {"values": [1, np.nan], "known": [True, False], "sources": ["explicit_charges", "unknown"]},
    ],
)
def test_stored_validation(stored):
    with pytest.raises(ValueError):
        resolve_charge_data(["Na", "Cl"], stored=stored)


@pytest.mark.parametrize("known,source", [(True, "unknown"), (False, "explicit_charges")])
def test_stored_known_mask_and_source_must_agree(known, source):
    stored = {"values": [0], "known": [known], "sources": [source]}
    with pytest.raises(ValueError, match="known mask and sources"):
        resolve_charge_data(["Na"], stored=stored)


def test_invalid_element_rejected():
    with pytest.raises(ValueError, match="element"):
        resolve_charge_data(["Fake"], charges=[0])


def test_ignored_lower_priority_sources_do_not_change_valid_selection():
    result = resolve_charge_data(["Na"], charges=[0], oxidation_states={"Na": "bad"})
    assert result["known"].tolist() == [True]
    assert result["overridden_sources"] == ["oxidation_mapping"]


def test_analysis_stored_data_and_explicit_override(slab):
    _store(
        slab, resolve_charge_data(slab.get_chemical_symbols(), oxidation_states={"Na": 1, "Cl": -1})
    )
    slab.set_initial_charges(np.full(4, 5.0))
    stored = analyze_charge_polarity(slab, finite_slab=True)
    assert stored["groups"]["total"]["sources"] == ["oxidation_mapping"]
    assert stored["overridden_sources"] == ["initial_charges"]
    explicit = analyze_charge_polarity(slab, charges=[0] * 4, finite_slab=True)
    assert explicit["groups"]["total"]["sources"] == ["explicit_charges"]
    assert explicit["overridden_sources"] == ["stored", "initial_charges"]
    mapping = analyze_charge_polarity(slab, oxidation_states={})
    assert mapping["groups"]["total"]["neutrality"] == "unknown"


def test_incomplete_stored_arrays_are_not_silently_ignored(slab):
    slab.set_array(ANALYSIS_CHARGE, np.zeros(4))
    with pytest.raises(ValueError, match="stored"):
        analyze_charge_polarity(slab)
    assert (
        analyze_charge_polarity(slab, charges=[0] * 4)["groups"]["total"]["neutrality"] == "neutral"
    )


def test_partial_mapping_unknown_never_sums_known_subset(slab):
    result = _analyze(slab, oxidation_states={"Na": 1})
    total = result["groups"]["total"]
    assert total["n_known"] == 2
    assert total["n_atoms"] == 4
    assert total["net_charge_e"] is None
    assert total["dipole_normal_eA"] is None
    assert total["neutrality"] == "unknown"
    assert charge_rejection_reasons(result, "report") == []
    assert "substrate: neutrality unknown" in charge_rejection_reasons(result, "require-neutral")


def test_mixed_valence_is_preserved_per_site():
    atoms = Atoms("Fe3O4", positions=np.zeros((7, 3)), cell=[2, 2, 10], pbc=[True, True, False])
    atoms.set_array("oxi_states", np.array([2, 3, 3, -2, -2, -2, -2]))
    result = analyze_charge_polarity(atoms, finite_slab=True)
    assert result["groups"]["total"]["net_charge_e"] == 0
    assert result["groups"]["total"]["sources"] == ["oxidation_states"]
    explicit = analyze_charge_polarity(atoms, charges=[3, 2, 3, -2, -2, -2, -2], finite_slab=True)
    assert explicit["groups"]["total"]["neutrality"] == "neutral"
    assert explicit["overridden_sources"] == ["oxidation_states"]


def test_mixed_sources_are_explicitly_identified(slab):
    data = resolve_charge_data(slab.get_chemical_symbols(), charges=[0] * 4)
    data["sources"] = np.array(["explicit_charges"] * 2 + ["oxidation_mapping"] * 2)
    _store(slab, data)
    result = analyze_charge_polarity(slab, finite_slab=True)
    assert "mixed_charge_sources" in result["groups"]["total"]["reasons"]
    assert result["groups"]["substrate"]["sources"] == ["explicit_charges"]
    assert result["groups"]["film"]["sources"] == ["oxidation_mapping"]


def test_charged_sides_cannot_hide_behind_neutral_total(slab):
    result = _analyze(slab, charges=[1, 1, -1, -1])
    assert result["groups"]["total"]["neutrality"] == "neutral"
    assert result["groups"]["substrate"]["net_charge_e"] == 2
    assert result["groups"]["film"]["net_charge_e"] == -2
    assert result["groups"]["substrate"]["dipole_normal_eA"] is None
    assert charge_rejection_reasons(result, "require-neutral") == [
        "substrate: neutrality charged",
        "film: neutrality charged",
    ]


def test_opposite_side_dipoles_cannot_hide_behind_nonpolar_total(slab):
    slab.positions[:, 2] = [2, 3, 6, 5]
    result = _analyze(slab)
    assert result["groups"]["total"]["polarity"] == "nonpolar"
    assert result["groups"]["substrate"]["dipole_normal_eA"] == -1
    assert result["groups"]["film"]["dipole_normal_eA"] == 1
    assert charge_rejection_reasons(result, "require-neutral") == []
    assert charge_rejection_reasons(result, "require-nonpolar") == [
        "substrate: polarity polar",
        "film: polarity polar",
    ]


def test_charge_tolerance_boundary(slab):
    boundary = _analyze(slab, charges=[1e-6, 0, 0, 0])
    above = _analyze(slab, charges=[np.nextafter(1e-6, np.inf), 0, 0, 0])
    assert boundary["groups"]["total"]["neutrality"] == "neutral"
    assert above["groups"]["total"]["neutrality"] == "charged"
    assert above["groups"]["total"]["dipole_normal_eA"] is None


def test_dipole_density_tolerance_boundary(slab):
    slab.positions[:, 2] = [2, 3, 6, 6]
    at_boundary = _analyze(slab, dipole_tolerance=0.25)
    below = _analyze(slab, dipole_tolerance=np.nextafter(0.25, 0))
    assert at_boundary["groups"]["total"]["dipole_density_e_per_A"] == -0.25
    assert at_boundary["groups"]["total"]["polarity"] == "nonpolar"
    assert below["groups"]["total"]["polarity"] == "polar"


@pytest.mark.parametrize("option", ["charge_tolerance", "dipole_tolerance"])
@pytest.mark.parametrize("value", [-1, np.nan, np.inf, -np.inf, True, None, "1e-3"])
def test_invalid_tolerances_raise(slab, option, value):
    with pytest.raises(ValueError):
        _analyze(slab, **{option: value})


def test_policy_validation(slab):
    assert validate_charge_options("report", 0, 0) is None
    with pytest.raises(ValueError, match="policy"):
        validate_charge_options("neutral", 1e-6, 1e-3)
    with pytest.raises(ValueError, match="policy"):
        charge_rejection_reasons(_analyze(slab), "neutral")


@pytest.mark.parametrize(
    "tags",
    [
        [0],
        [[0, 0, 1, 1]],
        [0, 0, 1, 2],
        [False, False, True, True],
        [0.0, 0.0, 1.0, 1.0],
        ["0", "0", "1", "1"],
    ],
)
def test_invalid_tags_raise(slab, tags):
    with pytest.raises(ValueError, match="slab_tags"):
        _analyze(slab, slab_tags=tags)


def test_missing_tags_keep_total_but_sides_unknown(slab):
    del slab.arrays[SLAB_ID]
    result = _analyze(slab)
    assert result["groups"]["total"]["polarity"] == "nonpolar"
    for name in ("substrate", "film"):
        assert result["groups"][name]["neutrality"] == "unknown"
        assert result["groups"][name]["reasons"] == ["slab_tags_missing"]
    assert "film: neutrality unknown" in charge_rejection_reasons(result, "require-neutral")


def test_explicit_tags_override_stored_and_empty_groups_remain_unknown(slab):
    result = _analyze(slab, slab_tags=[0] * 4)
    assert result["groups"]["substrate"]["n_atoms"] == 4
    assert result["groups"]["film"]["neutrality"] == "unknown"
    assert result["groups"]["film"]["reasons"] == ["empty_group"]


def test_geometry_must_be_declared_finite(slab):
    result = analyze_charge_polarity(slab, charges=[0] * 4)
    assert result["groups"]["total"]["neutrality"] == "neutral"
    assert result["groups"]["total"]["polarity"] == "unknown"
    assert "finite_slab_not_declared" in result["groups"]["total"]["reasons"]


def test_geometry_context_and_explicit_defaults_override_snapshot(slab):
    slab.info["charge_geometry"] = {"normal_axis": 2, "finite_slab": True, "vacuum_reference": None}
    slab.info["charge_analysis"] = {"groups": {"total": {"polarity": "polar"}}}
    result = analyze_charge_polarity(slab, charges=[0] * 4)
    assert result["groups"]["total"]["polarity"] == "nonpolar"
    assert result["snapshot"] == "input_geometry"
    assert (
        analyze_charge_polarity(slab, charges=[0] * 4, finite_slab=False)["groups"]["total"][
            "polarity"
        ]
        == "unknown"
    )
    slab.info["charge_geometry"]["normal_axis"] = 0
    assert (
        analyze_charge_polarity(slab, charges=[0] * 4, normal_axis=2)["geometry"]["normal_axis"]
        == 2
    )
    slab.info.pop("charge_geometry")
    assert (
        analyze_charge_polarity(slab, charges=[0] * 4)["groups"]["total"]["polarity"] == "unknown"
    )


def test_rotations_and_translations_preserve_area_and_dipoles(slab):
    slab.positions[:, 2] = [2, 3, 6, 6]
    original = _analyze(slab)
    slab.rotate(61, [1, 2, 3], rotate_cell=True)
    slab.translate([100, -23, 4.2])
    transformed = _analyze(slab)
    assert transformed["geometry"]["area_A2"] == pytest.approx(4)
    for name in ("total", "substrate", "film"):
        before, after = original["groups"][name], transformed["groups"][name]
        assert after["dipole_normal_eA"] == pytest.approx(before["dipole_normal_eA"], abs=1e-12)
        assert after["polarity"] == before["polarity"]


def test_tolerance_residual_uses_translation_invariant_origin(slab):
    slab.positions[:, 2] = [2, 3, 6, 6]
    charges = [1, -1 + 1e-7, 0, 0]
    before = _analyze(slab, charges=charges)
    slab.translate([2, 3, 1e5])
    after = _analyze(slab, charges=charges)
    assert before["groups"]["total"]["neutrality"] == "neutral"
    for name in ("total", "substrate", "film"):
        assert after["groups"][name]["dipole_normal_eA"] == pytest.approx(
            before["groups"][name]["dipole_normal_eA"], abs=1e-10
        )


def test_skew_cell_normal_is_cross_product_not_cartesian_axis(slab):
    slab.set_cell([[2, 0, 1], [0, 2, 1], [0, 0, 10]])
    slab.positions[:] = [[0, 0, 2], [2, 0, 3], [0, 0, 6], [0, 2, 7]]
    result = _analyze(slab)
    assert result["groups"]["total"]["dipole_normal_eA"] == pytest.approx(0, abs=1e-12)
    assert result["geometry"]["area_A2"] == pytest.approx(np.sqrt(24))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_all_normal_axis_choices(axis):
    cell = np.zeros((3, 3))
    cell[axis, 2] = 10
    cell[(axis + 1) % 3, 0] = 2
    cell[(axis + 2) % 3, 1] = 3
    atoms = Atoms("NaCl", positions=[[0, 0, 2], [0, 0, 3]], cell=cell, pbc=False)
    result = _analyze(atoms, normal_axis=axis)
    assert result["geometry"]["area_A2"] == 6
    assert result["groups"]["total"]["dipole_normal_eA"] == -1


@pytest.mark.parametrize("axis", [-1, 3, 2.0, True])
def test_invalid_normal_axis(slab, axis):
    with pytest.raises(ValueError, match="normal_axis"):
        _analyze(slab, normal_axis=axis)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_invalid_coordinates_raise(slab, invalid):
    slab.positions[0, 2] = invalid
    with pytest.raises(ValueError, match="positions"):
        _analyze(slab)


def test_invalid_cell_area_raises(slab):
    slab.cell[1] = slab.cell[0]
    with pytest.raises(ValueError, match="area"):
        _analyze(slab)


def test_empty_system_does_not_pass_as_neutral():
    result = analyze_charge_polarity(Atoms(cell=[2, 2, 10]), charges=[], finite_slab=True)
    assert result["groups"]["total"]["neutrality"] == "unknown"
    assert result["groups"]["total"]["reasons"] == ["empty_group"]


def test_periodic_normal_requires_explicit_empty_vacuum(slab):
    slab.pbc = True
    missing = _analyze(slab)
    assert missing["groups"]["total"]["polarity"] == "unknown"
    assert "vacuum_reference_missing" in missing["geometry"]["reasons"]
    valid = _analyze(slab, vacuum_reference=(0.9, 1.1))
    assert valid["groups"]["total"]["polarity"] == "nonpolar"
    crossing = _analyze(slab, vacuum_reference=(0.9, 0.1))
    assert crossing["groups"] == valid["groups"]
    occupied = _analyze(slab, vacuum_reference=(0.1, 0.3))
    assert occupied["groups"]["total"]["polarity"] == "unknown"
    assert "vacuum_reference_not_empty" in occupied["geometry"]["reasons"]


@pytest.mark.parametrize(
    "reference",
    [(0, 0), (0, 1), (0, 1.2), (np.nan, 0.1), (0.9, np.inf), (True, 0.1), [0.2], "vacuum"],
)
def test_invalid_vacuum_reference_raises(slab, reference):
    slab.pbc = True
    with pytest.raises(ValueError, match="vacuum_reference"):
        _analyze(slab, vacuum_reference=reference)


def test_vacuum_open_interval_can_end_at_surface_atoms(slab):
    slab.pbc = True
    result = _analyze(slab, vacuum_reference=(0.6, 1.2))
    assert result["geometry"]["dipole_geometry_valid"] is True
    assert result["groups"]["total"]["polarity"] == "nonpolar"
    slab.positions[0, 2] = 8
    occupied = _analyze(slab, vacuum_reference=(0.6, 1.2))
    assert occupied["groups"]["total"]["polarity"] == "unknown"
    assert "vacuum_reference_not_empty" in occupied["geometry"]["reasons"]


def test_explicit_none_clears_stored_vacuum_reference(slab):
    slab.pbc = True
    slab.info["charge_geometry"] = {"finite_slab": True, "vacuum_reference": [0.9, 1.1]}
    assert (
        analyze_charge_polarity(slab, charges=[0] * 4)["groups"]["total"]["polarity"] == "nonpolar"
    )
    result = analyze_charge_polarity(slab, charges=[0] * 4, vacuum_reference=None)
    assert result["groups"]["total"]["polarity"] == "unknown"


def test_periodic_wrap_translation_and_rotation_use_common_cut(slab):
    slab.pbc = True
    slab.positions[:, 2] = [9, 1, 1, 9]
    before = _analyze(slab, vacuum_reference=(0.3, 0.7))
    assert before["groups"]["substrate"]["dipole_normal_eA"] == pytest.approx(-2)
    assert before["groups"]["film"]["dipole_normal_eA"] == pytest.approx(2)
    assert before["groups"]["total"]["dipole_normal_eA"] == pytest.approx(0)
    slab.positions += np.array([[0, 0, -20], [0, 0, 30], [2, 2, -10], [0, 0, 10]])
    wrapped = _analyze(slab, vacuum_reference=(0.3, 0.7))
    slab.translate([1, 2, 5])
    slab.wrap()
    slab.rotate(29, [1, -1, 2], rotate_cell=True)
    shifted = _analyze(slab, vacuum_reference=(0.8, 1.2))
    for result in (wrapped, shifted):
        for name in ("total", "substrate", "film"):
            assert result["groups"][name]["dipole_normal_eA"] == pytest.approx(
                before["groups"][name]["dipole_normal_eA"], abs=1e-12
            )
            assert result["groups"][name]["polarity"] == before["groups"][name]["polarity"]


def test_vacuum_size_does_not_normalize_dipole_by_volume(slab):
    slab.pbc = True
    slab.positions[:, 2] = [2, 3, 6, 6]
    before = _analyze(slab, vacuum_reference=(0.9, 1.1))
    slab.cell[2, 2] = 20
    after = _analyze(slab, vacuum_reference=(0.6, 0.8))
    assert after["groups"]["total"]["dipole_density_e_per_A"] == pytest.approx(
        before["groups"]["total"]["dipole_density_e_per_A"]
    )


def test_nonperiodic_coordinates_are_not_wrapped(slab):
    slab.positions[:, 2] = [9, 11, 9, 11]
    result = _analyze(slab)
    assert result["groups"]["total"]["dipole_normal_eA"] == -4


def test_analysis_never_mutates_arrays_info_or_calculator(slab):
    slab.set_initial_charges([1, -1, 1, -1])
    slab.info["charge_geometry"] = {"finite_slab": True}
    slab.info["charge_analysis"] = {"obsolete": [1, 2, 3]}
    _store(slab, resolve_charge_data(slab.get_chemical_symbols(), charges=[0] * 4))
    slab.calc = SinglePointCalculator(
        slab, energy=3.0, forces=np.zeros((4, 3)), charges=np.full(4, 9.0)
    )
    arrays = {key: value.copy() for key, value in slab.arrays.items()}
    info = copy.deepcopy(slab.info)
    calculator = slab.calc
    results = copy.deepcopy(calculator.results)
    analyze_charge_polarity(slab, charges=[1, -1, 1, -1])
    check_neutral_slabs(slab, [0, 0, 1, 1])
    suggest_dipole_correction(slab)
    assert slab.calc is calculator
    assert slab.info == info
    assert slab.arrays.keys() == arrays.keys()
    for key, value in arrays.items():
        np.testing.assert_array_equal(slab.arrays[key], value)
    for key, value in results.items():
        np.testing.assert_array_equal(calculator.results[key], value)


@pytest.mark.parametrize("symbol", ["Fe", "Ni", "Cu"])
def test_legacy_hint_has_no_metal_substring_false_positives(symbol):
    hint, reason = is_potentially_polar(Atoms(symbol))
    assert hint is False
    # An anion-free composition carries no polar-termination warning at
    # all; the empty reason is not a nonpolarity determination.
    assert reason == ""


def test_legacy_hint_is_only_composition_based():
    hint, reason = is_potentially_polar(Atoms("NaCl"))
    assert hint is True
    assert "unknown" in reason
    assert is_potentially_polar(object())[0] is False
    assert "unknown" in is_potentially_polar(object())[1]


def test_legacy_neutrality_checker_reports_unknown_and_charged(slab):
    unknown = check_neutral_slabs(slab, [0, 0, 1, 1])
    assert "substrate: neutrality unknown" in unknown
    slab.set_initial_charges([1, 1, -1, -1])
    assert check_neutral_slabs(slab, [0, 0, 1, 1]) == [
        "substrate: neutrality charged",
        "film: neutrality charged",
    ]
    slab.set_initial_charges([1, -1, 1, -1])
    assert check_neutral_slabs(slab, [0, 0, 1, 1]) == []


def test_legacy_dipole_suggestion_is_three_state(slab):
    assert suggest_dipole_correction(slab)["dipole_correction_recommended"] is None
    slab.set_initial_charges([1, -1, 1, -1])
    slab.info["charge_geometry"] = {"finite_slab": True}
    assert suggest_dipole_correction(slab)["dipole_correction_recommended"] is False
    slab.positions[1, 2] += 1
    assert suggest_dipole_correction(slab)["dipole_correction_recommended"] is True
    slab.set_initial_charges([1, 1, 1, 1])
    recommendation = suggest_dipole_correction(slab)
    assert recommendation["dipole_correction_recommended"] is None
    assert "nonneutral_group" in recommendation["reason"]


def test_basic_analysis_never_imports_pymatgen(slab, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("pymatgen"):
            raise ImportError("pymatgen is intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert _analyze(slab)["groups"]["total"]["polarity"] == "nonpolar"
    assert is_potentially_polar(slab)[0] is True


def test_pymatgen_site_oxidation_adapter():
    core = pytest.importorskip("pymatgen.core")
    structure = core.Structure(core.Lattice.cubic(5), ["Na+", "Cl-"], [[0, 0, 0.3], [0, 0, 0.4]])
    result = analyze_charge_polarity(structure, finite_slab=True, vacuum_reference=(0.7, 0.9))
    assert result["groups"]["total"]["neutrality"] == "neutral"
    assert result["groups"]["total"]["dipole_normal_eA"] == pytest.approx(-0.5)
    assert result["groups"]["total"]["sources"] == ["oxidation_states"]
    assert is_potentially_polar(structure)[0] is True


def test_pymatgen_site_charge_adapter_preserves_priority_and_input():
    core = pytest.importorskip("pymatgen.core")
    structure = core.Structure(
        core.Lattice.cubic(5),
        ["Na+", "Cl-"],
        [[0, 0, 0.3], [0, 0, 0.4]],
        site_properties={"charge": [2, -2]},
    )
    before = copy.deepcopy(structure.as_dict())
    result = analyze_charge_polarity(structure, finite_slab=True, vacuum_reference=(0.7, 0.9))
    assert result["groups"]["total"]["dipole_normal_eA"] == pytest.approx(-1)
    assert result["groups"]["total"]["sources"] == ["initial_charges"]
    assert result["overridden_sources"] == ["oxidation_states"]
    assert structure.as_dict() == before
    mapping = analyze_charge_polarity(structure, oxidation_states={"Na": 0, "Cl": 0})
    assert mapping["groups"]["total"]["sources"] == ["oxidation_mapping"]
    assert mapping["overridden_sources"] == ["initial_charges", "oxidation_states"]
    structure.add_site_property("initial_charges", [0, 0])
    initial = analyze_charge_polarity(structure, finite_slab=True, vacuum_reference=(0.7, 0.9))
    assert initial["groups"]["total"]["dipole_normal_eA"] == 0
    assert initial["groups"]["total"]["sources"] == ["initial_charges"]


@pytest.mark.parametrize("site_charges", [[2, None], [2, np.nan], [True, -1]])
def test_pymatgen_site_charge_adapter_rejects_incomplete_or_invalid_data(site_charges):
    core = pytest.importorskip("pymatgen.core")
    structure = core.Structure(
        core.Lattice.cubic(5),
        ["Na+", "Cl-"],
        [[0, 0, 0.3], [0, 0, 0.4]],
        site_properties={"charge": site_charges},
    )
    with pytest.raises(ValueError, match="initial_charges"):
        analyze_charge_polarity(structure)


def test_tiny_nacl_001_and_111_have_different_formal_charge_polarity():
    core = pytest.importorskip("pymatgen.core")
    surface = pytest.importorskip("pymatgen.core.surface")
    structure = core.Structure.from_spacegroup(
        "Fm-3m", core.Lattice.cubic(5.64), ["Na+", "Cl-"], [[0, 0, 0], [0.5, 0, 0]]
    )
    polarities = []
    for miller in ((0, 0, 1), (1, 1, 1)):
        generated = surface.SlabGenerator(
            structure, miller, min_slab_size=6, min_vacuum_size=12, center_slab=True
        ).get_slab()
        result = analyze_charge_polarity(generated, finite_slab=True, vacuum_reference=(0.95, 1.05))
        assert result["groups"]["total"]["neutrality"] == "neutral"
        assert result["groups"]["total"]["n_known"] == len(generated)
        polarities.append(result["groups"]["total"]["polarity"])
    assert polarities == ["nonpolar", "polar"]


# ---------------------------------------------------------------------------
# Symmetric slab construction (Tasker Type 3 mitigation).
# ---------------------------------------------------------------------------


def _rocksalt():
    """Rocksalt with formal charges on every site, as a pymatgen Structure."""
    core = pytest.importorskip("pymatgen.core")
    return core.Structure.from_spacegroup(
        "Fm-3m", core.Lattice.cubic(5.64), ["Na+", "Cl-"], [[0, 0, 0], [0.5, 0, 0]]
    )


def test_symmetric_slab_is_a_palindrome_with_equal_terminations():
    bulk = _rocksalt()
    result = make_symmetric_slab(bulk, (1, 1, 1), min_thickness=8.0, vacuum=15.0)
    structure = result.structure

    matrix = np.asarray(structure.lattice.matrix, dtype=float)
    normal = np.cross(matrix[0], matrix[1])
    normal = normal / np.linalg.norm(normal)
    projected = np.asarray(structure.cart_coords, dtype=float) @ normal
    levels = np.unique(np.round(projected, 4))

    # The board is at least as thick as requested, and the (111) bilayer
    # spacing of a = 5.64 A rocksalt is a / sqrt(3) / 2 = 1.628 A.
    assert result.area > 0
    assert _normal_extent(structure) >= 8.0 - 1e-9
    assert len(levels) % 2 == 1  # odd plane count: the top plane is shared
    assert np.allclose(np.diff(levels), 5.64 / np.sqrt(3) / 2, rtol=1e-3)

    # The stacking sequence must read the same forwards and backwards,
    # with identical terminations on both surfaces.
    symbols = [site.specie.symbol for site in structure.sites]
    stacked = [
        "+".join(sorted(symbols[i] for i in np.flatnonzero(np.abs(projected - level) < 1e-3)))
        for level in levels
    ]
    assert stacked == stacked[::-1]
    assert stacked[0] == stacked[-1]
    assert isinstance(result, SymmetricSlab)


def test_symmetric_slab_reports_nonstoichiometric_board_without_fixing_it():
    bulk = _rocksalt()
    result = make_symmetric_slab(bulk, (1, 1, 1), min_thickness=8.0, vacuum=15.0)

    # Rocksalt (111) is Tasker Type 3: the symmetric board is necessarily
    # non-stoichiometric, and that is reported, never compensated.
    assert result.formula == "Na3Cl4"
    assert result.nominal_charge == pytest.approx(-1.0)
    assert result.tasker_type == 3
    # Formula units against the 8-site conventional reference cell
    # (4 Na + 4 Cl): (3/4 + 4/4) / 2 = 0.875.
    assert result.n_formula_units == pytest.approx(0.875)
    assert any("nominal charge" in note for note in result.warnings)
    assert result.structure.properties["ai_slab_generated"] is True


def test_symmetric_slab_does_not_mutate_the_input():
    bulk = _rocksalt()
    before = copy.deepcopy(bulk.as_dict())
    make_symmetric_slab(bulk, (1, 1, 1), min_thickness=8.0, vacuum=15.0)
    assert bulk.as_dict() == before


def test_symmetric_slab_accepts_ase_input():
    pytest.importorskip("pymatgen.core")
    copper = Atoms(
        "Cu4",
        positions=[[0, 0, 0], [0, 1.8075, 1.8075], [1.8075, 0, 1.8075], [1.8075, 1.8075, 0]],
        cell=[3.615, 3.615, 3.615],
        pbc=True,
    )
    result = make_symmetric_slab(copper, (0, 0, 1), min_thickness=8.0, vacuum=15.0)
    assert result.structure.composition.reduced_formula == "Cu"
    # No charge model was supplied: unknown is not zero.
    assert result.nominal_charge is None
    assert result.tasker_type is None
    assert any("tasker type unknown" in note for note in result.warnings)


def test_symmetric_slab_termination_selection():
    bulk = _rocksalt()
    explicit = make_symmetric_slab(bulk, (1, 1, 1), termination=0, min_thickness=8.0)
    assert explicit.structure.composition.reduced_formula == "Na3Cl4"
    with pytest.raises(ValueError, match="out of range"):
        make_symmetric_slab(bulk, (1, 1, 1), termination=3, min_thickness=8.0)


def test_symmetric_slab_keeps_charge_distinct_terminations_apart():
    """A mixed-valence structure must not have its terminations merged.

    ``primitive=True`` would collapse the two Fe(111) cuts through
    species-only symmetry; the charge-aware enumeration keeps them apart
    and the two boards carry opposite nominal charges.
    """
    core = pytest.importorskip("pymatgen.core")
    bulk = core.Structure(core.Lattice.cubic(4.0), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    bulk.add_site_property(ANALYSIS_CHARGE, [1.0, -1.0])
    bulk.add_site_property(CHARGE_KNOWN, [True, True])
    bulk.add_site_property(CHARGE_SOURCE, ["explicit_charges"] * 2)

    first = make_symmetric_slab(bulk, (1, 1, 1), termination=0, min_thickness=8.0)
    second = make_symmetric_slab(bulk, (1, 1, 1), termination=1, min_thickness=8.0)
    assert first.nominal_charge == pytest.approx(-1.0)
    assert second.nominal_charge == pytest.approx(1.0)
    with pytest.raises(ValueError, match="out of range"):
        make_symmetric_slab(bulk, (1, 1, 1), termination=2, min_thickness=8.0)


def test_symmetric_slab_max_layers_truncates_the_board():
    bulk = _rocksalt()
    result = make_symmetric_slab(bulk, (1, 1, 1), min_thickness=20.0, max_layers=3)
    # Three planes mirrored about the top plane: 2n - 1 = 5 atoms.
    assert len(result.structure) == 5
    assert result.structure.composition.as_dict() == {"Na+": 2.0, "Cl-": 3.0}


def test_symmetric_slab_unknown_charges_stay_unknown():
    core = pytest.importorskip("pymatgen.core")
    bulk = core.Structure(core.Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0, 0]])
    result = make_symmetric_slab(bulk, (1, 1, 1), min_thickness=8.0)
    assert result.nominal_charge is None
    assert result.tasker_type is None
    assert any("tasker type unknown" in note for note in result.warnings)


@pytest.mark.parametrize(
    "options",
    [
        {"min_thickness": 0},
        {"min_thickness": -1.0},
        {"vacuum": -1.0},
        {"max_layers": 0},
        {"max_layers": True},
    ],
)
def test_symmetric_slab_rejects_invalid_options(options):
    bulk = _rocksalt()
    with pytest.raises(ValueError):
        make_symmetric_slab(bulk, (1, 1, 1), **options)
