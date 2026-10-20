"""Charge snapshots agree across candidate tables, CLI reports and extxyz."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from atominterface.builders.base import Candidate
from atominterface.builders.candidates import charge_summary, summarize, to_csv, to_markdown
from atominterface.cli_build import _print_candidates, _write_outputs

LEGACY_SUMMARY_FIELDS = [
    "interface_id",
    "substrate",
    "film",
    "sub_miller",
    "film_miller",
    "sub_termination",
    "film_termination",
    "strain_sub",
    "strain_film",
    "mean_strain",
    "area_A2",
    "n_substrate",
    "n_film",
    "n_total",
]
LEGACY_CLI_FIELDS = [
    "rank",
    "interface_id",
    "builder",
    "substrate",
    "film",
    "substrate_termination",
    "film_termination",
    "strain_substrate",
    "strain_film",
    "area",
    "n_atoms",
]


@pytest.fixture
def candidate():
    group = {
        "n_atoms": 2,
        "n_known": 2,
        "net_charge_e": 0.0,
        "neutrality": "neutral",
        "dipole_normal_eA": 0.0,
        "dipole_density_e_per_A": 0.0,
        "polarity": "nonpolar",
        "sources": ["explicit_charges"],
        "reasons": [],
    }
    analysis = {
        "schema_version": 1,
        "groups": {
            "total": {**group, "n_atoms": 4, "n_known": 4},
            "substrate": dict(group),
            "film": dict(group),
        },
        "thresholds": {"charge_tolerance_e": 1e-6, "dipole_tolerance_e_per_A": 1e-3},
        "geometry": {
            "finite_slab": True,
            "normal_axis": 2,
            "area_A2": 16.0,
            "vacuum_reference": [0.8, 0.9],
        },
        "snapshot": "input_geometry",
    }
    atoms = Atoms(
        "NaClNaCl",
        positions=[[0, 0, 2], [1, 1, 2], [0, 0, 5], [1, 1, 5]],
        cell=[4, 4, 12],
        pbc=True,
    )
    atoms.new_array("ai_charge", np.array([1.0, -1.0, 1.0, -1.0]))
    atoms.new_array("ai_charge_known", np.array([True, True, True, True]))
    atoms.new_array("ai_charge_source", np.array(["explicit_charges"] * 4))
    atoms.new_array("ai_slab_id", np.array([0, 0, 1, 1]))
    return Candidate(
        atoms=atoms,
        interface_id="neutral-1",
        builder_name="coherent",
        substrate_formula="NaCl",
        film_formula="NaCl",
        substrate_miller=(0, 0, 1),
        film_miller=(0, 0, 1),
        substrate_termination="0",
        film_termination="1",
        strain_substrate=0.012345,
        strain_film=0.034567,
        area=16.0,
        n_substrate=2,
        n_film=2,
        metadata={
            "charge_analysis": analysis,
            "charge_geometry": analysis["geometry"],
            "private_builder_state": "not exported to atoms.info",
        },
    )


@pytest.mark.parametrize("snapshot", ["input_geometry", "built_geometry"])
def test_summary_keeps_legacy_columns_and_appends_shared_charge_fields(candidate, snapshot):
    candidate.metadata["charge_analysis"]["snapshot"] = snapshot
    row = summarize([candidate])[0]
    assert list(row)[: len(LEGACY_SUMMARY_FIELDS)] == LEGACY_SUMMARY_FIELDS
    assert row["strain_sub"] == 0.0123
    assert row["n_total"] == 4
    expected = charge_summary(candidate.metadata["charge_analysis"])
    assert {key: row[key] for key in expected} == expected
    assert row["total_net_charge_e"] == 0
    assert row["film_dipole_density_e_per_A"] == 0
    assert row["substrate_sources"] == "explicit_charges"
    assert row["charge_tolerance_e"] == 1e-6
    assert row["charge_snapshot"] == snapshot


def test_legacy_candidate_is_not_analyzed_in_all_reports(candidate, tmp_path, capsys):
    candidate.metadata.clear()
    summary = charge_summary(None)
    assert list(summary) == list(charge_summary({}))
    assert summary["total_neutrality"] == "not_analyzed"
    assert summary["film_polarity"] == "not_analyzed"
    assert summary["total_net_charge_e"] is None
    assert summarize([candidate])[0]["substrate_neutrality"] == "not_analyzed"
    assert "not_analyzed" in to_markdown([candidate])
    _print_candidates([candidate])
    assert "not_analyzed/not_analyzed" in capsys.readouterr().out
    csv_path = _write_outputs([candidate], tmp_path, "extxyz")
    with csv_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["total_polarity"] == "not_analyzed"
    assert row["total_net_charge_e"] == ""
    assert json.loads((tmp_path / "candidates.json").read_text())[0]["charge_analysis"] is None


def test_unknown_charge_preserves_none_and_complete_json(candidate, tmp_path):
    analysis = candidate.metadata["charge_analysis"]
    analysis["groups"]["film"].update(
        n_known=1,
        net_charge_e=None,
        neutrality="unknown",
        dipole_normal_eA=None,
        dipole_density_e_per_A=None,
        polarity="unknown",
        reasons=["unknown_charge"],
    )
    summary = charge_summary(analysis)
    assert summary["film_neutrality"] == "unknown"
    assert summary["film_polarity"] == "unknown"
    assert summary["film_net_charge_e"] is None
    assert summary["film_dipole_density_e_per_A"] is None
    assert summary["film_reasons"] == "unknown_charge"
    _write_outputs([candidate], tmp_path, "extxyz")
    content = (tmp_path / "candidates.json").read_text()
    assert "NaN" not in content and "Infinity" not in content
    document = json.loads(content)
    assert document[0]["charge_analysis"] == analysis
    assert document[0]["film_net_charge_e"] is None


def test_csv_and_cli_csv_share_charge_columns_without_changing_legacy_fields(candidate, tmp_path):
    standalone_path = tmp_path / "standalone.csv"
    to_csv([candidate], str(standalone_path))
    cli_path = _write_outputs([candidate], tmp_path / "cli", "extxyz")
    with standalone_path.open(newline="") as handle:
        standalone = next(csv.DictReader(handle))
    with cli_path.open(newline="") as handle:
        cli = next(csv.DictReader(handle))
    assert list(standalone)[: len(LEGACY_SUMMARY_FIELDS)] == LEGACY_SUMMARY_FIELDS
    assert list(cli)[: len(LEGACY_CLI_FIELDS)] == LEGACY_CLI_FIELDS
    for name, value in charge_summary(candidate.metadata["charge_analysis"]).items():
        expected = "" if value is None else str(value)
        assert standalone[name] == cli[name] == expected
    assert cli["strain_substrate"] == "0.012345"
    assert cli["area"] == "16.0000"


def test_markdown_escapes_charge_reason_newlines_and_pipes(candidate):
    candidate.metadata["charge_analysis"]["groups"]["total"]["reasons"] = [
        "first|second\r\nthird\rfourth\nfifth",
        "another reason",
    ]
    markdown = to_markdown([candidate])
    assert len(markdown.splitlines()) == 3
    assert "first\\|second<br>third<br>fourth<br>fifth; another reason" in markdown
    assert "total_neutrality" in markdown
    assert "neutral" in markdown


def test_empty_reports_have_stable_csv_headers_and_json(tmp_path):
    report = {
        "charge_policy": "require-nonpolar",
        "attempted": 1,
        "assembled": 1,
        "assembly_failed": 0,
        "analyzed": 1,
        "kept": 0,
        "rejected": 1,
        "rejection_reasons": {"film:polar": 1},
        "stopped_at_limit": False,
        "empty_stage": "charge_filter",
        "thresholds": {"charge_tolerance_e": 1e-6, "dipole_tolerance_e_per_A": 1e-3},
    }
    csv_path = _write_outputs([], tmp_path / "cli", "extxyz", build_report=report)
    to_csv([], str(tmp_path / "standalone.csv"))
    for path in (csv_path, tmp_path / "standalone.csv"):
        with path.open(newline="") as handle:
            table = csv.DictReader(handle)
            assert "total_neutrality" in table.fieldnames
            assert "film_polarity" in table.fieldnames
            assert list(table) == []
    assert json.loads((csv_path.parent / "candidates.json").read_text()) == []
    assert json.loads((csv_path.parent / "build_report.json").read_text()) == report
    assert to_markdown([]) == "*No candidates passed the filters.*\n"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_json_rejects_nonfinite_charge_values(candidate, tmp_path, value):
    candidate.metadata["charge_analysis"]["groups"]["total"]["net_charge_e"] = value
    with pytest.raises(ValueError, match="JSON compliant"):
        _write_outputs([candidate], tmp_path / "invalid", "extxyz")
    assert not (tmp_path / "invalid").exists()


def test_json_rejects_nonfinite_build_report(tmp_path):
    with pytest.raises(ValueError, match="JSON compliant"):
        _write_outputs([], tmp_path / "invalid", "extxyz", {"attempted": float("inf")})
    assert not (tmp_path / "invalid").exists()


def test_extxyz_roundtrip_preserves_charge_namespace_and_arrays(candidate, tmp_path):
    _write_outputs([candidate], tmp_path, "extxyz")
    restored = read(tmp_path / "candidate_01.extxyz")
    assert restored.info["charge_analysis"] == candidate.metadata["charge_analysis"]
    assert restored.info["charge_geometry"] == candidate.metadata["charge_geometry"]
    assert "private_builder_state" not in restored.info
    for name in ("ai_charge", "ai_charge_known", "ai_charge_source", "ai_slab_id"):
        np.testing.assert_array_equal(restored.arrays[name], candidate.atoms.arrays[name])
    assert "initial_charges" not in restored.arrays
