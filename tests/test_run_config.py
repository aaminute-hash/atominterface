"""Reproducible run configurations: round-trip and CLI replay.

``atominterface.run_config`` is the load/save half of reproducibility
(:mod:`atominterface.provenance` only records).  These tests pin the
round-trip contract and the ``atominterface build --config`` replay
semantics: a written ``run_config.yaml`` must reproduce the run, and an
explicit command-line flag must always beat the file.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from atominterface import cli_build
from atominterface.cli import app
from atominterface.run_config import load_run_config, save_run_config

runner = CliRunner()


def _combined_output(result) -> str:
    """Join stdout with stderr across Click generations (see test_cli)."""
    output = result.output or ""
    try:
        stderr = result.stderr or ""
    except ValueError:
        stderr = ""
    return output + stderr


@pytest.fixture
def fake_build(monkeypatch):
    """Exercise the real config assembly without lattice enumeration."""
    state = SimpleNamespace(
        candidates=[],
        report={
            "charge_policy": "report",
            "attempted": 0,
            "assembled": 0,
            "assembly_failed": 0,
            "analyzed": 0,
            "kept": 0,
            "rejected": 0,
            "rejection_reasons": {},
            "stopped_at_limit": False,
            "empty_stage": "no_matches",
            "thresholds": {"charge_tolerance_e": 1e-6, "dipole_tolerance_e_per_A": 1e-3},
        },
        config=None,
    )

    class FakeBuilder:
        def __init__(self, *, substrate, film, config):
            state.config = config
            self.last_build_report = state.report

        def build(self):
            return state.candidates

        def preview_matches(self, limit):
            return []

    def load(path):
        return SimpleNamespace(composition=SimpleNamespace(reduced_formula="NaCl"))

    monkeypatch.setattr(cli_build, "_resolve_builder", lambda name: FakeBuilder)
    monkeypatch.setattr(cli_build, "_load_structure", load)
    return state


def test_roundtrip_is_lossless_for_yaml_native_values(tmp_path):
    params = {
        "builder": "coherent",
        "sub_miller": [0, 0, 1],
        "max_strain": 0.05,
        "min_thickness_substrate": None,
        "symmetric": True,
        "substrate_oxidation": ["Na=1", "Cl=-1"],
        "nested": {"gap": 2.0, "labels": ["a", "b"]},
    }
    path = save_run_config(params, tmp_path / "run.yaml")
    assert load_run_config(path) == params
    # A second round-trip is a fixed point: no drift on re-save.
    assert load_run_config(save_run_config(load_run_config(path), tmp_path / "again")) == params


def test_save_normalizes_tuples_and_paths_without_losing_data(tmp_path):
    params = {
        "sub_miller": (0, 0, 1),
        "film_miller": (1, 1, 1),
        "outdir": Path("outputs/build"),
    }
    loaded = load_run_config(save_run_config(params, tmp_path / "run_config"))
    # Tuples come back as lists and Paths as strings; the values survive.
    assert loaded == {
        "sub_miller": [0, 0, 1],
        "film_miller": [1, 1, 1],
        "outdir": str(Path("outputs/build")),
    }


def test_save_forces_a_yaml_suffix(tmp_path):
    path = save_run_config({"a": 1}, tmp_path / "run.txt")
    assert path.name == "run.yaml"
    assert path.is_file()


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="run configuration not found"):
        load_run_config(tmp_path / "nope.yaml")


def test_load_rejects_a_non_mapping_document(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_run_config(path)


def test_load_empty_file_means_no_overrides(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_run_config(path) == {}


def test_save_rejects_non_mapping_params(tmp_path):
    with pytest.raises(TypeError, match="must be a mapping"):
        save_run_config([("a", 1)], tmp_path / "run.yaml")


def test_build_replays_a_config_file(fake_build, tmp_path):
    config_path = save_run_config(
        {
            "min_thickness": 9.0,
            "min_thickness_substrate": 12.0,
            "min_thickness_film": 4.0,
            "vacuum": 20.0,
            "gap": 3.0,
            "max_candidates": 5,
            "charge_policy": "require-neutral",
            "symmetric": True,
        },
        tmp_path / "run.yaml",
    )
    result = runner.invoke(
        app,
        [
            "build",
            "sub.cif",
            "film.cif",
            "--config",
            str(config_path),
            "--outdir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, _combined_output(result)
    config = fake_build.config
    assert config.min_thickness == 9.0
    assert config.min_thickness_substrate == 12.0
    assert config.min_thickness_film == 4.0
    assert config.vacuum_thickness == 20.0
    assert config.gap == 3.0
    assert config.max_candidates == 5
    assert config.charge_policy == "require-neutral"
    assert config.symmetric_slabs is True


def test_build_config_file_is_equivalent_to_command_line(fake_build, tmp_path):
    flags = [
        "--min-thickness-substrate",
        "11",
        "--min-thickness-film",
        "5",
        "--vacuum",
        "18",
        "--max-strain",
        "0.03",
    ]
    via_flags = runner.invoke(
        app, ["build", "sub.cif", "film.cif", *flags, "--outdir", str(tmp_path / "a")]
    )
    assert via_flags.exit_code == 0, _combined_output(via_flags)
    from_flags = fake_build.config

    config_path = save_run_config(
        {
            "min_thickness_substrate": 11.0,
            "min_thickness_film": 5.0,
            "vacuum": 18.0,
            "max_strain": 0.03,
        },
        tmp_path / "run.yaml",
    )
    via_file = runner.invoke(
        app,
        [
            "build",
            "sub.cif",
            "film.cif",
            "--config",
            str(config_path),
            "--outdir",
            str(tmp_path / "b"),
        ],
    )
    assert via_file.exit_code == 0, _combined_output(via_file)
    # The assembled config dataclass is identical either way.
    assert fake_build.config == from_flags


def test_explicit_flags_override_the_config_file(fake_build, tmp_path):
    config_path = save_run_config({"min_thickness": 9.0, "vacuum": 20.0}, tmp_path / "run.yaml")
    result = runner.invoke(
        app,
        [
            "build",
            "sub.cif",
            "film.cif",
            "--config",
            str(config_path),
            "--min-thickness",
            "7",
            "--outdir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, _combined_output(result)
    # The explicit flag wins; the untouched key still comes from the file.
    assert fake_build.config.min_thickness == 7.0
    assert fake_build.config.vacuum_thickness == 20.0


def test_build_writes_an_immediately_replayable_run_config(fake_build, tmp_path):
    outdir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "build",
            "sub.cif",
            "film.cif",
            "--min-thickness-substrate",
            "12",
            "--vacuum",
            "20",
            "--symmetric",
            "--outdir",
            str(outdir),
        ],
    )
    assert result.exit_code == 0, _combined_output(result)
    first_config = fake_build.config

    written = load_run_config(outdir / "run_config.yaml")
    assert written["min_thickness_substrate"] == 12.0
    assert written["min_thickness_film"] is None
    assert written["vacuum"] == 20.0
    assert written["symmetric"] is True

    # Replaying the recorded file must reproduce the same builder config.
    replay = runner.invoke(
        app,
        [
            "build",
            "sub.cif",
            "film.cif",
            "--config",
            str(outdir / "run_config.yaml"),
            "--outdir",
            str(tmp_path / "replay"),
        ],
    )
    assert replay.exit_code == 0, _combined_output(replay)
    assert fake_build.config == first_config


def test_build_rejects_unknown_config_keys(fake_build, tmp_path):
    config_path = save_run_config(
        {"min_thikness": 9.0},
        tmp_path / "run.yaml",  # typo must not be dropped silently
    )
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", "--config", str(config_path)])
    assert result.exit_code == 2
    assert "unknown run configuration key" in _combined_output(result)
    assert "min_thikness" in _combined_output(result)


def test_build_rejects_a_missing_config_file(fake_build):
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", "--config", "nope.yaml"])
    assert result.exit_code == 2
    assert "run configuration not found" in _combined_output(result)
