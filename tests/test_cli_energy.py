"""CLI energy-path tests: --relax wiring, backend/device checks, example 03.

Everything here runs on the zero-dependency dummy backend (or on no
backend at all), so the suite stays fast and green on a slim
installation; the real MACE/CHGNet paths are covered by
``tests/test_backends_real.py``.  The Cu/Ni example data doubles as the
fixture, mirroring the style of ``tests/test_cli.py`` (CliRunner) and
``tests/test_examples.py`` (subprocess for the example script).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from atominterface.cli import app

runner = CliRunner()

ROOT = Path(__file__).parents[1]
DATA = ROOT / "examples" / "data"
EXAMPLE03 = ROOT / "examples" / "03_interface_energy.py"
SUBSTRATE = DATA / "Cu_fcc.vasp"
FILM = DATA / "Ni_fcc.vasp"


def _combined_output(result) -> str:
    """Join stdout with stderr across Click generations.

    Click >= 8.2 captures stderr separately, so ``result.output`` only
    holds stdout.  Click < 8.2 mixes both into ``result.output`` and
    makes ``result.stderr`` *raise* ``ValueError("stderr not separately
    captured")`` -- an attribute that exists but explodes, which is why a
    plain ``getattr`` default does not help.
    """
    output = result.output or ""
    try:
        stderr = result.stderr or ""
    except ValueError:
        stderr = ""
    return output + stderr


def _read_metadata(path: Path) -> dict:
    """Parse the provenance file, tolerating the YAML/JSON fallback.

    ``write_metadata`` degrades to JSON when PyYAML cannot represent the
    payload, and the file keeps its ``.yaml`` name either way, so the
    parser is chosen by content, not by suffix.
    """
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]

        return dict(yaml.safe_load(text))
    except Exception:  # noqa: BLE001
        return dict(json.loads(text))


@pytest.mark.skipif(not SUBSTRATE.exists(), reason="example data missing")
def test_build_relax_dummy_writes_ood_columns(tmp_path):
    """``build --relax`` on the dummy backend runs and appends OOD columns."""
    pytest.importorskip("pymatgen")
    outdir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "build",
            str(SUBSTRATE),
            str(FILM),
            "--relax",
            "--backend",
            "dummy",
            "--outdir",
            str(outdir),
        ],
    )
    assert result.exit_code == 0, _combined_output(result)

    with (outdir / "candidates.csv").open(newline="") as handle:
        table = csv.DictReader(handle)
        assert "ood_score" in table.fieldnames
        assert "ood_step" in table.fieldnames
        rows = list(table)
    assert rows, "expected at least one candidate"
    # A relaxed candidate reports its peak OOD score and a step index
    # (-1 when the threshold was never crossed).
    assert rows[0]["ood_score"] != ""
    assert rows[0]["ood_step"].lstrip("-").isdigit()
    assert (outdir / "candidate_01.extxyz").is_file()

    payload = json.loads((outdir / "candidates.json").read_text(encoding="utf-8"))
    assert payload[0]["relax"]["backend"] == "dummy"
    assert payload[0]["relax"]["ood_step"] == -1
    # The relaxation must actually run: before the dummy backend accepted
    # ASE's force_consistent kwarg, the optimizer constructor raised
    # TypeError and the guard swallowed it as relax_exception (a silent
    # no-op relax behind a green exit code).
    assert not any(
        reason.startswith("relax_exception") for reason in payload[0]["relax"]["reasons"]
    )


@pytest.mark.skipif(not SUBSTRATE.exists(), reason="example data missing")
def test_build_without_relax_does_not_instantiate_calculator(tmp_path, monkeypatch):
    """Geometry-only runs stay light: no calculator is ever constructed."""

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("calculator must not be built without --relax")

    monkeypatch.setattr("atominterface.backends.get_calculator", _boom)
    outdir = tmp_path / "out"
    result = runner.invoke(app, ["build", str(SUBSTRATE), str(FILM), "--outdir", str(outdir)])
    assert result.exit_code == 0, _combined_output(result)
    assert (outdir / "candidates.csv").is_file()


def test_build_rejects_unknown_backend():
    """An unknown backend fails fast and lists what is available."""
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", "--backend", "nope"])
    output = _combined_output(result)
    assert result.exit_code != 0
    assert "unknown backend" in output
    assert "dummy" in output


def test_build_rejects_unknown_device():
    """Only cpu/cuda are accepted; the message names both."""
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", "--device", "tpu"])
    output = _combined_output(result)
    assert result.exit_code != 0
    assert "must be one of" in output
    assert "cpu" in output
    assert "cuda" in output


@pytest.mark.skipif(not EXAMPLE03.exists(), reason="example missing")
def test_example_03_dummy_end_to_end(tmp_path):
    """Example 03 runs the whole energy ledger on the dummy backend."""
    pytest.importorskip("pymatgen")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    outdir = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE03),
            "--backend",
            "dummy",
            "--outdir",
            str(outdir),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined

    expected_columns = {
        "rank",
        "interface_id",
        "substrate",
        "film",
        "sub_termination",
        "film_termination",
        "area_A2",
        "n_atoms",
        "E_bulk_sub_ev_per_atom",
        "E_bulk_film_ev_per_atom",
        "E_slab_sub_ev",
        "E_slab_film_ev",
        "E_interface_ev",
        "gamma_surf_sub_eV_A2",
        "gamma_surf_film_eV_A2",
        "gamma_eV_per_A2",
        "gamma_J_per_m2",
        "ood_score",
        "ood_step",
        "ood_triggered",
        "backend",
        "model_hash",
    }
    with (outdir / "candidates.csv").open(newline="") as handle:
        table = csv.DictReader(handle)
        assert expected_columns <= set(table.fieldnames)
        rows = list(table)
    assert len(rows) == 1
    assert rows[0]["backend"] == "dummy"
    assert rows[0]["substrate"] == "Cu"
    assert rows[0]["film"] == "Ni"

    metadata = _read_metadata(outdir / "metadata.yaml")
    assert metadata["backend"] == "dummy"
    assert metadata["seed"] == 0
    assert metadata["model_hash"]  # identity string, not a bare hash
    assert "candidates.csv" in metadata["artifacts"]
    assert len(metadata["artifacts"]["candidates.csv"]) == 64  # full sha256 hex

    for name in (
        "interface_relaxed.extxyz",
        "slab_sub_relaxed.extxyz",
        "slab_film_relaxed.extxyz",
        "bulk_sub.extxyz",
        "bulk_film.extxyz",
    ):
        assert (outdir / name).is_file(), name
