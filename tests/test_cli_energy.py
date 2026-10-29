"""CLI energy-path tests: --relax wiring, backend/device checks, example 03.

Everything here runs on the zero-dependency dummy backend (or on no
backend at all), so the suite stays fast and green on a slim
installation; the real MACE/CHGNet paths are covered by
``tests/test_backends_real.py``.  The Cu/Ni example data doubles as the
fixture, mirroring the style of ``tests/test_cli.py`` (CliRunner) and
``tests/test_examples.py`` (subprocess for the example script).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
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


@pytest.fixture
def example03():
    """Import the numbered example as a module for failure-path injection."""
    spec = importlib.util.spec_from_file_location("atominterface_example03_test", EXAMPLE03)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec_module (the stdlib importlib recipe): the
    # module-level dataclasses in the example resolve their annotations
    # through sys.modules, and without this they raise AttributeError.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _csv_row(outdir: Path) -> dict[str, str]:
    with (outdir / "candidates.csv").open(newline="", encoding="utf-8") as handle:
        return dict(next(csv.DictReader(handle)))


def _csv_rows(outdir: Path) -> list[dict[str, str]]:
    with (outdir / "candidates.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_example_main(example03, outdir: Path, *extra: str) -> tuple[int, dict[str, str]]:
    code = example03.main(["--backend", "dummy", "--fmax", "0.05", "--outdir", str(outdir), *extra])
    return code, _csv_row(outdir)


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
        assert "valid" in table.fieldnames
        assert "committee_enabled" in table.fieldnames
        assert "ood_monitoring" in table.fieldnames
        rows = list(table)
    assert rows, "expected at least one candidate"
    assert rows[0]["valid"] == "True"
    assert rows[0]["committee_enabled"] == "False"
    assert rows[0]["ood_monitoring"] == "heuristics_only"
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
def test_build_relax_invalid_candidate_exits_nonzero_after_writing_diagnostics(tmp_path):
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
            "--max-candidates",
            "1",
            "--max-steps",
            "1",
            "--fmax",
            "0.000001",
            "--outdir",
            str(outdir),
        ],
    )

    row = _csv_row(outdir)
    assert result.exit_code != 0
    assert row["valid"] == "False"
    assert row["converged"] == "False"
    assert (
        "step_budget_exhausted"
        in json.loads((outdir / "candidates.json").read_text(encoding="utf-8"))[0]["relax"][
            "reasons"
        ][0]
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
            "--fmax",
            "0.05",
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
        "valid",
        "converged",
        "energies_finite",
        "invalid_reasons",
        "committee_enabled",
        "ood_monitoring",
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
    assert rows[0]["valid"] == "True"
    assert rows[0]["gamma_eV_per_A2"] != ""
    assert rows[0]["committee_enabled"] == "False"
    assert rows[0]["ood_monitoring"] == "heuristics_only"

    metadata = _read_metadata(outdir / "metadata.yaml")
    assert metadata["backend"] == "dummy"
    assert metadata["seed"] == 0
    assert metadata["model_hash"]  # identity string, not a bare hash
    assert "candidates.csv" in metadata["artifacts"]
    assert len(metadata["artifacts"]["candidates.csv"]) == 64  # full sha256 hex

    # Relaxed structures are rank-prefixed so a multi-candidate ledger
    # cannot overwrite itself; the bulk single points are shared.
    for name in (
        "01_interface_relaxed.extxyz",
        "01_slab_sub_relaxed.extxyz",
        "01_slab_film_relaxed.extxyz",
        "bulk_sub.extxyz",
        "bulk_film.extxyz",
    ):
        assert (outdir / name).is_file(), name


@pytest.mark.skipif(not EXAMPLE03.exists(), reason="example missing")
def test_example_03_dummy_nonconvergence_fails_closed(tmp_path):
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
            "--max-steps",
            "1",
            "--fmax",
            "0.000001",
            "--outdir",
            str(outdir),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )

    row = _csv_row(outdir)
    assert proc.returncode != 0
    assert row["valid"] == "False"
    assert row["gamma_eV_per_A2"] == ""
    assert row["gamma_J_per_m2"] == ""
    assert "step_budget_exhausted" in row["invalid_reasons"]


def test_example_03_reference_slab_failure_invalidates_gamma(example03, tmp_path, monkeypatch):
    original = example03._guarded_relax
    calls = 0

    def fail_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 1:
            result.valid = False
            result.converged = False
            result.exception = "RuntimeError: reference slab failed"
            result.reasons = ["reference slab failed"]
        return result

    monkeypatch.setattr(example03, "_guarded_relax", fail_first)
    code, row = _run_example_main(example03, tmp_path / "out")

    assert code != 0
    assert row["slab_sub_valid"] == "False"
    assert row["slab_film_valid"] == "True"
    assert row["interface_valid"] == "True"
    assert row["gamma_eV_per_A2"] == ""
    assert "reference slab failed" in row["invalid_reasons"]


def test_example_03_interface_failure_invalidates_gamma(example03, tmp_path, monkeypatch):
    original = example03._guarded_relax
    calls = 0

    def fail_third(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 3:
            result.valid = False
            result.converged = False
            result.exception = "RuntimeError: interface failed"
            result.reasons = ["interface failed"]
        return result

    monkeypatch.setattr(example03, "_guarded_relax", fail_third)
    code, row = _run_example_main(example03, tmp_path / "out")

    assert code != 0
    assert row["interface_valid"] == "False"
    assert row["gamma_eV_per_A2"] == ""
    assert "interface failed" in row["invalid_reasons"]


def test_example_03_ood_trigger_invalidates_gamma(example03, tmp_path, monkeypatch):
    original = example03._guarded_relax
    calls = 0

    def trigger_ood_on_interface(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 3:
            result.valid = False
            result.converged = False
            result.ood_triggered = True
            result.ood_step = 0
            result.max_ood_score = 0.95
            result.reasons = ["committee_disagreement"]
            result.metadata["committee_enabled"] = True
            result.metadata["ood_monitoring"] = "committee+heuristics"
        return result

    monkeypatch.setattr(example03, "_guarded_relax", trigger_ood_on_interface)
    code, row = _run_example_main(example03, tmp_path / "out")

    assert code != 0
    assert row["valid"] == "False"
    assert row["ood_triggered"] == "True"
    assert row["committee_enabled"] == "True"
    assert "committee+heuristics" in row["ood_monitoring"]
    assert row["gamma_eV_per_A2"] == ""


def test_example_03_calculator_exception_invalidates_gamma(example03, tmp_path, monkeypatch):
    def failing_factory(atoms):
        raise RuntimeError("weights unavailable")

    monkeypatch.setattr(
        example03, "_make_calculator", lambda backend, device, dtype: failing_factory
    )
    code, row = _run_example_main(example03, tmp_path / "out")

    assert code != 0
    assert row["valid"] == "False"
    assert row["gamma_eV_per_A2"] == ""
    assert "weights unavailable" in row["invalid_reasons"]


def test_example_03_relaxation_construction_failure_invalidates_gamma(example03, tmp_path):
    """A construction-time calculator failure must not crash the driver.

    ``GuardedRelaxer._wrap`` raises instead of falling back to a zero-force
    dummy; the ledger has to record it as an invalid candidate rather than
    aborting the whole run with a traceback.
    """
    from atominterface.relax import GuardedRelaxer

    def broken_wrap(self, atoms):
        raise RuntimeError("checkpoint unreadable")

    original = GuardedRelaxer._wrap
    GuardedRelaxer._wrap = broken_wrap
    try:
        code, row = _run_example_main(example03, tmp_path / "out")
    finally:
        GuardedRelaxer._wrap = original

    assert code != 0
    assert row["valid"] == "False"
    assert row["interface_valid"] == "False"
    assert row["gamma_eV_per_A2"] == ""
    assert "checkpoint unreadable" in row["invalid_reasons"]


@pytest.mark.parametrize("nonfinite", ["energy", "forces"])
def test_example_03_nonfinite_calculator_invalidates_gamma(
    example03, tmp_path, monkeypatch, nonfinite
):
    class NonFiniteCalculator:
        def __init__(self, atoms):
            self.atoms = atoms

        def get_potential_energy(self, atoms, force_consistent=False):
            return float("nan") if nonfinite == "energy" else 0.001 * len(atoms)

        def get_forces(self, atoms):
            import numpy as np

            forces = np.zeros((len(atoms), 3))
            if nonfinite == "forces":
                forces[0, 0] = np.inf
            return forces

        @property
        def results(self):
            return {
                "energy": self.get_potential_energy(self.atoms),
                "forces": self.get_forces(self.atoms),
            }

    monkeypatch.setattr(
        example03,
        "_make_calculator",
        lambda backend, device, dtype: NonFiniteCalculator,
    )
    code, row = _run_example_main(example03, tmp_path / "out")

    assert code != 0
    assert row["valid"] == "False"
    assert row["energies_finite"] == "False"
    assert row["gamma_eV_per_A2"] == ""
    assert "not finite" in row["invalid_reasons"] or "non_finite" in row["invalid_reasons"]


# ------------------------------------------------------------------
# multi-candidate ledger: every candidate gets a row, judged on its own
# ------------------------------------------------------------------
def _run_multi(example03, outdir: Path, *extra: str) -> int:
    # Annotated local: the example03 fixture is dynamically imported, so
    # ``example03.main`` is ``Any`` to the type checker; the annotation
    # keeps the declared ``-> int`` from leaking that ``Any`` outward.
    code: int = example03.main(
        [
            "--backend",
            "dummy",
            "--fmax",
            "0.05",
            "--max-candidates",
            "3",
            "--outdir",
            str(outdir),
            *extra,
        ]
    )
    return code


def test_example_03_ledgers_every_candidate(example03, tmp_path):
    """--max-candidates N yields N ledger rows, not just the first one."""
    pytest.importorskip("pymatgen")
    outdir = tmp_path / "out"
    code = _run_multi(example03, outdir)

    rows = _csv_rows(outdir)
    assert code == 0
    assert len(rows) == 3
    assert [row["rank"] for row in rows] == ["1", "2", "3"]
    assert len({row["interface_id"] for row in rows}) == 3
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["valid"] == "True" for row in rows)
    assert all(row["gamma_eV_per_A2"] != "" for row in rows)


def test_example_03_ledger_row_carries_the_full_diagnostics(example03, tmp_path):
    """The row covers id, energies, area, reference, convergence, OOD, gamma."""
    pytest.importorskip("pymatgen")
    outdir = tmp_path / "out"
    _run_multi(example03, outdir)
    row = _csv_rows(outdir)[0]

    for column in (
        "rank",
        "interface_id",
        "E_slab_sub_ev",
        "E_slab_film_ev",
        "E_interface_ev",
        "area_A2",
        "area_source",
        "area_sub_A2",
        "area_film_A2",
        "area_consistent",
        "n_layers_sub",
        "n_layers_film",
        "vacuum_A",
        "min_thickness_A",
        "reference_state",
        "converged",
        "status",
        "error_reason",
        "ood_score",
        "gamma_eV_per_A2",
        "gamma_J_per_m2",
    ):
        assert column in row, column
    assert row["reference_state"] == "strained-coherent"
    assert int(row["n_layers_sub"]) > 0 and int(row["n_layers_film"]) > 0


def test_example_03_middle_candidate_failure_keeps_the_others(example03, tmp_path, monkeypatch):
    """A candidate that fails mid-list is recorded, not silently dropped.

    ``_relax_ledger`` relaxes substrate, film and interface in that order,
    so candidate ``n`` owns calls ``3n-2 .. 3n``; failing 4..6 kills the
    middle candidate only.
    """
    pytest.importorskip("pymatgen")
    original = example03._guarded_relax
    calls = 0

    def fail_second_candidate(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if 4 <= calls <= 6:
            result.valid = False
            result.converged = False
            result.exception = "RuntimeError: injected failure"
            result.reasons = ["injected failure"]
        return result

    monkeypatch.setattr(example03, "_guarded_relax", fail_second_candidate)
    outdir = tmp_path / "out"
    code = _run_multi(example03, outdir)

    rows = _csv_rows(outdir)
    assert code == 2
    assert len(rows) == 3
    assert [row["status"] for row in rows] == ["ok", "failed", "ok"]

    failed = rows[1]
    assert failed["valid"] == "False"
    assert failed["gamma_eV_per_A2"] == ""
    assert failed["gamma_J_per_m2"] == ""
    assert failed["gamma_surf_sub_eV_A2"] == ""
    assert failed["gamma_surf_film_eV_A2"] == ""
    assert failed["error_reason"] != ""
    assert "injected failure" in failed["error_reason"]
    # The successful candidates still ship their numbers.
    assert rows[0]["gamma_eV_per_A2"] != ""
    assert rows[2]["gamma_eV_per_A2"] != ""
    assert rows[0]["gamma_J_per_m2"] != ""
    assert rows[2]["gamma_J_per_m2"] != ""


def test_example_03_all_candidates_ok_exits_zero(example03, tmp_path):
    """Clean multi-candidate ledger: exit 0 (the failure case is above)."""
    pytest.importorskip("pymatgen")
    outdir = tmp_path / "out"
    code = _run_multi(example03, outdir)
    rows = _csv_rows(outdir)

    assert code == 0
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["error_reason"] == "" for row in rows)


def test_example_03_multi_candidate_structures_do_not_overwrite(example03, tmp_path):
    """Rank-prefixed extxyz names: three candidates, three distinct files."""
    pytest.importorskip("pymatgen")
    outdir = tmp_path / "out"
    assert _run_multi(example03, outdir) == 0

    for rank in (1, 2, 3):
        for name in ("interface_relaxed", "slab_sub_relaxed", "slab_film_relaxed"):
            assert (outdir / f"{rank:02d}_{name}.extxyz").is_file(), f"{rank:02d}_{name}"
    assert (outdir / "bulk_sub.extxyz").is_file()
    assert (outdir / "bulk_film.extxyz").is_file()

    payloads = [
        (outdir / f"{rank:02d}_interface_relaxed.extxyz").read_bytes() for rank in (1, 2, 3)
    ]
    assert len(set(payloads)) == 3


def test_example_03_single_candidate_default_is_unchanged(example03, tmp_path):
    """--max-candidates stays 1 by default: one row, rank 1, exit 0."""
    pytest.importorskip("pymatgen")
    outdir = tmp_path / "out"
    code = example03.main(["--backend", "dummy", "--fmax", "0.05", "--outdir", str(outdir)])
    rows = _csv_rows(outdir)

    assert code == 0
    assert len(rows) == 1
    assert rows[0]["rank"] == "1"
    assert rows[0]["status"] == "ok"
    assert rows[0]["gamma_eV_per_A2"] != ""


# --- exposed knobs: --dtype and the per-side thickness overrides ---------
#
# Both were available to the builder/registry layers before this demo
# exposed them; the tests below pin the forwarding and, more importantly,
# that leaving them at their defaults reproduces the old behaviour bit
# for bit.


class _Capture:
    """Calculator factory stand-in that records kwargs and still works.

    The captured factory has to return a *usable* ASE calculator: the
    caller asserts on ``main()``'s exit code too, and a factory that
    returns None makes every candidate fail for reasons unrelated to the
    forwarding being tested.  The dummy backend is the real delegate.
    """

    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.kwargs: dict[str, object] = {}

    def __call__(self, *args, **kwargs):
        self.kwargs = dict(kwargs)
        return self._delegate(*args, **kwargs)


def _capture_calculator(monkeypatch, example03) -> _Capture:
    from atominterface import backends

    real = backends.get_calculator
    capture = _Capture(real)
    monkeypatch.setattr(backends, "get_calculator", capture)
    return capture


def test_example_03_dtype_defaults_to_float32(example03, tmp_path, monkeypatch):
    """The default dtype is float32, matching mace-torch's own default."""
    capture = _capture_calculator(monkeypatch, example03)
    code = example03.main(
        ["--backend", "dummy", "--fmax", "0.05", "--outdir", str(tmp_path / "out")]
    )

    assert code == 0
    assert capture.kwargs["dtype"] == "float32"
    assert capture.kwargs["device"] == "cpu"
    assert capture.kwargs["seed"] == example03.SEED


def test_example_03_dtype_float64_is_forwarded(example03, tmp_path, monkeypatch):
    """--dtype float64 reaches the backend factory verbatim."""
    capture = _capture_calculator(monkeypatch, example03)
    code = example03.main(
        [
            "--backend",
            "dummy",
            "--fmax",
            "0.05",
            "--dtype",
            "float64",
            "--outdir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 0
    assert capture.kwargs["dtype"] == "float64"


def test_example_03_dtype_is_recorded_in_the_provenance(example03, tmp_path):
    """cli_args in metadata.yaml carries the dtype so a run is reproducible."""
    pytest.importorskip("pymatgen")
    outdir = tmp_path / "out"
    example03.main(
        ["--backend", "dummy", "--fmax", "0.05", "--dtype", "float64", "--outdir", str(outdir)]
    )
    meta = _read_metadata(outdir / "metadata.yaml")

    assert meta["cli_args"]["dtype"] == "float64"


def _fake_args(**overrides) -> argparse.Namespace:
    """Namespace shaped like _parse_args' output, for _build_candidates."""
    values = {
        "backend": "dummy",
        "device": "cpu",
        "dtype": "float32",
        "min_thickness": 14.4,
        "min_thickness_substrate": None,
        "min_thickness_film": None,
        "vacuum": 15.0,
        "max_candidates": 3,
        "fmax": 0.01,
        "max_steps": 200,
        "outdir": "outputs/energy_cu_ni",
        "sub_miller": [0, 0, 1],
        "film_miller": [0, 0, 1],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_example_03_per_side_thickness_defaults_fall_back(example03):
    """No per-side override: both sides resolve to the shared --min-thickness."""
    pytest.importorskip("pymatgen")
    substrate, film = example03._load_bulks()
    _, config = example03._build_candidates(substrate, film, _fake_args())

    assert config.slab_thicknesses() == (14.4, 14.4)
    assert config.min_thickness == 14.4


def test_example_03_per_side_thickness_overrides_are_forwarded(example03):
    """Explicit per-side overrides win over the shared value, per side."""
    pytest.importorskip("pymatgen")
    substrate, film = example03._load_bulks()
    _, config = example03._build_candidates(
        substrate,
        film,
        _fake_args(min_thickness_substrate=18.0, min_thickness_film=21.6),
    )

    assert config.slab_thicknesses() == (18.0, 21.6)
    # The shared value is left alone, not overwritten by the override.
    assert config.min_thickness == 14.4


def test_example_03_default_knobs_leave_the_ledger_unchanged(example03, tmp_path):
    """Default --dtype/--min-thickness* reproduce the pre-existing ledger.

    The reference point is the 14.4 A plateau documented in
    docs/validation_cu_ni_001.md: 8 Cu layers on the substrate side and 10
    Ni layers on the film side, unchanged by the new flags.
    """
    pytest.importorskip("pymatgen")
    outdir = tmp_path / "out"
    code = _run_multi(example03, outdir)

    assert code == 0
    row = _csv_row(outdir)
    assert row["min_thickness_A"] == "14.4000"
    assert row["n_layers_sub"] == "8"
    assert row["n_layers_film"] == "10"
    assert row["vacuum_A"] == "15.0000"


def test_example_03_shared_thickness_matches_per_side_columns(example03, tmp_path):
    """Without per-side overrides the per-side columns equal the shared value.

    ``min_thickness_substrate_A`` / ``min_thickness_film_A`` record the
    *resolved* thicknesses from ``config.slab_thicknesses()``; with no
    override each side falls back to the shared ``--min-thickness``, so
    all three columns must agree.
    """
    pytest.importorskip("pymatgen")
    outdir = tmp_path / "out"
    code = _run_multi(example03, outdir)

    assert code == 0
    row = _csv_row(outdir)
    assert row["min_thickness_A"] == "14.4000"
    assert row["min_thickness_substrate_A"] == "14.4000"
    assert row["min_thickness_film_A"] == "14.4000"


def test_example_03_per_side_thickness_columns(example03, tmp_path):
    """Per-side overrides land in the CSV as resolved values.

    The shared ``min_thickness_A`` keeps reporting ``--min-thickness``
    (14.4 A), while the two per-side columns report what each side was
    actually cut at -- 18.0 A on the substrate, the shared 14.4 A on the
    film.  Before these columns existed the row showed
    ``min_thickness_A=14.4`` next to ``n_layers_sub=10`` (a layer count
    generated from 18.0 A), which no reader could reconcile.
    """
    pytest.importorskip("pymatgen")
    outdir = tmp_path / "out"
    code = _run_multi(
        example03,
        outdir,
        "--min-thickness-substrate",
        "18.0",
        "--min-thickness-film",
        "14.4",
    )

    assert code == 0
    row = _csv_row(outdir)
    assert row["min_thickness_A"] == "14.4000"
    assert row["min_thickness_substrate_A"] == "18.0000"
    assert row["min_thickness_film_A"] == "14.4000"
    # Self-consistency: the substrate layer count follows the recorded
    # per-side thickness (18.0 A cuts 10 Cu planes, the 14.4 A plateau
    # cuts 8), while the film side keeps the 14.4 A count.
    assert row["n_layers_sub"] == "10"
    assert row["n_layers_film"] == "10"
