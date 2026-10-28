"""CLI smoke tests: argument wiring must work without optional heavy deps."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
from ase import Atoms
from typer.testing import CliRunner

from atominterface import __version__, cli_build
from atominterface.builders.base import Candidate
from atominterface.cli import app
from atominterface.relax import RelaxResult
from atominterface.run_config import load_run_config

runner = CliRunner()


def _combined_output(result) -> str:
    """Join stdout with stderr across Click generations.

    Click >= 8.2 captures stderr separately, so ``result.output`` only
    holds stdout.  Click < 8.2 mixes both into ``result.output`` and makes
    ``result.stderr`` *raise* ``ValueError("stderr not separately
    captured")`` -- an attribute that exists but explodes, which is why a
    plain ``getattr`` default does not help.
    """
    output = result.output or ""
    try:
        stderr = result.stderr or ""
    except ValueError:
        stderr = ""
    return output + stderr


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(text: str) -> str:
    """Strip ANSI colour and attribute codes before matching on output.

    Rich (through Typer) colourises CLI output whenever it decides the
    stream is a terminal, and that decision is environment-dependent:
    GitHub Actions enables it, a plain Windows console does not. The
    escape sequences land *inside* option names, so
    ``"--symmetric/--no-symmetric" in output`` passes locally and fails
    on CI with the flag rendered as
    ``--\\x1b[1;36m-symmetric\\x1b[0m...``. Assertions must not depend on
    whether the environment claims to be a terminal.
    """
    return _ANSI_ESCAPE.sub("", text)


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in _combined_output(result)


def test_builders_list_builders() -> None:
    result = runner.invoke(app, ["builders", "list-builders"])
    assert result.exit_code == 0
    assert "coherent" in _combined_output(result)


def test_build_rejects_unknown_builder(tmp_path) -> None:
    substrate = tmp_path / "substrate.cif"
    film = tmp_path / "film.cif"
    substrate.write_text("data_dummy\n", encoding="utf-8")
    film.write_text("data_dummy\n", encoding="utf-8")

    result = runner.invoke(app, ["build", str(substrate), str(film), "--builder", "nope"])

    assert result.exit_code != 0
    assert "unknown builder" in _combined_output(result)


@pytest.fixture
def fake_build(monkeypatch):
    """Exercise the actual config/CLI without expensive lattice enumeration."""
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
        loads=[],
        config=None,
        previews=0,
    )

    class FakeBuilder:
        def __init__(self, *, substrate, film, config):
            state.config = config
            state.substrate = substrate
            state.film = film
            self.last_build_report = state.report

        def build(self):
            return state.candidates

        def preview_matches(self, limit):
            state.previews += 1
            assert limit == 5
            return []

    def load(path):
        state.loads.append(path)
        return SimpleNamespace(composition=SimpleNamespace(reduced_formula="NaCl"))

    monkeypatch.setattr(cli_build, "_resolve_builder", lambda name: FakeBuilder)
    monkeypatch.setattr(cli_build, "_load_structure", load)
    return state


def test_build_wires_charge_options(fake_build, tmp_path):
    result = runner.invoke(
        app,
        [
            "build",
            "substrate.cif",
            "film.cif",
            "--charge-policy",
            "require-nonpolar",
            "--substrate-oxidation",
            "Na=1",
            "--substrate-oxidation",
            "Cl=-1",
            "--film-oxidation",
            "Cu=0",
            "--charge-tolerance",
            "0",
            "--dipole-tolerance",
            "0.02",
            "--outdir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, _combined_output(result)
    config = fake_build.config
    assert config.charge_policy == "require-nonpolar"
    assert config.substrate_oxidation == {"Na": 1.0, "Cl": -1.0}
    assert config.film_oxidation == {"Cu": 0.0}
    assert config.substrate_charges is None
    assert config.film_charges is None
    assert config.charge_tolerance == 0
    assert config.dipole_tolerance == 0.02
    assert len(fake_build.loads) == 2


def test_build_default_charge_options(fake_build, tmp_path):
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", "--outdir", str(tmp_path)])
    assert result.exit_code == 0, _combined_output(result)
    config = fake_build.config
    assert config.charge_policy == "report"
    assert config.substrate_oxidation is None
    assert config.film_oxidation is None
    assert config.charge_tolerance == 1e-6
    assert config.dipole_tolerance == 1e-3


def test_build_wires_symmetric_option(fake_build, tmp_path):
    result = runner.invoke(
        app, ["build", "sub.cif", "film.cif", "--symmetric", "--outdir", str(tmp_path)]
    )
    assert result.exit_code == 0, _combined_output(result)
    assert fake_build.config.symmetric_slabs is True


def test_build_wires_no_symmetric_option(fake_build, tmp_path):
    result = runner.invoke(
        app, ["build", "sub.cif", "film.cif", "--no-symmetric", "--outdir", str(tmp_path)]
    )
    assert result.exit_code == 0, _combined_output(result)
    assert fake_build.config.symmetric_slabs is False


def test_build_default_symmetric_option_is_auto(fake_build, tmp_path):
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", "--outdir", str(tmp_path)])
    assert result.exit_code == 0, _combined_output(result)
    # Unspecified means warn-only: never silently transform a structure.
    assert fake_build.config.symmetric_slabs == "auto"


def test_other_builder_rejects_symmetric_flag(fake_build, tmp_path):
    result = runner.invoke(
        app, ["build", "sub.cif", "film.cif", "--builder", "plugin", "--symmetric"]
    )
    assert result.exit_code == 2
    output = _plain(_combined_output(result))
    assert "does not support charge" in output
    # Matched in two pieces: the flag is rendered as
    # ``--symmetric/--no-symmetric`` by some Click/Typer releases and as
    # ``--symmetric/-no-symmetric`` by others, so asserting the whole
    # spelling pins the suite to one dependency version.
    assert "--symmetric" in output
    assert "no-symmetric" in output
    assert fake_build.loads == []


@pytest.mark.parametrize("option", ["--substrate-oxidation", "--film-oxidation"])
@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("Na", "Element=value"),
        ("=1", "Element=value"),
        ("Na=", "Element=value"),
        ("NotAnElement=1", "unknown element"),
        ("X=0", "unknown element"),
        ("na=1", "unknown element"),
        ("Na=text", "finite"),
        ("Na=1=2", "finite"),
        ("Na=nan", "finite"),
        ("Na=inf", "finite"),
        ("Na=-inf", "finite"),
    ],
)
def test_build_rejects_bad_oxidation(fake_build, tmp_path, option, value, message):
    result = runner.invoke(
        app, ["build", "sub.cif", "film.cif", option, value, "--outdir", str(tmp_path)]
    )
    assert result.exit_code == 2
    assert message in _combined_output(result)
    assert fake_build.loads == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("option", ["--substrate-oxidation", "--film-oxidation"])
def test_build_rejects_duplicate_oxidation(fake_build, option):
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", option, "Na=1", option, "Na=2"])
    assert result.exit_code == 2
    assert "duplicate element" in _combined_output(result)
    assert fake_build.loads == []


@pytest.mark.parametrize("option", ["--charge-tolerance", "--dipole-tolerance"])
@pytest.mark.parametrize("value", ["-0.1", "nan", "inf", "-inf"])
def test_build_rejects_invalid_charge_tolerance(fake_build, option, value):
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", option, value])
    assert result.exit_code == 2
    assert "finite and non-negative" in _combined_output(result)
    assert fake_build.loads == []


def test_build_rejects_invalid_charge_policy(fake_build):
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", "--charge-policy", "neutral-ish"])
    assert result.exit_code == 2
    assert "must be one of" in _combined_output(result)
    assert fake_build.loads == []


def test_build_wires_thickness_and_vacuum_options(fake_build, tmp_path):
    result = runner.invoke(
        app,
        [
            "build",
            "sub.cif",
            "film.cif",
            "--min-thickness",
            "9",
            "--min-thickness-substrate",
            "12",
            "--min-thickness-film",
            "4",
            "--vacuum",
            "20",
            "--outdir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, _combined_output(result)
    config = fake_build.config
    assert config.min_thickness == 9.0
    assert config.min_thickness_substrate == 12.0
    assert config.min_thickness_film == 4.0
    assert config.vacuum_thickness == 20.0


def test_build_default_thickness_and_vacuum_match_the_config_class(fake_build, tmp_path):
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", "--outdir", str(tmp_path)])
    assert result.exit_code == 0, _combined_output(result)
    config = fake_build.config
    # Defaults are read off CoherentInterfaceConfig itself, so the CLI can
    # never drift from the builder's own defaults.
    assert config.min_thickness == 6.0
    assert config.min_thickness_substrate is None
    assert config.min_thickness_film is None
    assert config.vacuum_thickness == 15.0
    assert config.slab_thicknesses() == (6.0, 6.0)


@pytest.mark.parametrize("option", ["--min-thickness-substrate", "--min-thickness-film"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_build_rejects_nonpositive_side_thickness(fake_build, option, value):
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", option, value])
    assert result.exit_code == 2
    assert "finite and positive" in _combined_output(result)
    assert fake_build.loads == []


@pytest.mark.parametrize("option", ["--min-thickness", "--vacuum"])
@pytest.mark.parametrize("value", ["-0.1", "nan", "inf", "-inf"])
def test_build_rejects_invalid_thickness_or_vacuum(fake_build, option, value):
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", option, value])
    assert result.exit_code == 2
    assert "finite and non-negative" in _combined_output(result)
    assert fake_build.loads == []


@pytest.mark.parametrize(
    "options",
    [
        ["--charge-policy", "report"],
        ["--charge-policy", "require-neutral"],
        ["--substrate-oxidation", "Na=1"],
        ["--film-oxidation", "Cl=-1"],
        ["--charge-tolerance", "1e-6"],
        ["--dipole-tolerance", "1e-3"],
    ],
)
def test_other_builder_rejects_explicit_charge_options(fake_build, options):
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", "--builder", "plugin", *options])
    assert result.exit_code == 2
    assert "does not support charge" in _combined_output(result)
    assert "options:" in _combined_output(result)
    assert fake_build.loads == []


def test_other_builder_retains_defaults_without_charge_options(fake_build, tmp_path):
    result = runner.invoke(
        app,
        ["build", "sub.cif", "film.cif", "--builder", "plugin", "--outdir", str(tmp_path)],
    )
    assert result.exit_code == 0, _combined_output(result)
    assert fake_build.config is None
    assert len(fake_build.loads) == 2


def test_charge_filter_empty_reports_real_reasons_and_writes_files(fake_build, tmp_path):
    fake_build.report.update(
        charge_policy="require-neutral",
        attempted=2,
        assembled=2,
        analyzed=2,
        rejected=2,
        rejection_reasons={"film: neutrality charged": 1, "total: neutrality unknown": 2},
        empty_stage="charge_filter",
    )
    result = runner.invoke(
        app,
        [
            "build",
            "sub.cif",
            "film.cif",
            "--charge-policy",
            "require-neutral",
            "--outdir",
            str(tmp_path),
        ],
    )
    output = _combined_output(result)
    assert result.exit_code == 0, output
    assert "No candidate passed the charge/polarity policy" in output
    assert "film: neutrality charged: 1" in output
    assert "total: neutrality unknown: 2" in output
    assert "may overlap" in output
    assert "max-strain" not in output
    assert "max-area" not in output
    assert fake_build.previews == 0
    assert json.loads((tmp_path / "build_report.json").read_text()) == fake_build.report
    assert json.loads((tmp_path / "candidates.json").read_text()) == []
    with (tmp_path / "candidates.csv").open(newline="") as handle:
        table = csv.DictReader(handle)
        assert "total_neutrality" in table.fieldnames
        assert list(table) == []
    assert list(tmp_path.glob("candidate_*.extxyz")) == []


def test_assembly_empty_does_not_offer_strain_advice(fake_build, tmp_path):
    fake_build.report.update(attempted=2, assembly_failed=2, empty_stage="assembly")
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", "--outdir", str(tmp_path)])
    output = _combined_output(result)
    assert result.exit_code == 0, output
    assert "no interface could be assembled" in output
    assert "Assembly failed for 2" in output
    assert "max-strain" not in output
    assert fake_build.previews == 0
    assert json.loads((tmp_path / "build_report.json").read_text()) == fake_build.report


def test_no_lattice_matches_uses_geometry_advice(fake_build, tmp_path):
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", "--outdir", str(tmp_path)])
    assert result.exit_code == 0, _combined_output(result)
    assert "No lattice matches survived the geometry filters" in _combined_output(result)
    assert "--max-area" in _combined_output(result)
    assert fake_build.previews == 1


def test_success_writes_reports_and_shows_charge_summary(fake_build, tmp_path):
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
            "total": group,
            "substrate": {**group, "n_atoms": 1, "n_known": 1},
            "film": {**group, "n_atoms": 1, "n_known": 1},
        },
        "thresholds": fake_build.report["thresholds"],
        "geometry": {"finite_slab": True},
        "snapshot": "input_geometry",
    }
    candidate = Candidate(
        atoms=Atoms("Cu2", positions=[[0, 0, 2], [0, 0, 4]], cell=[3, 3, 10]),
        interface_id="fake-1",
        builder_name="coherent",
        substrate_formula="Cu",
        film_formula="Cu",
        substrate_miller=(0, 0, 1),
        film_miller=(0, 0, 1),
        area=9,
        n_substrate=1,
        n_film=1,
        metadata={"charge_analysis": analysis},
    )
    fake_build.candidates = [candidate]
    fake_build.report.update(
        attempted=1, assembled=1, analyzed=1, kept=1, stopped_at_limit=True, empty_stage=None
    )
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", "--outdir", str(tmp_path)])
    assert result.exit_code == 0, _combined_output(result)
    assert "neutral/nonpolar | neutral/nonpolar | neutral/nonpolar" in result.output
    assert "report covers attempted candidates" in result.output
    assert (tmp_path / "candidate_01.extxyz").is_file()
    assert json.loads((tmp_path / "candidates.json").read_text())[0]["charge_analysis"] == analysis
    assert json.loads((tmp_path / "build_report.json").read_text()) == fake_build.report
    with (tmp_path / "candidates.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["interface_id"] == "fake-1"
    assert row["total_neutrality"] == "neutral"
    assert row["film_polarity"] == "nonpolar"
    assert row["charge_snapshot"] == "input_geometry"


def test_legacy_builder_no_candidates_keeps_preview_advice(capsys):
    match = SimpleNamespace(match_area=8, strain=0.06, shear_strain=0.01)
    builder = SimpleNamespace(preview_matches=lambda limit: [match])
    cli_build._report_no_candidates(builder, max_strain=0.05, max_area=100)
    assert "Re-run with --max-strain 0.06" in capsys.readouterr().out


def _relax_candidate():
    """One fake candidate for the ``--relax`` wiring tests."""
    return Candidate(
        atoms=Atoms("Cu2", positions=[[0, 0, 2], [0, 0, 4]], cell=[3, 3, 10]),
        interface_id="fake-1",
        builder_name="coherent",
        substrate_formula="Cu",
        film_formula="Cu",
        substrate_miller=(0, 0, 1),
        film_miller=(0, 0, 1),
        area=9,
        n_substrate=1,
        n_film=1,
        metadata={"charge_analysis": None},
    )


def _valid_relax_result(atoms) -> RelaxResult:
    """A converged, valid RelaxResult: keeps ``build`` off the exit-2 path."""
    return RelaxResult(
        atoms=atoms,
        n_steps=1,
        converged=True,
        valid=True,
        energies_finite=True,
        forces_finite=True,
        final_energy=0.0,
        metadata={"committee_enabled": False, "ood_monitoring": "heuristics_only"},
    )


@pytest.mark.parametrize(
    "flags",
    [
        ["--committee-backend", "dummy,dummy"],
        ["--committee-backend", "dummy", "--committee-backend", "dummy"],
    ],
)
def test_build_wires_committee_backends_into_from_calculators(
    fake_build, tmp_path, monkeypatch, flags
):
    """--committee-backend mounts the extra models via from_calculators.

    ``dummy`` is the only backend name that validates without optional
    heavy dependencies; the pair exercises the two accepted spellings
    (comma-separated and repeated).
    """
    fake_build.candidates = [_relax_candidate()]
    captured: list = []

    class _FakeRelaxer:
        def run(self, atoms):
            return _valid_relax_result(atoms)

    def fake_from_calculators(calculators, *, primary=0, **kw):
        captured.append((list(calculators), primary, kw))
        return _FakeRelaxer()

    monkeypatch.setattr(
        "atominterface.relax.GuardedRelaxer.from_calculators", fake_from_calculators
    )
    monkeypatch.setattr(
        "atominterface.backends.get_calculator",
        lambda name, **kwargs: lambda atoms: SimpleNamespace(backend=name),
    )
    result = runner.invoke(
        app,
        [
            "build",
            "sub.cif",
            "film.cif",
            "--relax",
            "--backend",
            "dummy",
            *flags,
            "--outdir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, _combined_output(result)
    calcs, primary, kw = captured[0]
    assert primary == 0
    assert len(calcs) == 3  # primary factory + two resolved committee members
    assert [member.backend for member in calcs[1:]] == ["dummy", "dummy"]
    assert kw["fmax"] == 0.05
    assert kw["max_steps"] == 200
    assert kw["ood_threshold"] == 0.8
    assert kw["quarantine_dir"] == str(tmp_path / "flagged")
    # The CLI does not override the calibrated committee threshold.
    assert "committee_threshold" not in kw
    assert "committee backend(s): dummy, dummy" in result.output


def test_build_without_committee_backend_keeps_direct_construction(
    fake_build, tmp_path, monkeypatch
):
    """No --committee-backend: the historical direct GuardedRelaxer path."""
    fake_build.candidates = [_relax_candidate()]
    created: list = []

    class _RecordingRelaxer:
        def __init__(self, calculator, **kwargs):
            created.append((calculator, kwargs))

        @classmethod
        def from_calculators(cls, calculators, *, primary=0, **kwargs):
            raise AssertionError("from_calculators must not be used without --committee-backend")

        def run(self, atoms):
            return _valid_relax_result(atoms)

    monkeypatch.setattr("atominterface.relax.GuardedRelaxer", _RecordingRelaxer)
    monkeypatch.setattr(
        "atominterface.backends.get_calculator",
        lambda name, **kwargs: lambda atoms: SimpleNamespace(backend=name),
    )
    result = runner.invoke(
        app, ["build", "sub.cif", "film.cif", "--relax", "--outdir", str(tmp_path)]
    )
    assert result.exit_code == 0, _combined_output(result)
    assert len(created) == 1
    calculator, kwargs = created[0]
    assert not isinstance(calculator, list)  # a single primary, not a committee list
    assert "committee_calculators" not in kwargs
    assert kwargs["ood_threshold"] == 0.8
    assert "committee backend(s)" not in _combined_output(result)


def test_build_rejects_unknown_committee_backend(fake_build):
    result = runner.invoke(app, ["build", "sub.cif", "film.cif", "--committee-backend", "nope"])
    assert result.exit_code == 2
    assert "unknown backend" in _combined_output(result)
    assert fake_build.loads == []


def test_build_records_committee_backend_in_run_config(fake_build, tmp_path):
    """The effective committee list is recorded and an empty value disables it."""
    result = runner.invoke(
        app,
        [
            "build",
            "sub.cif",
            "film.cif",
            "--committee-backend",
            "dummy, dummy",
            "--outdir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, _combined_output(result)
    assert load_run_config(tmp_path / "run_config.yaml")["committee_backend"] == ["dummy", "dummy"]

    empty = runner.invoke(
        app,
        [
            "build",
            "sub.cif",
            "film.cif",
            "--committee-backend",
            "",
            "--outdir",
            str(tmp_path / "empty"),
        ],
    )
    assert empty.exit_code == 0, _combined_output(empty)
    assert load_run_config(tmp_path / "empty" / "run_config.yaml")["committee_backend"] == []


def test_help_and_version_without_pymatgen():
    script = """
import importlib.abc
import re
import sys
class BlockPymatgen(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pymatgen" or fullname.startswith("pymatgen."):
            raise ModuleNotFoundError("pymatgen deliberately unavailable")
sys.meta_path.insert(0, BlockPymatgen())
from typer.testing import CliRunner
from atominterface.cli import app
# Rich colourises help output when it thinks the stream is a terminal,
# which GitHub Actions does and a plain console does not. The escape
# sequences land inside option names, so strip them before matching.
_ANSI = re.compile(r"\\x1b\\[[0-9;]*[A-Za-z]")
def plain(text):
    return _ANSI.sub("", text)
runner = CliRunner()
for args in (["build", "--help"], ["--version"]):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, (result.output, result.exception)
    if args[0] == "build":
        assert "--charge-policy" in plain(result.output)
        assert "--substrate-oxidation" in plain(result.output)
print("optional-dependency smoke test passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
