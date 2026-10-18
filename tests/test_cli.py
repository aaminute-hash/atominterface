"""CLI smoke tests: argument wiring must work without optional heavy deps."""

from __future__ import annotations

from typer.testing import CliRunner

from atominterface import __version__
from atominterface.cli import app

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
