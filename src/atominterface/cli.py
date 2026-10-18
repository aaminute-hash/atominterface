"""AtomInterface command-line entry point.

Thin typer app.  Every subcommand lives in its own module so the CLI
stays loadable even when optional dependencies (pymatgen, mace,
chgnet) are missing.
"""

from __future__ import annotations

import importlib
import sys

import typer

from atominterface import __version__

app = typer.Typer(
    name="atominterface",
    help="Reliable interface modeling for AI4S.",
    no_args_is_help=True,
    add_completion=False,
)

# Sub-command modules are optional: a missing one must not make the whole
# CLI unusable, which is why they are attached defensively.
_SUBCOMMANDS = {
    "atominterface.cli_build": "build",
    "atominterface.cli_builders": "builders",
}
for _module_path, _command_name in _SUBCOMMANDS.items():
    try:
        app.add_typer(importlib.import_module(_module_path).app, name=_command_name)
    except ModuleNotFoundError:  # pragma: no cover - optional sub-command
        continue


@app.callback(invoke_without_command=True)
def callback(
    version: bool = typer.Option(False, "--version", "-V", help="Show version and exit."),
) -> None:
    """Reliable interface modeling for AI4S."""
    if version:
        typer.echo(__version__)
        raise typer.Exit()


def main(argv: list[str] | None = None) -> int:
    app(argv if argv is not None else sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
