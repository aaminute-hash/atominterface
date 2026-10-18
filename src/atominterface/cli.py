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
# CLI unusable, which is why they are attached defensively.  A ``None``
# command name merges the module's commands at the top level (single-command
# modules such as ``cli_build``); otherwise the module becomes a named group.
_SUBCOMMANDS: tuple[tuple[str, str | None], ...] = (
    ("atominterface.cli_build", None),
    ("atominterface.cli_builders", "builders"),
)
for _module_path, _command_name in _SUBCOMMANDS:
    try:
        _subcommand = importlib.import_module(_module_path)
    except ModuleNotFoundError:  # pragma: no cover - optional sub-command
        continue
    if _command_name is None:
        app.add_typer(_subcommand.app)
    else:
        app.add_typer(_subcommand.app, name=_command_name)


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
