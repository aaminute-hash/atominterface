"""AtomInterface command-line entry point.

Thin typer app.  Every subcommand lives in its own module so the CLI
stays loadable even when optional dependencies (pymatgen, mace,
chgnet) are missing.
"""

from __future__ import annotations

import sys

import typer

app = typer.Typer(
    name="atominterface",
    help="Reliable interface modeling for AI4S.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(__import__("atominterface.cli_build", fromlist=["app"]).app)
app.add_typer(__import__("atominterface.cli_builders", fromlist=["app"]).app)


@app.callback()
def callback(
    version: bool = typer.Option(False, "--version", "-V", help="Show version and exit."),
) -> None:
    """Reliable interface modeling for AI4S."""


def main(argv: list[str] | None = None) -> int:
    app(argv if argv is not None else sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
