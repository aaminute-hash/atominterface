"""``atominterface builders`` sub-commands."""

from __future__ import annotations

import typer

import atominterface.builders  # noqa: F401  (import side effect: registers built-in builders)
from atominterface.builders.registry import builder_registry

app = typer.Typer(
    help="Inspect the interface builders known to this installation.",
    no_args_is_help=True,
)


@app.command("list-builders")
def list_builders() -> None:
    """List every registered interface builder."""
    builders = builder_registry.list()
    if not builders:
        typer.echo("No builders registered.")
        raise typer.Exit(code=1)
    for name in sorted(builders):
        typer.echo(f"{name}\t{builders[name].__name__}")


if __name__ == "__main__":
    app()
