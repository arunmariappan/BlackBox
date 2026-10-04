"""The `blackbox` command line."""

from typing import Annotated

import typer

from blackbox import __version__

app = typer.Typer(name="blackbox", help="A flight recorder for AI agents.", no_args_is_help=True)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"blackbox {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Print the version and exit.")
    ] = False,
) -> None:
    """A flight recorder for AI agents: record, replay, fork, score and cluster agent runs."""
