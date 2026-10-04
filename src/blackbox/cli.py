"""The `blackbox` command line."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from blackbox import __version__
from blackbox.config import Settings, load_settings
from blackbox.store import Store

app = typer.Typer(name="blackbox", help="A flight recorder for AI agents.", no_args_is_help=True)
db_app = typer.Typer(help="Database maintenance.", no_args_is_help=True)
runs_app = typer.Typer(help="List, show, export and import runs.", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(runs_app, name="runs")

console = Console()
_state: dict[str, Path | None] = {"config": None}


def settings() -> Settings:
    return load_settings(_state["config"])


def run_with_store[T](fn: Callable[[Store], Awaitable[T]]) -> T:
    async def main() -> T:
        store = await Store.open(settings().store.path)
        try:
            return await fn(store)
        finally:
            await store.close()

    return asyncio.run(main())


def _version(value: bool) -> None:
    if value:
        typer.echo(f"blackbox {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Print the version and exit.")
    ] = False,
    config: Annotated[
        Path | None, typer.Option("--config", help="Configuration file (default: $BLACKBOX_CONFIG or blackbox.toml).")
    ] = None,
) -> None:
    """A flight recorder for AI agents: record, replay, fork, score and cluster agent runs."""
    _state["config"] = config


@db_app.command("upgrade")
def db_upgrade() -> None:
    """Create the database or upgrade it to the latest schema."""
    from blackbox.store import db

    path = settings().store.path
    db.upgrade(path)
    console.print(f"database [bold]{path}[/] is at revision {db.current_revision(path)}")


@db_app.command("prune")
def db_prune(
    older_than: Annotated[str, typer.Option("--older-than", help="Age cutoff, e.g. 30d or 12h.")] = "30d",
) -> None:
    """Delete runs older than the cutoff (except baseline and labelled runs), then unreferenced blobs."""
    from blackbox.store.prune import prune
    from blackbox.util import now_ms, parse_duration_ms

    cutoff = now_ms() - parse_duration_ms(older_than)
    result = run_with_store(lambda store: prune(store, cutoff_ms=cutoff))
    console.print(f"deleted {result.runs} runs and {result.blobs} blobs")


@runs_app.command("export")
def runs_export(
    run: Annotated[str, typer.Argument(help="Run id, trace id or unique id prefix.")],
    out: Annotated[Path, typer.Option("--out", help="Bundle directory to write.")],
) -> None:
    """Export a run as a bundle (run.json plus blobs/)."""
    from blackbox.store.bundles import export_run

    async def go(store: Store) -> Path:
        found = await store.reader.find_run(run)
        if found is None:
            raise typer.BadParameter(f"no run {run!r}")
        return await export_run(store, found.id, out)

    console.print(f"wrote {run_with_store(go)}")


@runs_app.command("import")
def runs_import(
    bundle: Annotated[Path, typer.Argument(help="Bundle directory (containing run.json).")],
) -> None:
    """Import a run bundle. Importing the same bundle twice changes nothing."""
    from blackbox.store.bundles import import_bundle

    run_id = run_with_store(lambda store: import_bundle(store, bundle))
    console.print(f"imported run {run_id}")
