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


def _api_url() -> str:
    return settings().server.base_url


def api_request(method: str, path: str, **kwargs: object) -> object:
    """Call the running BlackBox server's API."""
    import httpx

    try:
        response = httpx.request(method, f"{_api_url()}{path}", timeout=30, trust_env=False, **kwargs)  # type: ignore[arg-type]
    except httpx.ConnectError:
        console.print(f"[red]BlackBox isn't running at {_api_url()}[/]; start it with [bold]blackbox serve[/].")
        raise typer.Exit(1) from None
    if response.status_code >= 400:
        console.print(f"[red]{method} {path} failed ({response.status_code}):[/] {response.text}")
        raise typer.Exit(1)
    return response.json()


@app.command()
def serve() -> None:
    """Start BlackBox: UI, REST API and OTLP receiver, proxy listeners and the worker."""
    import logging

    from blackbox.server import Running
    from blackbox.server import serve as serve_forever

    config = settings()
    logging.basicConfig(level=config.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    def ready(running: Running) -> None:
        console.print(f"BlackBox {__version__} on [bold]{running.base_url}[/] (OTLP at {running.base_url}/v1/traces)")
        for name, _, _ in running.servers[1:]:
            console.print(f"  {name}")
        console.print("Ctrl+C stops it.")

    asyncio.run(serve_forever(config, ready))


@app.command("run")
def run_command(
    profile: Annotated[str, typer.Argument(help="Profile of the agent to run, e.g. paperpilot.")],
    text: Annotated[str | None, typer.Argument(help="The input, e.g. a question.")] = None,
    top_k: Annotated[int | None, typer.Option("--top-k", help="PaperPilot: chunks to retrieve.")] = None,
    model: Annotated[str | None, typer.Option("--model", help="Model the agent should use.")] = None,
    wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Wait for the run to complete.")] = True,
) -> None:
    """Start a recorded run of an agent with a traceparent BlackBox chose."""
    import time

    if text is None:
        raise typer.BadParameter("give an input")
    options: dict[str, object] = {}
    if top_k is not None:
        options["top_k"] = top_k
    if model is not None:
        options["model"] = model
    started = api_request("POST", "/api/runs", json={"profile": profile, "input": text, "options": options})
    assert isinstance(started, dict)
    console.print(f"run [bold]{started['run_id']}[/] started: {started['url']}")
    if not wait:
        return
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        run = api_request("GET", f"/api/runs/{started['run_id']}")
        assert isinstance(run, dict)
        if run["status"] != "open":
            console.print(
                f"{run['status']}: ending [bold]{run['ending']}[/], {run['step_count']} steps, "
                f"{run['input_tokens'] + run['output_tokens']} tokens"
            )
            return
        time.sleep(1)
    console.print("[yellow]still open after 30 minutes[/]")


@runs_app.command("list")
def runs_list(
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    ending: Annotated[str | None, typer.Option("--ending")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    """List recent runs."""
    from rich.markup import escape
    from rich.table import Table

    from blackbox.web.format import fmt_duration, fmt_time

    runs = run_with_store(lambda store: store.reader.runs(profile=profile, ending=ending, limit=limit))
    table = Table(
        "id", "started (UTC)", "profile", "source", "status", "ending", "steps", "tokens", "duration", "input"
    )
    for run in runs:
        table.add_row(
            run.id,
            fmt_time(run.started_ms),
            run.profile or "",
            run.source,
            run.status,
            run.ending or "",
            str(run.step_count),
            str(run.input_tokens + run.output_tokens),
            fmt_duration(run.duration_ms),
            escape((run.input_text or "")[:50]),
        )
    console.print(table)


@runs_app.command("show")
def runs_show(run: Annotated[str, typer.Argument(help="Run id, trace id or unique id prefix.")]) -> None:
    """Show a run as a tree of nodes and steps."""
    from rich.markup import escape
    from rich.tree import Tree

    from blackbox.store.models import Run, Step
    from blackbox.web.format import fmt_duration

    async def load(store: Store) -> tuple[Run, list[Step]]:
        found = await store.reader.find_run(run)
        if found is None:
            raise typer.BadParameter(f"no run {run!r}")
        return found, list(await store.reader.steps(found.id))

    found, steps = run_with_store(load)
    tree = Tree(
        f"[bold]{found.id}[/] {found.profile or '?'} · {found.status} · ending [bold]{found.ending}[/] · "
        f"{fmt_duration(found.duration_ms)} · {found.input_tokens}+{found.output_tokens} tokens"
    )
    if found.input_text:
        tree.add(f"input: {escape(found.input_text[:120])}")
    node_branch = None
    current = object()
    for step in steps:
        if step.node != current:
            current = step.node
            node_branch = tree.add(f"[cyan]{escape(step.node or 'unknown')}[/]")
        assert node_branch is not None
        detail = step.tool_name or step.model or ""
        tokens = f" {step.input_tokens}→{step.output_tokens} tok" if step.input_tokens is not None else ""
        status = "" if step.status == "ok" else f" [red]{step.status}[/]"
        node_branch.add(f"#{step.idx} {step.kind} {escape(detail)}{tokens} {fmt_duration(step.latency_ms)}{status}")
    if found.output_text:
        tree.add(f"output: {escape(found.output_text[:200])}")
    console.print(tree)
