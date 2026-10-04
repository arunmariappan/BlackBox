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
    dataset: Annotated[Path | None, typer.Option("--dataset", help="Run every reviewed question in a file.")] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="With --dataset: at most N questions.")] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation for long GPU batches.")] = False,
    task: Annotated[str | None, typer.Option("--task", help="OpsDesk: the task to run.")] = None,
    mode: Annotated[str | None, typer.Option("--mode", help="OpsDesk: seeded or chaotic.")] = None,
    all_tasks: Annotated[bool, typer.Option("--all-tasks", help="OpsDesk: every task in turn.")] = False,
    repeats: Annotated[int, typer.Option("--repeats", help="With --all-tasks: runs per task.")] = 1,
    wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Wait for the run to complete.")] = True,
) -> None:
    """Start a recorded run of an agent with a traceparent BlackBox chose (or a whole question set or task list)."""
    options: dict[str, object] = {}
    if top_k is not None:
        options["top_k"] = top_k
    if model is not None:
        options["model"] = model
    if mode is not None:
        options["mode"] = mode
    if dataset is not None:
        _run_dataset(profile, dataset, options, limit=limit, yes=yes)
        return
    if all_tasks:
        _run_all_tasks(profile, options, repeats=repeats, limit=limit, yes=yes)
        return
    if task is not None:
        options["task"] = task
    if text is None and task is None:
        raise typer.BadParameter("give an input, --task, --all-tasks or --dataset FILE")
    started = api_request("POST", "/api/runs", json={"profile": profile, "input": text or "", "options": options})
    assert isinstance(started, dict)
    console.print(f"run [bold]{started['run_id']}[/] started: {started['url']}")
    if wait:
        _wait_for_run(str(started["run_id"]))


def _wait_for_run(run_id: str, timeout_s: float = 1800) -> dict[str, object] | None:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        run = api_request("GET", f"/api/runs/{run_id}")
        assert isinstance(run, dict)
        if run["status"] != "open":
            console.print(
                f"{run['status']}: ending [bold]{run['ending']}[/], {run['step_count']} steps, "
                f"{run['input_tokens'] + run['output_tokens']} tokens"
            )
            return run
        time.sleep(1)
    console.print("[yellow]still open after 30 minutes[/]")
    return None


GPU_CONFIRM_RUNS = 30


def _run_all_tasks(profile: str, options: dict[str, object], *, repeats: int, limit: int | None, yes: bool) -> None:
    """Run every OpsDesk task `repeats` times, one run at a time."""
    import httpx

    env_url = str(settings().profiles.get(profile, {}).get("env_url", "http://127.0.0.1:8221"))
    tasks = httpx.get(f"{env_url}/_tasks", timeout=10, trust_env=False).json()
    plan = [task["id"] for task in tasks for _ in range(repeats)][: limit or None]
    if len(plan) > GPU_CONFIRM_RUNS and not yes:
        typer.confirm(
            f"{len(plan)} runs use the GPU for a long time (this PC has shut down during long GPU runs). Go on?",
            abort=True,
        )
    passed = 0
    for i, task_id in enumerate(plan, start=1):
        body = {"profile": profile, "input": "", "options": {**options, "task": task_id}, "tags": {"task": task_id}}
        started = api_request("POST", "/api/runs", json=body)
        assert isinstance(started, dict)
        console.print(f"[{i}/{len(plan)}] {task_id}: run {started['run_id']}")
        run = _wait_for_run(str(started["run_id"]))
        if run is not None:
            time_limit = 30.0
            import time

            deadline = time.monotonic() + time_limit  # the checker runs just after completion
            while time.monotonic() < deadline:
                detail = api_request("GET", f"/api/runs/{started['run_id']}")
                assert isinstance(detail, dict)
                checker = [s for s in detail["scores"] if s["kind"] == "checker"]
                if checker:
                    passed += checker[0]["label"] == "pass"
                    console.print(f"  checker: {checker[0]['label']}")
                    break
                time.sleep(0.5)
    console.print(f"{passed}/{len(plan)} runs passed the checker")


def _run_dataset(profile: str, path: Path, options: dict[str, object], *, limit: int | None, yes: bool) -> None:
    """Run a question set in sequence, one run at a time, each tagged with its case and expected ending."""
    from blackbox.datasets import load_dataset

    dataset = load_dataset(path)
    questions = dataset.reviewed()[:limit] if limit else dataset.reviewed()
    skipped = len(dataset.questions) - len(dataset.reviewed())
    console.print(f"{len(questions)} questions from {path}" + (f" ({skipped} drafts skipped)" if skipped else ""))
    if len(questions) > GPU_CONFIRM_RUNS and not yes:
        typer.confirm(
            f"{len(questions)} runs use the GPU for a long time (this PC has shut down during long GPU runs). Go on?",
            abort=True,
        )
    for i, question in enumerate(questions, start=1):
        tags: dict[str, object] = {
            "dataset": dataset.name,
            "case": question.id,
            "expected_ending": question.expected_ending,
        }
        tags.update({f"option_{k}": v for k, v in options.items()})
        body = {"profile": profile, "input": question.question, "options": options, "tags": tags}
        started = api_request("POST", "/api/runs", json=body)
        assert isinstance(started, dict)
        console.print(f"[{i}/{len(questions)}] {question.id}: run {started['run_id']}")
        run = _wait_for_run(str(started["run_id"]))
        if run is not None and run.get("ending") != question.expected_ending:
            console.print(f"  [yellow]expected {question.expected_ending}[/]")


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


@app.command()
def replay(
    run: Annotated[str, typer.Argument(help="Run id, trace id or unique id prefix.")],
    from_step: Annotated[int | None, typer.Option("--from-step", help="Fork: tape before step N, live from N.")] = None,
    auto_fork: Annotated[bool, typer.Option("--auto-fork", help="Tape until the first request that differs.")] = False,
    model: Annotated[str | None, typer.Option("--model", help="Model for live steps.")] = None,
    patch: Annotated[Path | None, typer.Option("--patch", help="YAML file of request patches.")] = None,
    speed: Annotated[float, typer.Option("--speed", help="0 = as fast as possible, 1 = recorded timing.")] = 0.0,
    lenient: Annotated[bool, typer.Option("--lenient", help="Exact: serve the next tape step on divergence.")] = False,
    times: Annotated[int, typer.Option("--times", help="Repeat the replay K times.")] = 1,
) -> None:
    """Replay a recorded run exactly, fork it from step N, or auto-fork it after a change."""
    import time

    from rich.markup import escape
    from rich.table import Table

    if from_step is not None and auto_fork:
        raise typer.BadParameter("choose --from-step or --auto-fork, not both")
    mode = "fork" if from_step is not None else "auto_fork" if auto_fork else "exact"
    patches = patch.read_text(encoding="utf-8") if patch is not None else []
    if mode == "exact" and patches:
        console.print(
            "[red]exact replay with patches is refused[/]: a patched request can't match the tape. Try --auto-fork."
        )
        raise typer.Exit(2)
    body = {
        "source_run_id": run,
        "mode": mode,
        "fork_step": from_step,
        "model": model,
        "patches": patches,
        "speed": speed,
        "lenient": lenient,
    }
    failures = 0
    for attempt in range(1, times + 1):
        created = api_request("POST", "/api/sessions", json=body)
        assert isinstance(created, dict)
        console.print(f"[{attempt}/{times}] session {created['session_id']}: {created['url']}")
        while True:
            session = api_request("GET", f"/api/sessions/{created['session_id']}")
            assert isinstance(session, dict)
            if session["status"] != "active":
                break
            time.sleep(0.5)
        report = session.get("result") or {}
        if "steps" not in report:
            console.print(f"[red]{session['status']}: {escape(str(report.get('error')))}[/]")
            failures += 1
            continue
        table = Table("#", "node", "kind", "served", "tape step", "changed")
        for step in report["steps"]:
            changed = step.get("differs_from_tape")
            table.add_row(
                str(step["idx"]),
                escape(step.get("node") or "unknown"),
                step["kind"],
                step["served"],
                str(step.get("tape_step") or ""),
                "" if changed is None else ("response differs" if changed else "same"),
            )
        console.print(table)
        divergence = report.get("first_divergence")
        if divergence:
            console.print(
                f"first divergence: tape step {divergence.get('step')} ({escape(str(divergence.get('node')))})"
            )
        verdict = "[green]EXACT[/]" if report["exact"] else "[yellow]not exact[/]"
        console.print(
            f"{verdict} · {report['live_calls']} live calls · output "
            f"{'identical' if report['outputs']['equal'] else 'differs'} · ending "
            f"{report['endings']['source']} → {report['endings']['replay']} · replay run {report['replay_run']}"
        )
        if mode == "exact" and not report["exact"]:
            failures += 1
    if failures:
        raise typer.Exit(1)


# Judges, labels and datasets ----------------------------------------------------------------------------------------

judge_app = typer.Typer(help="LLM judges: run, calibrate, stability and sensitivity.", no_args_is_help=True)
labels_app = typer.Typer(help="Your labels.", no_args_is_help=True)
dataset_app = typer.Typer(help="Question sets.", no_args_is_help=True)
app.add_typer(judge_app, name="judge")
app.add_typer(labels_app, name="labels")
app.add_typer(dataset_app, name="dataset")


def _judging[T](fn: Callable[[Store, object, dict[str, object]], Awaitable[T]]) -> T:
    """Run `fn(store, runner, judges)` with a local store and an Ollama client."""
    from blackbox.judges.framework import load_judges
    from blackbox.judges.runner import JudgeRunner
    from blackbox.llm.client import OllamaJSON

    config = settings()

    async def go(store: Store) -> T:
        llm = OllamaJSON(config.ollama)
        try:
            return await fn(store, JudgeRunner(store, llm), dict(load_judges(config.ollama.model)))
        finally:
            await llm.close()

    return run_with_store(go)


def _judge_named(judges: dict[str, object], name: str) -> object:
    if name not in judges:
        raise typer.BadParameter(f"no judge {name!r} (known: {', '.join(sorted(judges))})")
    return judges[name]


@judge_app.command("list")
def judge_list() -> None:
    """List the judges with their current versions and trust."""
    from rich.table import Table

    from blackbox.judges import calibrate
    from blackbox.judges.framework import Judge

    async def go(store: Store, runner: object, judges: dict[str, object]) -> None:
        table = Table("judge", "profile", "version", "label question", "held-out κ", "n", "trusted")
        for judge in judges.values():
            assert isinstance(judge, Judge)
            rows = {row.version: row for row in await calibrate.versions(store, judge.name)}
            row = rows.get(judge.version)
            held = (row.agreement or {}).get("held_out", {}) if row else {}
            kappa = held.get("kappa")
            table.add_row(
                judge.name,
                judge.profile,
                judge.version,
                str(judge.label_question or ""),
                f"{kappa:.2f}" if isinstance(kappa, float) else "—",
                str(held.get("n", 0)),
                "yes" if row and row.trusted else "no",
            )
        console.print(table)

    _judging(go)


@judge_app.command("run")
def judge_run(
    name: Annotated[str, typer.Argument(help="Judge name, e.g. pp_faithfulness.")],
    runs: Annotated[list[str] | None, typer.Option("--run", help="A run to judge (repeatable).")] = None,
    profile: Annotated[str | None, typer.Option("--profile", help="Judge the last N runs of a profile.")] = None,
    last: Annotated[int, typer.Option("--last", help="With --profile: how many runs.")] = 20,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Call the model even for inputs already judged.")
    ] = False,
) -> None:
    """Judge runs with the judge's current version."""
    from collections import Counter

    from blackbox.judges.framework import Judge
    from blackbox.judges.runner import JudgeRunner
    from blackbox.store.models import Run

    async def go(store: Store, runner: object, judges: dict[str, object]) -> None:
        judge = _judge_named(judges, name)
        assert isinstance(judge, Judge) and isinstance(runner, JudgeRunner)
        targets: list[Run] = []
        for ref in runs or []:
            found = await store.reader.find_run(ref)
            if found is None:
                raise typer.BadParameter(f"no run {ref!r}")
            targets.append(found)
        if profile or not runs:
            targets.extend(await store.reader.runs(profile=profile or judge.profile, status="complete", limit=last))
        counts: Counter[str] = Counter()
        for run in targets:
            outcome = await runner.judge_run(judge, run, use_cache=not no_cache)
            counts[outcome.status] += 1
            if outcome.label:
                console.print(f"{run.id} {outcome.label} ({outcome.status}) {outcome.rationale or ''}")
        console.print(dict(counts))

    _judging(go)


@judge_app.command("calibrate")
def judge_calibrate(name: Annotated[str, typer.Argument(help="Judge name.")]) -> None:
    """Judge every labelled run with the current version and show each version's agreement with your labels."""
    from rich.table import Table

    from blackbox.judges import calibrate
    from blackbox.judges.framework import Judge
    from blackbox.judges.runner import JudgeRunner

    def fmt(value: object) -> str:
        return f"{value:.2f}" if isinstance(value, float) else "—"

    async def go(store: Store, runner: object, judges: dict[str, object]) -> None:
        judge = _judge_named(judges, name)
        assert isinstance(judge, Judge) and isinstance(runner, JudgeRunner)
        results = await calibrate.calibrate(store, runner, judge)
        table = Table(
            "version", "held-out n", "held-out κ", "95% interval", "all κ", "precision/recall on fail", "trusted"
        )
        for version, stats in results.items():
            held, every = stats["held_out"], stats["all"]
            table.add_row(
                version + (" (current)" if version == judge.version else ""),
                str(held["n"]),
                fmt(held["kappa"]),
                f"{fmt(held['ci'][0])} to {fmt(held['ci'][1])}",
                fmt(every["kappa"]),
                f"{fmt(every['precision_fail'])} / {fmt(every['recall_fail'])}",
                "yes" if stats["trusted"] else "no: " + "; ".join(stats["gate_reasons"]),
            )
        console.print(table)

    _judging(go)


@judge_app.command("stability")
def judge_stability(
    name: Annotated[str, typer.Argument(help="Judge name.")],
    runs: Annotated[int, typer.Option("--runs", help="How many runs to repeat.")] = 20,
    repeats: Annotated[int, typer.Option("--repeats", help="Repeats per run and temperature.")] = 3,
) -> None:
    """How often the judge's verdict flips on the same input, at temperature 0 and 0.7."""
    from blackbox.judges import calibrate
    from blackbox.judges.framework import Judge
    from blackbox.judges.runner import JudgeRunner
    from blackbox.runs.context import load_run_context

    async def go(store: Store, runner: object, judges: dict[str, object]) -> None:
        judge = _judge_named(judges, name)
        assert isinstance(judge, Judge) and isinstance(runner, JudgeRunner)
        inputs = []
        for run in await store.reader.runs(profile=judge.profile, status="complete", limit=runs * 5):
            if judge.applies(run):
                built = judge.input_builder.build(await load_run_context(store, run))
                if built is not None:
                    inputs.append(built)
            if len(inputs) >= runs:
                break
        for result in await calibrate.stability(runner, judge, inputs, repeats=repeats):
            console.print(
                f"temperature {result.temperature}: {result.flipped}/{result.runs} runs flipped "
                f"({result.flip_rate:.0%})"
            )
            if result.temperature == 0 and result.flip_rate > 0.05:
                console.print(
                    "[yellow]above 5% at temperature 0[/]: set `options: {samples: 3, temperature: 0.3}` in "
                    f"{judge.path} (a majority vote of three; it becomes a new version)."
                )

    _judging(go)


@judge_app.command("sensitivity")
def judge_sensitivity(
    name: Annotated[str, typer.Argument(help="Judge name (pp_faithfulness).")] = "pp_faithfulness",
    runs: Annotated[int, typer.Option("--runs", help="How many passing answers to plant a claim in.")] = 10,
) -> None:
    """Append a made-up claim to answers that passed; a faithfulness judge must fail at least 80% of them."""
    from blackbox.judges import calibrate
    from blackbox.judges.framework import Judge
    from blackbox.judges.runner import JudgeRunner
    from blackbox.runs.context import load_run_context

    async def go(store: Store, runner: object, judges: dict[str, object]) -> None:
        judge = _judge_named(judges, name)
        assert isinstance(judge, Judge) and isinstance(runner, JudgeRunner)
        planted: list[dict[str, str]] = []
        for run in await store.reader.runs(profile=judge.profile, status="complete", limit=500):
            scores = [
                s for s in await store.reader.scores(run.id) if s.name == judge.name and s.version == judge.version
            ]
            if scores and scores[0].label == "pass":
                built = judge.input_builder.build(await load_run_context(store, run))
                if built is not None:
                    planted.append(calibrate.with_made_up_claim(built, len(planted)))
            if len(planted) >= runs:
                break
        if not planted:
            console.print("[yellow]no passing runs to test; judge some answered runs first[/]")
            raise typer.Exit(1)
        failed = 0
        for item in planted:
            verdict = await runner.decide(judge, item)
            failed += bool(verdict.valid and verdict.parsed and verdict.parsed.get("verdict") == "fail")
        needed = -(-len(planted) * 8 // 10)
        ok = failed >= needed
        console.print(f"{failed}/{len(planted)} planted claims caught (needs {needed}): {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise typer.Exit(1)

    _judging(go)


@labels_app.command("export")
def labels_export(
    out: Annotated[Path, typer.Option("--out", help="Directory for the bundles.")] = Path(
        "datasets/paperpilot/labelled"
    ),
    question: Annotated[str | None, typer.Option("--question", help="Only runs labelled for this question.")] = None,
) -> None:
    """Export every labelled run (with its labels) as a bundle, so labels survive a database reset."""
    from sqlalchemy import select

    from blackbox.store.bundles import export_run
    from blackbox.store.models import Label

    async def go(store: Store) -> int:
        query = select(Label.run_id).distinct()
        if question:
            query = query.where(Label.question == question)
        async with store.read() as s:
            run_ids = list((await s.execute(query)).scalars())
        for run_id in run_ids:
            await export_run(store, run_id, out / run_id)
        return len(run_ids)

    console.print(f"exported {run_with_store(go)} labelled runs to {out}")


@dataset_app.command("draft")
def dataset_draft(
    profile: Annotated[str, typer.Argument(help="Only paperpilot is supported.")] = "paperpilot",
    count: Annotated[int, typer.Option("--count", help="How many questions to draft.")] = 30,
    out: Annotated[Path, typer.Option("--out", help="YAML file to write the drafts to.")] = Path(
        "datasets/paperpilot/drafts.yaml"
    ),
    index: Annotated[str, typer.Option("--index", help="OpenSearch index of paper chunks.")] = "arxiv-papers-chunks",
) -> None:
    """Sample papers from PaperPilot's OpenSearch index and draft one question per paper for you to review."""
    import random

    import httpx
    import yaml

    from blackbox.datasets import DraftQuestion
    from blackbox.llm.client import OllamaJSON

    if profile != "paperpilot":
        raise typer.BadParameter("only paperpilot has a dataset drafter")
    config = settings()
    target = config.proxy.upstream("opensearch").target

    async def go() -> list[dict[str, object]]:
        query = {"size": count * 4, "query": {"function_score": {"random_score": {"seed": random.randint(1, 10**6)}}}}
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            response = await client.post(f"{target}/{index}/_search", json=query)
        response.raise_for_status()
        papers: dict[str, dict[str, object]] = {}
        for hit in response.json()["hits"]["hits"]:
            source = hit.get("_source", {})
            paper = str(source.get("arxiv_id") or source.get("paper_id") or hit["_id"])
            papers.setdefault(
                paper, {"title": source.get("title"), "text": source.get("chunk_text") or source.get("text")}
            )
        llm = OllamaJSON(config.ollama)
        drafted: list[dict[str, object]] = []
        try:
            for paper, info in list(papers.items()):
                if len(drafted) >= count:
                    break
                prompt = (
                    "Write one question a researcher might ask that this excerpt of an arXiv paper answers. "
                    "Don't mention 'the excerpt' or 'the paper'; ask about the idea itself.\n\n"
                    f"Title: {info['title']}\nExcerpt:\n{info['text']}"
                )
                result = await llm.call([{"role": "user", "content": prompt}], DraftQuestion)
                if result.valid and result.parsed is not None and result.parsed.answerable_from_excerpt:
                    drafted.append(
                        {
                            "id": f"ans-{len(drafted) + 1:02d}",
                            "group": "answerable",
                            "question": result.parsed.question,
                            "expected_ending": "answered",
                            "must_cite": True,
                            "paper_id": paper,
                            "status": "draft",
                        }
                    )
                    console.print(f"{paper}: {result.parsed.question}")
        finally:
            await llm.close()
        return drafted

    drafted = asyncio.run(go())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump({"questions": drafted}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    console.print(f"wrote {len(drafted)} drafts to {out}; review them, then move them into questions.yaml")
