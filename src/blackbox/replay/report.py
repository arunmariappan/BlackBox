"""The fidelity report: how each step of a replay or fork was served, where it diverged, and whether the output is
the same. Later phases add judge and metric changes through `Services.report_hooks`."""

from typing import TYPE_CHECKING, Any

from blackbox.runs.context import StepDraft, load_run_context

if TYPE_CHECKING:
    from blackbox.proxy.sessions import ReplaySession
    from blackbox.services import Services


def _reply(step: StepDraft | None) -> Any:
    if step is None:
        return None
    view = step.view
    return view.get("reply", view.get("hits", view.get("tool", view.get("count"))))


def _served(served_from: str) -> str:
    return "tape" if served_from.startswith("tape:") else served_from


async def build_report(
    services: Services,
    session: ReplaySession,
    source_run_id: str,
    replay_run_id: str,
    *,
    rebuilt: bool = False,
    entry_error: str | None = None,
) -> dict[str, Any]:
    store = services.store
    source = await store.reader.run(source_run_id)
    replay = await store.reader.run(replay_run_id)
    assert source is not None and replay is not None
    src = await load_run_context(store, source)
    rep = await load_run_context(store, replay)
    profile = services.profiles.find(source.profile) or session.profile
    source_steps = {step.idx: step for step in src.steps}
    source_by_exchange = {step.exchange_id: step for step in src.steps if step.exchange_id}
    exchanges = {exchange.id: exchange for exchange in rep.exchanges}
    steps: list[dict[str, Any]] = []
    for step in rep.steps:
        exchange = exchanges.get(step.exchange_id or "")
        served_from = exchange.row.served_from if exchange is not None else "span"
        served = _served(served_from)
        tape_step = None
        if served == "tape":
            tape_source = source_by_exchange.get(served_from.split(":", 1)[1])
            tape_step = tape_source.idx if tape_source else None
        entry: dict[str, Any] = {
            "idx": step.idx,
            "node": step.node,
            "kind": step.kind,
            "served": served,
            "tape_step": tape_step,
            "model": step.model,
            "status": step.status,
        }
        if exchange is not None and exchange.row.divergence:
            entry["divergence"] = exchange.row.divergence
        if exchange is not None and exchange.sent_request_body is not None:
            entry["sent_differs"] = True
        if served in ("live", "patched"):
            entry["differs_from_tape"] = _reply(step) != _reply(source_steps.get(step.idx))
        steps.append(entry)
    source_output = profile.read_output(src)
    replay_output = profile.read_output(rep)
    outputs = profile.compare_outputs(source_output, replay_output)
    all_tape = bool(steps) and all(s["served"] == "tape" for s in steps)
    exact = (
        all_tape
        and session.first_divergence is None
        and outputs["equal"]
        and len(steps) == len(src.steps)
        and replay.status == "complete"
    )
    report: dict[str, Any] = {
        "session_id": session.id,
        "mode": session.spec.mode,
        "fork_step": session.spec.fork_step,
        "model": session.spec.model,
        "patches": [patch.name for patch in session.spec.patches],
        "source_run": source.id,
        "replay_run": replay.id,
        "rebuilt_input": rebuilt,
        "entry_error": entry_error,
        "source_steps": len(src.steps),
        "replay_steps": len(steps),
        "steps": steps,
        "first_divergence": session.first_divergence,
        "went_live_at_seq": session.went_live_at,
        "live_calls": sum(1 for s in steps if s["served"] in ("live", "patched")),
        "outputs": outputs,
        "endings": {"source": source.ending, "replay": replay.ending},
        "exact": exact,
    }
    for hook in services.report_hooks:
        report.update(await hook(services, session, src, rep))
    return report


def summary_lines(report: dict[str, Any]) -> list[str]:
    """A few plain lines for logs and the CLI."""
    if "error" in report and "steps" not in report:
        return [f"replay failed: {report['error']}"]
    lines = [
        f"{report['mode']} replay of {report['source_run']} → {report['replay_run']}: "
        f"{'EXACT' if report['exact'] else 'not exact'}; {report['live_calls']} live calls; "
        f"output {'identical' if report['outputs']['equal'] else 'differs'}; "
        f"ending {report['endings']['source']} → {report['endings']['replay']}"
    ]
    divergence = report.get("first_divergence")
    if divergence:
        lines.append(f"first divergence at tape step {divergence.get('step')} ({divergence.get('node')})")
    return lines
