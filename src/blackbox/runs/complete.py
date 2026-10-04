"""Run completion: match a profile, build steps, read output and ending, and queue the `run_completed` job."""

import json
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.profiles import ProfileRegistry
from blackbox.runs.context import RunContext, load_run_context
from blackbox.runs.steps import build_steps
from blackbox.store import PreparedBlob, Store, insert_blob
from blackbox.store.models import Exchange, Job, RecordedValue, Run, Span, Step
from blackbox.util import new_id, now_ms

log = logging.getLogger(__name__)


@dataclass
class CompletionResult:
    trace_id: str
    run_id: str | None
    outcome: str  # complete, deleted, skipped


def _summary(text: str | None, limit: int = 500) -> str | None:
    if text is None:
        return None
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def delete_trace(session: AsyncSession, run_id: str, trace_id: str) -> None:
    await session.execute(delete(Span).where(Span.trace_id == trace_id))
    await session.execute(delete(Exchange).where(Exchange.trace_id == trace_id))
    await session.execute(delete(RecordedValue).where(RecordedValue.trace_id == trace_id))
    await session.execute(delete(Run).where(Run.id == run_id))


async def complete_run(
    store: Store, profiles: ProfileRegistry, trace_id: str, *, keep_unmatched: bool = False
) -> CompletionResult:
    run = await store.reader.run_by_trace(trace_id)
    if run is None or run.status != "open":
        return CompletionResult(trace_id, run.id if run else None, "skipped")
    ctx = await load_run_context(store, run, with_steps=False)
    profile = profiles.find(run.profile) or profiles.match(ctx.spans)
    if profile is None and not keep_unmatched:
        run_id = run.id

        async def drop(session: AsyncSession) -> None:
            await delete_trace(session, run_id, trace_id)

        await store.write(drop)
        log.info("deleted trace %s: no profile matches it", trace_id)
        return CompletionResult(trace_id, run_id, "deleted")

    from blackbox.profiles.base import Profile

    active = profile or Profile()
    steps = build_steps(ctx, active)
    ctx.steps = steps
    output = active.read_output(ctx)
    ending = active.ending(ctx, output)
    if ending is None and not ctx.spans and not ctx.exchanges and run.tags.get("start_error"):
        ending = "start_failed"
    output_blob = None
    if run.output_blob is None and output is not None:
        raw = output.encode() if isinstance(output, str) else json.dumps(output, ensure_ascii=False).encode()
        output_blob = PreparedBlob.of(raw, "text/plain" if isinstance(output, str) else "application/json")
    values = run_summary(ctx, steps)
    values.update(
        profile=active.name or None,
        status="complete",
        ending=ending,
        input_text=_summary(active.input_text(ctx)) or run.input_text,
        output_text=_summary(active.output_text(output)),
        replayable=bool(ctx.exchanges),
    )
    run_id = run.id

    async def write(session: AsyncSession) -> bool:
        current = (await session.execute(select(Run.status).where(Run.id == run_id))).scalar_one_or_none()
        if current != "open":
            return False
        if output_blob is not None:
            values["output_blob"] = await insert_blob(session, output_blob)
        await session.execute(delete(Step).where(Step.run_id == run_id))
        if steps:
            await session.execute(insert(Step), [step.to_row(run_id) for step in steps])
        await session.execute(update(Run).where(Run.id == run_id).values(**values))
        now = now_ms()
        session.add(
            Job(
                id=new_id(),
                kind="run_completed",
                lane="cpu",
                run_id=run_id,
                payload={"profile": values["profile"]},
                available_ms=now,
                created_ms=now,
                updated_ms=now,
            )
        )
        return True

    if not await store.write(write):
        return CompletionResult(trace_id, run_id, "skipped")
    return CompletionResult(trace_id, run_id, "complete")


def run_summary(ctx: RunContext, steps: list[Any]) -> dict[str, Any]:
    starts = [span.start_ns for span in ctx.spans]
    ends = [span.end_ns for span in ctx.spans]
    root = ctx.root
    started_ms = ctx.run.started_ms
    if starts:
        started_ms = min(starts) // 1_000_000 if started_ms is None else min(started_ms, min(starts) // 1_000_000)
    ended_ms = max(ends) // 1_000_000 if ends else None
    for exchange in ctx.exchanges:
        if exchange.row.ended_ms is not None:
            ended_ms = max(ended_ms or 0, exchange.row.ended_ms)
        started_ms = min(started_ms, exchange.row.started_ms) if started_ms is not None else exchange.row.started_ms
    duration = None
    if root is not None and root.end_ns > root.start_ns:
        duration = round(root.duration_ms)
    elif started_ms is not None and ended_ms is not None:
        duration = ended_ms - started_ms
    models = Counter(step.model for step in steps if step.kind == "llm" and step.model)
    return {
        "started_ms": started_ms,
        "ended_ms": ended_ms,
        "duration_ms": duration,
        "step_count": len(steps),
        "input_tokens": sum(step.input_tokens or 0 for step in steps),
        "output_tokens": sum(step.output_tokens or 0 for step in steps),
        "model": models.most_common(1)[0][0] if models else ctx.run.model,
    }
