"""Where metrics plug in: after each run completes (the worker), in fidelity reports, and as run-list badges."""

from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from blackbox.metrics.framework import MetricInput, compute, compute_and_store
from blackbox.runs.context import RunContext
from blackbox.store.models import Run, Score

if TYPE_CHECKING:
    from blackbox.proxy.sessions import ReplaySession
    from blackbox.services import Services

FLAG_BADGES = {
    "loop": "loop",
    "wrong_tool": "wrong tool",
    "unrecovered_errors": "unrecovered error",
    "policy_violations": "policy",
    "citations_valid": "bad citation",
    "citations_present": "no citation",
    "wasted_rewrite": "wasted rewrite",
    "scope_mismatch": "scope mismatch",
}


async def metrics_after_complete(services: Services, run: Run) -> None:
    await compute_and_store(services.store, services.profiles, run)


async def metric_changes(
    services: Services, session: ReplaySession, source: RunContext, replay: RunContext
) -> dict[str, Any]:
    """Metric values of the source run and its replay, for the ones that differ."""

    async def values(ctx: RunContext) -> dict[str, float | None]:
        async with services.store.read() as s:
            scores = list((await s.execute(select(Score).where(Score.run_id == ctx.run.id))).scalars())
        computed = compute(MetricInput(ctx, services.profiles.find(ctx.run.profile), scores))
        return {value.name: value.value for _, value in computed if value.value is not None}

    a, b = await values(source), await values(replay)
    changes = {
        name: {"source": a.get(name), "replay": b.get(name)}
        for name in sorted(set(a) | set(b))
        if a.get(name) != b.get(name)
    }
    return {"metric_changes": changes}


async def metric_badges(services: Services, run_ids: list[str]) -> dict[str, list[dict[str, str]]]:
    async with services.store.read() as s:
        rows = (
            await s.execute(
                select(Score).where(
                    Score.run_id.in_(run_ids), Score.kind == "metric", Score.name.in_(list(FLAG_BADGES))
                )
            )
        ).scalars()
        badges: dict[str, list[dict[str, str]]] = {}
        for score in rows:
            if score.details.get("flag"):
                badges.setdefault(score.run_id, []).append({"css": "warn", "text": FLAG_BADGES[score.name]})
    return badges
