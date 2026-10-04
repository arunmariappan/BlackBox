"""A profile's signals, read from its recent runs and their scores.

Rates (`True` is the good outcome): the pass rate (the checker, else all trusted judges passing), each trusted
judge's fail rate, each metric flag's rate and each ending's share. Values: latency, tokens and steps per run. Only
runs from the watched sources (`live.sources`: live and traffic by default) count; replays and suites never do.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from sqlalchemy import func, select

from blackbox.judges.calibrate import trusted_versions
from blackbox.live.detectors import ClusterActivity
from blackbox.store.models import Alert, Cluster, Failure, Marker, Run, Score

if TYPE_CHECKING:
    from blackbox.services import Services

HISTORY_RUNS = 600


@dataclass
class RunPoint:
    run_id: str
    t_ms: int
    ending: str | None
    duration_ms: int | None
    tokens: int
    steps: int
    checker: str | None = None
    judges: dict[str, str] = field(default_factory=dict)  # trusted judge → pass or fail
    flags: dict[str, bool] = field(default_factory=dict)  # metric with a flag → flagged


@dataclass
class Signal:
    key: str  # e.g. pass_rate, judge:pp_answer_quality, flag:loop, ending:out_of_scope, latency
    label: str
    kind: Literal["rate", "value"]
    points: list[tuple[RunPoint, float]]  # chronological; for rates 1.0 is the good outcome

    @property
    def direction(self) -> str:
        return "dropped" if self.key == "pass_rate" else "rose"


async def load_points(services: Services, profile: str, limit: int = HISTORY_RUNS) -> list[RunPoint]:
    sources = services.settings.live.sources
    t = func.coalesce(Run.ended_ms, Run.updated_ms)
    async with services.store.read() as s:
        runs = list(
            (
                await s.execute(
                    select(Run)
                    .where(Run.profile == profile, Run.status == "complete", Run.source.in_(sources))
                    .order_by(t.desc(), Run.id.desc())
                    .limit(limit)
                )
            ).scalars()
        )
        runs.reverse()
        ids = [run.id for run in runs]
        scores = list(
            (
                await s.execute(
                    select(Score).where(Score.run_id.in_(ids), Score.kind.in_(("checker", "judge", "metric")))
                )
            ).scalars()
        )
    trusted = await trusted_versions(services.store)
    by_run: dict[str, list[Score]] = defaultdict(list)
    for score in scores:
        by_run[score.run_id].append(score)
    points = []
    for run in runs:
        point = RunPoint(
            run.id,
            run.ended_ms or run.updated_ms,
            run.ending,
            run.duration_ms,
            run.input_tokens + run.output_tokens,
            run.step_count,
        )
        for score in by_run[run.id]:
            if score.kind == "checker":
                point.checker = score.label
            elif score.kind == "judge" and score.label in ("pass", "fail"):
                if score.version in trusted.get(score.name, set()):
                    point.judges[score.name] = score.label
            elif score.kind == "metric" and "flag" in score.details:
                point.flags[score.name] = bool(score.details.get("flag"))
        points.append(point)
    return points


def signals(points: Sequence[RunPoint]) -> list[Signal]:
    out: list[Signal] = []
    passes = []
    for p in points:
        if p.checker in ("pass", "fail"):
            passes.append((p, 1.0 if p.checker == "pass" else 0.0))
        elif p.judges:
            passes.append((p, 1.0 if all(label == "pass" for label in p.judges.values()) else 0.0))
    if passes:
        out.append(Signal("pass_rate", "pass rate", "rate", passes))
    judge_names = sorted({name for p in points for name in p.judges})
    for name in judge_names:
        rows = [(p, 0.0 if p.judges[name] == "fail" else 1.0) for p in points if name in p.judges]
        out.append(Signal(f"judge:{name}", f"{name} fail rate", "rate", rows))
    for name in sorted({name for p in points for name in p.flags}):
        rows = [(p, 0.0 if p.flags[name] else 1.0) for p in points if name in p.flags]
        out.append(Signal(f"flag:{name}", f"{name} flag rate", "rate", rows))
    for ending in sorted({p.ending for p in points if p.ending}):
        rows = [(p, 0.0 if p.ending == ending else 1.0) for p in points if p.ending]
        out.append(Signal(f"ending:{ending}", f"share of ending {ending}", "rate", rows))
    latency = [(p, float(p.duration_ms)) for p in points if p.duration_ms is not None]
    if latency:
        out.append(Signal("latency", "latency (ms)", "value", latency))
    out.append(Signal("tokens", "tokens per run", "value", [(p, float(p.tokens)) for p in points]))
    out.append(Signal("steps", "steps per run", "value", [(p, float(p.steps)) for p in points]))
    return out


async def alert_periods(services: Services, profile: str, now: int) -> list[tuple[int, int]]:
    async with services.store.read() as s:
        rows = (await s.execute(select(Alert.opened_ms, Alert.closed_ms).where(Alert.profile == profile))).all()
    return [(opened, closed or now) for opened, closed in rows]


def in_periods(t_ms: int, periods: Sequence[tuple[int, int]]) -> bool:
    return any(start <= t_ms <= end for start, end in periods)


async def cluster_activity(services: Services, profile: str, since_ms: int) -> list[ClusterActivity]:
    async with services.store.read() as s:
        clusters = list(
            (await s.execute(select(Cluster).where(Cluster.profile == profile, Cluster.status != "retired"))).scalars()
        )
        joined = (
            await s.execute(
                select(Failure.cluster_id, Failure.run_id)
                .where(Failure.profile == profile, Failure.cluster_id.is_not(None), Failure.created_ms >= since_ms)
                .order_by(Failure.created_ms)
            )
        ).all()
    members: dict[str, list[str]] = defaultdict(list)
    for cluster_id, run_id in joined:
        members[str(cluster_id)].append(run_id)
    return [ClusterActivity(c.id, c.title, c.created_ms, members.get(c.id, [])) for c in clusters]


async def failures_in(services: Services, run_ids: Sequence[str]) -> list[dict[str, object]]:
    """The failure clusters of `run_ids`, largest first, with example runs."""
    if not run_ids:
        return []
    async with services.store.read() as s:
        rows = (
            await s.execute(
                select(Failure.cluster_id, Failure.run_id, Cluster.title)
                .join(Cluster, Cluster.id == Failure.cluster_id)
                .where(Failure.run_id.in_(list(run_ids)))
            )
        ).all()
    grouped: dict[str, dict[str, object]] = {}
    for cluster_id, run_id, title in rows:
        entry = grouped.setdefault(str(cluster_id), {"cluster_id": cluster_id, "title": title, "runs": []})
        runs = entry["runs"]
        assert isinstance(runs, list)
        runs.append(run_id)
    return sorted(grouped.values(), key=lambda g: -len(g["runs"]))  # type: ignore[arg-type]


async def markers_between(services: Services, profile: str, start_ms: int, end_ms: int) -> list[Marker]:
    async with services.store.read() as s:
        return list(
            (
                await s.execute(
                    select(Marker)
                    .where(Marker.profile == profile, Marker.created_ms >= start_ms, Marker.created_ms <= end_ms)
                    .order_by(Marker.created_ms)
                )
            ).scalars()
        )
