"""Which completed runs the trusted judges score live.

A run is sampled by a hash of its trace id, so the decision can be reproduced. Runs with a metric flag or a failure
ending are always judged when `always_judge_flagged` is on. Every judge job counts its planned model calls against
the profile's hourly budget; a run that would go over it isn't judged (its metrics and checker still run).
"""

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from blackbox.judges.calibrate import trusted_versions
from blackbox.judges.framework import Judge
from blackbox.store.models import Job, Run, Score
from blackbox.util import now_ms

if TYPE_CHECKING:
    from blackbox.services import Services

HOUR_MS = 3_600_000


def sample_point(trace_id: str) -> float:
    """A number in [0, 1) fixed by the trace id."""
    digest = hashlib.sha256(trace_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def is_sampled(trace_id: str, rate: float) -> bool:
    return sample_point(trace_id) < rate


@dataclass
class JudgePlan:
    judges: list[str] = field(default_factory=list)
    calls: int = 0
    reason: str = ""  # sampled, flagged, not_sampled, no_trusted_judges, over_budget


async def trusted_judges_for(services: Services, run: Run) -> list[Judge]:
    trusted = await trusted_versions(services.store)
    return [
        judge
        for name, judge in sorted(services.judges.items())
        if judge.version in trusted.get(name, set()) and judge.applies(run)
    ]


async def is_flagged(services: Services, run: Run) -> bool:
    profile = services.profiles.find(run.profile)
    if profile is not None and run.ending in profile.failure_endings:
        return True
    async with services.store.read() as s:
        scores = (await s.execute(select(Score).where(Score.run_id == run.id, Score.kind == "metric"))).scalars()
        return any(score.details.get("flag") for score in scores)


async def calls_this_hour(services: Services, profile: str) -> int:
    """Model calls planned by judge jobs queued for `profile` in the last hour."""
    async with services.store.read() as s:
        total = (
            await s.execute(
                select(func.coalesce(func.sum(func.json_extract(Job.payload, "$.calls")), 0)).where(
                    Job.kind == "judge",
                    Job.created_ms >= now_ms() - HOUR_MS,
                    func.json_extract(Job.payload, "$.profile") == profile,
                )
            )
        ).scalar_one()
    return int(total or 0)


async def plan_judges(services: Services, run: Run) -> JudgePlan:
    profile = run.profile or ""
    config = services.settings.live.sampling_for(profile)
    judges = await trusted_judges_for(services, run)
    if not judges:
        return JudgePlan(reason="no_trusted_judges")
    if config.always_judge_flagged and await is_flagged(services, run):
        reason = "flagged"
    elif is_sampled(run.trace_id, config.judge_rate):
        reason = "sampled"
    else:
        return JudgePlan(reason="not_sampled")
    calls = sum(judge.samples for judge in judges)
    if await calls_this_hour(services, profile) + calls > config.max_judge_calls_per_hour:
        return JudgePlan(reason="over_budget")
    return JudgePlan([judge.name for judge in judges], calls, reason)
