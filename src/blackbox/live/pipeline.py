"""Live scoring: what happens to a run after it completes, in the worker.

`run_completed` (CPU lane) runs the checker and metrics inline, then decides whether trusted judges score the run
(`live.sampling`). If they do, a `judge` job goes to the model lane; after it, or straight away if not, the run is
checked for failure (a failed run gets `describe_failure`, which also assigns its cluster) and a `detect` job
re-evaluates the profile's detectors. Re-clustering and failure descriptions also end in `detect`, for the
new-failure-mode rule.

The model lane waits while any agent call is in flight at the proxy, so judging never slows the agent being measured;
a job that has waited `model_lane_max_wait_seconds` runs anyway, so the lane can't starve.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from blackbox.clusters.jobs import failures_after_complete
from blackbox.live.sampling import plan_judges
from blackbox.live.worker import Handler, JobView, enqueue
from blackbox.store.models import Job, Run
from blackbox.util import now_ms

if TYPE_CHECKING:
    from blackbox.services import Services

log = logging.getLogger(__name__)


async def live_after_complete(services: Services, run: Run) -> None:
    """Completion handler (after the checker and metrics): queue the sampled judges, or go on to failure detection."""
    plan = await plan_judges(services, run)
    if plan.judges:
        payload = {"profile": run.profile, "judges": plan.judges, "calls": plan.calls, "reason": plan.reason}
        await enqueue(services, "judge", run_id=run.id, lane="model", payload=payload)
        return
    await after_judging(services, run)


async def after_judging(services: Services, run: Run) -> None:
    await failures_after_complete(services, run)
    if run.profile:
        await request_detect(services, run.profile)


async def judge_job(services: Services, job: JobView) -> None:
    """Run the sampled trusted judges on one run (a retry costs nothing for verdicts already in the cache)."""
    if job.run_id is None:
        return
    run = await services.store.reader.run(job.run_id)
    if run is None:
        return
    for name in job.payload.get("judges", []):
        judge = services.judges.get(str(name))
        if judge is not None:
            await services.judge_runner.judge_run(judge, run)
    await after_judging(services, run)


async def judge_gave_up(services: Services, job: JobView) -> None:
    """After a judge job's last failed attempt the run still goes on, without its judge scores."""
    if job.run_id is None:
        return
    run = await services.store.reader.run(job.run_id)
    if run is not None:
        await after_judging(services, run)


async def request_detect(services: Services, profile: str) -> None:
    """Queue a `detect` job for `profile` unless one is already waiting."""
    async with services.store.read() as s:
        waiting = (
            await s.execute(
                select(func.count())
                .select_from(Job)
                .where(
                    Job.kind == "detect",
                    Job.status == "queued",
                    func.json_extract(Job.payload, "$.profile") == profile,
                )
            )
        ).scalar_one()
    if not waiting:
        await enqueue(services, "detect", payload={"profile": profile}, lane="cpu")


async def detect_job(services: Services, job: JobView) -> None:
    if services.live is not None:
        transitions = await services.live.engine.evaluate(str(job.payload["profile"]))
        for t in transitions:
            if t.change in ("opened", "resolved"):
                log.info("alert %s %s (%s)", t.alert_id, t.change, t.rule)


def then_detect(handler: Handler) -> Handler:
    """Wrap a job handler whose payload names a profile so it ends with a `detect` request."""

    async def wrapped(services: Services, job: JobView) -> None:
        await handler(services, job)
        profile = job.payload.get("profile")
        if profile:
            await request_detect(services, str(profile))

    return wrapped


class ModelLaneGate:
    """May the model lane take a job now? Yes when no agent call is in flight at the proxy, or when the oldest
    queued model job has waited `max_wait_seconds`."""

    def __init__(
        self, services: Services, *, in_flight: Callable[[], int] | None = None, max_wait_seconds: float | None = None
    ) -> None:
        self.services = services
        self.in_flight = in_flight or services.assembler.in_flight
        self.max_wait_ms = int(
            1000
            * (max_wait_seconds if max_wait_seconds is not None else services.settings.live.model_lane_max_wait_seconds)
        )
        self.forced = 0  # jobs let through after the time limit

    async def __call__(self) -> bool:
        if self.in_flight() == 0:
            return True
        async with self.services.store.read() as s:
            oldest = (
                await s.execute(
                    select(func.min(Job.available_ms)).where(
                        Job.lane == "model", Job.status == "queued", Job.available_ms <= now_ms()
                    )
                )
            ).scalar_one()
        if oldest is not None and now_ms() - int(oldest) >= self.max_wait_ms:
            self.forced += 1
            return True
        return False
