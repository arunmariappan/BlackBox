"""Failure jobs for the worker: describe a failed run (model lane), re-cluster a profile, and the nightly check."""

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from blackbox.live.worker import JobView, enqueue
from blackbox.store.models import ClusterSpace, Failure, Job, Run
from blackbox.util import now_ms

if TYPE_CHECKING:
    from blackbox.services import Services

log = logging.getLogger(__name__)

NIGHTLY_MS = 24 * 3_600_000


async def failures_after_complete(services: Services, run: Run) -> None:
    """A failed run gets a `describe_failure` job on the model lane; a run that no longer fails loses its row."""
    clusters = services.clusters
    if clusters is None:
        return
    if await clusters.is_failed(run):
        await enqueue(services, "describe_failure", run_id=run.id, lane="model", payload={"profile": run.profile})
    else:
        await clusters.record(run)  # clears a stale failure row, no model call


async def describe_failure(services: Services, job: JobView) -> None:
    clusters = services.clusters
    if clusters is None or job.run_id is None:
        return
    run = await services.store.reader.run(job.run_id)
    if run is None:
        return
    recorded = await clusters.record(run)
    profile = run.profile or ""
    waiting = recorded.failed and recorded.cluster_id is None and await clusters.unclustered(profile)
    if waiting and waiting >= services.settings.clusters.recluster_after:
        await enqueue_recluster(services, profile)


async def enqueue_recluster(services: Services, profile: str) -> None:
    async with services.store.read() as s:
        waiting = (
            await s.execute(
                select(func.count())
                .select_from(Job)
                .where(
                    Job.kind == "recluster",
                    Job.status.in_(("queued", "running")),
                    func.json_extract(Job.payload, "$.profile") == profile,
                )
            )
        ).scalar_one()
    if not waiting:
        await enqueue(services, "recluster", payload={"profile": profile}, lane="model")


async def recluster(services: Services, job: JobView) -> None:
    if services.clusters is not None:
        summary = await services.clusters.recluster(str(job.payload["profile"]))
        log.info("re-clustered %s: %s", job.payload["profile"], summary)


async def nightly(services: Services, every_seconds: float = 3600) -> None:
    """Every hour: re-cluster any profile whose clustering is more than a day old and has failures."""
    while True:
        with contextlib.suppress(Exception):
            async with services.store.read() as s:
                profiles = set((await s.execute(select(Failure.profile).distinct())).scalars())
                spaces = {sp.profile: sp for sp in (await s.execute(select(ClusterSpace))).scalars()}
            for profile in profiles:
                space = spaces.get(profile)
                if space is None or now_ms() - space.clustered_ms > NIGHTLY_MS:
                    await enqueue_recluster(services, profile)
        await asyncio.sleep(every_seconds)
