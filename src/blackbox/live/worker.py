"""The worker: durable jobs from the `jobs` table, so a restart loses nothing.

Each job kind has a handler. A failing job is retried with backoff (2, 4 s) up to three attempts, then left `failed`
with its last error, listed in the UI (`/jobs`), and its kind's give-up handler runs, if it has one. Jobs still
`running` after a restart are put back in the queue. A lane can have a gate that decides whether it may take a job now
(the model lane waits for agent calls in flight, see `live.pipeline`).
"""

import asyncio
import contextlib
import logging
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.store.models import Job
from blackbox.util import new_id, now_ms

if TYPE_CHECKING:
    from blackbox.services import Services

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class JobView:
    id: str
    kind: str
    lane: str
    run_id: str | None
    payload: dict[str, Any]
    attempts: int


type Handler = Callable[["Services", JobView], Awaitable[None]]


def new_job(
    kind: str, *, run_id: str | None = None, payload: dict[str, Any] | None = None, lane: str = "cpu", delay_ms: int = 0
) -> Job:
    now = now_ms()
    return Job(
        id=new_id(),
        kind=kind,
        lane=lane,
        run_id=run_id,
        payload=payload or {},
        status="queued",
        attempts=0,
        available_ms=now + delay_ms,
        created_ms=now,
        updated_ms=now,
    )


async def enqueue(
    services: Services,
    kind: str,
    *,
    run_id: str | None = None,
    payload: dict[str, Any] | None = None,
    lane: str = "cpu",
) -> str:
    job = new_job(kind, run_id=run_id, payload=payload, lane=lane)

    async def op(session: AsyncSession) -> str:
        session.add(job)
        return job.id

    job_id = await services.store.write(op)
    if services.worker is not None:
        services.worker.wake()
    return job_id


class Worker:
    def __init__(
        self,
        services: Services,
        *,
        lanes: tuple[str, ...] = ("cpu", "model"),
        poll_seconds: float = 0.5,
        backoff_ms: int = 1000,
    ) -> None:
        self.services = services
        self.handlers: dict[str, Handler] = {}
        self.lanes = lanes
        self.poll_seconds = poll_seconds
        self._tasks: list[asyncio.Task[None]] = []
        self._wake = asyncio.Event()
        self.gates: dict[str, Callable[[], Awaitable[bool]]] = {}  # lane → "may a job run now?"
        self.on_give_up: dict[str, Handler] = {}  # kind → called once a job of that kind has failed for good
        self.backoff_ms = backoff_ms
        self.processed = 0

    def register(self, kind: str, handler: Handler) -> None:
        self.handlers[kind] = handler

    def wake(self) -> None:
        self._wake.set()

    async def start(self) -> None:
        await self._requeue_running()
        for lane in self.lanes:
            self._tasks.append(asyncio.create_task(self._loop(lane), name=f"blackbox-worker-{lane}"))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []

    async def _requeue_running(self) -> None:
        async def op(session: AsyncSession) -> None:
            await session.execute(
                update(Job).where(Job.status == "running").values(status="queued", updated_ms=now_ms())
            )

        await self.services.store.write(op)

    async def _loop(self, lane: str) -> None:
        while True:
            try:
                gate = self.gates.get(lane)
                if gate is not None and not await gate():
                    await self._sleep()
                    continue
                job = await self.claim(lane)
                if job is None:
                    await self._sleep()
                    continue
                await self.run(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("worker lane %s failed; continuing", lane)
                await asyncio.sleep(self.poll_seconds)

    async def _sleep(self) -> None:
        self._wake.clear()
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(self.poll_seconds):
                await self._wake.wait()

    async def claim(self, lane: str) -> JobView | None:
        kinds = list(self.handlers)
        now = now_ms()

        async def op(session: AsyncSession) -> JobView | None:
            job = (
                await session.execute(
                    select(Job)
                    .where(Job.status == "queued", Job.lane == lane, Job.available_ms <= now, Job.kind.in_(kinds))
                    .order_by(Job.available_ms, Job.created_ms)
                    .limit(1)
                )
            ).scalar_one_or_none()
            if job is None:
                return None
            job.status, job.attempts, job.updated_ms = "running", job.attempts + 1, now
            return JobView(job.id, job.kind, job.lane, job.run_id, dict(job.payload), job.attempts)

        return await self.services.store.write(op)

    async def run(self, job: JobView) -> None:
        handler = self.handlers[job.kind]
        try:
            await handler(self.services, job)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}"[-4000:]
            failed = job.attempts >= MAX_ATTEMPTS
            delay = 0 if failed else 2**job.attempts * self.backoff_ms
            log.warning("job %s (%s) failed, attempt %d: %s", job.id, job.kind, job.attempts, exc)

            async def op(session: AsyncSession) -> None:
                await session.execute(
                    update(Job)
                    .where(Job.id == job.id)
                    .values(
                        status="failed" if failed else "queued",
                        last_error=error,
                        available_ms=now_ms() + delay,
                        updated_ms=now_ms(),
                    )
                )

            await self.services.store.write(op)
            self.services.bus.publish("job.failed" if failed else "job.retry", job_id=job.id, kind=job.kind)
            give_up = self.on_give_up.get(job.kind)
            if failed and give_up is not None:
                try:
                    await give_up(self.services, job)
                except Exception:
                    log.exception("give-up handler for job %s (%s) failed", job.id, job.kind)
            return

        async def done(session: AsyncSession) -> None:
            await session.execute(update(Job).where(Job.id == job.id).values(status="done", updated_ms=now_ms()))

        await self.services.store.write(done)
        self.processed += 1
        self.services.bus.publish("job.done", job_id=job.id, kind=job.kind, run_id=job.run_id)


async def run_completed(services: Services, job: JobView) -> None:
    """Fan out a completed run's work: every registered completion handler, in order."""
    if job.run_id is None:
        return
    run = await services.store.reader.run(job.run_id)
    if run is None or run.status != "complete":
        return
    for handler in services.completion_handlers:
        await handler(services, run)


async def profile_after_complete(services: Services, run: Any) -> None:
    from blackbox.runs.context import load_run_context

    profile = services.profiles.find(run.profile)
    if profile is None:
        return
    ctx = await load_run_context(services.store, run)
    await profile.after_complete(services, run, ctx)
