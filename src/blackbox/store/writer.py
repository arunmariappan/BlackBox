"""The single writer: one asyncio task owns the only write connection.

Other components call `await writer.submit(op)`, where `op` is an async function that receives an `AsyncSession`.
Queued operations run in small batches, one transaction per batch, each operation inside its own SAVEPOINT: a
failing operation fails only its own caller, and the rest of its batch still commits.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

log = logging.getLogger(__name__)

type WriteOp[T] = Callable[[AsyncSession], Awaitable[T]]


@dataclass
class _Queued:
    fn: WriteOp[Any]
    future: asyncio.Future[Any]
    result: Any = None
    error: BaseException | None = None
    done: bool = field(default=False)


class StoreWriter:
    def __init__(self, engine: AsyncEngine, *, max_batch: int = 50, max_batch_ms: float = 50) -> None:
        self._engine = engine
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)
        self._queue: asyncio.Queue[_Queued | None] = asyncio.Queue()
        self._max_batch = max_batch
        self._max_batch_s = max_batch_ms / 1000
        self._task: asyncio.Task[None] | None = None
        self.batches = 0  # for tests and /health

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="blackbox-store-writer")

    async def stop(self) -> None:
        """Finish every queued operation, then stop."""
        if self._task is None:
            return
        await self._queue.put(None)
        await self._task
        self._task = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def submit[T](self, fn: WriteOp[T]) -> T:
        if not self.running:
            raise RuntimeError("the store writer is not running")
        future: asyncio.Future[T] = asyncio.get_running_loop().create_future()
        await self._queue.put(_Queued(fn, future))
        return await future

    async def _run(self) -> None:
        stopping = False
        while not stopping:
            first = await self._queue.get()
            if first is None:
                break
            batch = [first]
            started = time.monotonic()
            while len(batch) < self._max_batch and time.monotonic() - started < self._max_batch_s:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    stopping = True
                    break
                batch.append(item)
            await self._execute(batch)
        # Anything queued after the stop sentinel still runs, so no caller waits forever.
        rest: list[_Queued] = []
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if item is not None:
                rest.append(item)
        if rest:
            await self._execute(rest)

    async def _execute(self, batch: list[_Queued]) -> None:
        self.batches += 1
        try:
            async with self._sessions() as session, session.begin():
                for item in batch:
                    if item.future.done():  # the caller gave up (cancelled)
                        item.done = True
                        continue
                    try:
                        async with session.begin_nested():
                            item.result = await item.fn(session)
                    except Exception as exc:
                        item.error = exc
                    item.done = True
        except Exception as exc:
            log.warning("write batch of %d failed to commit (%s); retrying one by one", len(batch), exc)
            for item in batch:
                await self._execute_one(item)
            return
        for item in batch:
            self._settle(item)

    async def _execute_one(self, item: _Queued) -> None:
        item.error = None
        try:
            async with self._sessions() as session, session.begin():
                item.result = await item.fn(session)
        except Exception as exc:
            item.error = exc
        self._settle(item)

    @staticmethod
    def _settle(item: _Queued) -> None:
        if item.future.done():
            return
        if item.error is not None:
            item.future.set_exception(item.error)
        else:
            item.future.set_result(item.result)
