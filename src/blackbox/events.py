"""An in-process event bus: components publish, the SSE endpoint and tests subscribe."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    type: str  # run.created, run.updated, run.completed, session.updated, alert.opened, ...
    data: dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self, max_queue: int = 1000) -> None:
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._max_queue = max_queue

    def publish(self, type_: str, **data: Any) -> None:
        event = Event(type_, data)
        for queue in list(self._subscribers):
            if queue.qsize() < self._max_queue:  # a slow subscriber loses events rather than blocking publishers
                queue.put_nowait(event)

    @contextlib.asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[Event]]:
        queue: asyncio.Queue[Event] = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)
