"""The parts of a running BlackBox, created once by `blackbox serve` (or a test) and shared by everything."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from blackbox.config import Settings
from blackbox.events import EventBus
from blackbox.judges.framework import Judge, load_judges
from blackbox.judges.runner import JudgeRunner
from blackbox.llm.client import OllamaJSON
from blackbox.net import make_client
from blackbox.profiles import ProfileRegistry, default_registry
from blackbox.runs.assembler import RunAssembler
from blackbox.store import Store

if TYPE_CHECKING:
    from blackbox.proxy.core import Proxy

log = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    store: Store
    bus: EventBus
    profiles: ProfileRegistry
    assembler: RunAssembler
    http: httpx.AsyncClient
    llm: OllamaJSON
    judges: dict[str, Judge]
    judge_runner: JudgeRunner
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    health_hooks: list[Callable[[], Awaitable[dict[str, Any]]]] = field(default_factory=list)
    proxy: Proxy | None = None
    # Each returns extra fields for a replay's fidelity report (judges in phase 5, metrics in phase 7).
    report_hooks: list[Callable[..., Awaitable[dict[str, Any]]]] = field(default_factory=list)
    # Each returns run id → extra badges for the runs list (metric flags in phase 7).
    badge_hooks: list[Callable[..., Awaitable[dict[str, list[dict[str, str]]]]]] = field(default_factory=list)

    @classmethod
    async def create(
        cls, settings: Settings, *, profiles: ProfileRegistry | None = None, migrate: bool = True
    ) -> Services:
        store = await Store.open(settings.store.path, migrate=migrate)
        bus = EventBus()
        registry = profiles or default_registry(settings.profiles)
        assembler = RunAssembler(store, registry, bus, settings.runs)
        llm = OllamaJSON(settings.ollama)
        services = cls(
            settings=settings,
            store=store,
            bus=bus,
            profiles=registry,
            assembler=assembler,
            http=make_client(timeout=600),
            llm=llm,
            judges=load_judges(settings.ollama.model),
            judge_runner=JudgeRunner(store, llm),
        )
        from blackbox.judges.replay import judge_report

        services.report_hooks.append(judge_report)
        return services

    async def start(self) -> None:
        await self.assembler.start()

    async def close(self) -> None:
        for task in list(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.assembler.stop()
        await self.http.aclose()
        await self.llm.close()
        await self.store.close()

    def spawn(self, coro: Coroutine[Any, Any, Any], *, name: str | None = None) -> asyncio.Task[Any]:
        """Run `coro` in the background; errors are logged, never lost."""
        task = asyncio.create_task(coro, name=name)
        self.tasks.add(task)

        def done(t: asyncio.Task[Any]) -> None:
            self.tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                log.error("background task %s failed", t.get_name(), exc_info=t.exception())

        task.add_done_callback(done)
        return task
