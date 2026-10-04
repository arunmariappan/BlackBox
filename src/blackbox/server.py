"""`blackbox serve`: the web app, the OTLP receiver, the proxy listeners and the worker on one event loop."""

import asyncio
import contextlib
import logging
import signal
import socket
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI

from blackbox.config import Settings
from blackbox.profiles import ProfileRegistry
from blackbox.services import Services
from blackbox.web.app import create_app

log = logging.getLogger(__name__)


class QuietServer(uvicorn.Server):
    """A uvicorn server that leaves signal handling to `serve()`, so several can share one loop."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def bind(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(2048)
    sock.setblocking(False)
    return sock


async def start_server(app: Any, host: str, port: int, name: str) -> tuple[QuietServer, asyncio.Task[None], int]:
    sock = bind(host, port)
    actual_port = sock.getsockname()[1]
    config = uvicorn.Config(app, lifespan="off", access_log=False, log_level="warning", timeout_graceful_shutdown=3)
    server = QuietServer(config)
    task = asyncio.create_task(server.serve(sockets=[sock]), name=f"uvicorn-{name}")
    while not server.started:
        if task.done():
            task.result()
        await asyncio.sleep(0.01)
    return server, task, actual_port


@dataclass
class Running:
    services: Services
    app: FastAPI
    port: int
    servers: list[tuple[str, QuietServer, asyncio.Task[None]]] = field(default_factory=list)
    stoppers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.services.settings.server.host}:{self.port}"

    async def stop(self) -> None:
        for _, server, _ in self.servers:
            server.should_exit = True
        await asyncio.gather(*(task for _, _, task in self.servers), return_exceptions=True)
        for stopper in reversed(self.stoppers):
            await stopper()
        await self.services.close()


# Later phases (proxy, worker) add themselves here: each hook may start listeners and register stoppers.
type StartHook = Callable[[Running], Awaitable[None]]
START_HOOKS: list[StartHook] = []


async def start_blackbox(
    settings: Settings, *, profiles: ProfileRegistry | None = None, migrate: bool = True, hooks: bool = True
) -> Running:
    services = await Services.create(settings, profiles=profiles, migrate=migrate)
    app = create_app(services)
    await services.start()
    server, task, port = await start_server(app, settings.server.host, settings.server.port, "web")
    if settings.server.port == 0:
        settings.server.public_url = settings.server.public_url or f"http://{settings.server.host}:{port}"
    running = Running(services, app, port, [("web", server, task)])
    if hooks:
        for hook in START_HOOKS:
            await hook(running)
    return running


async def serve(settings: Settings, ready: Callable[[Running], None] | None = None) -> None:
    running = await start_blackbox(settings)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))
    if ready is not None:
        ready(running)
    try:
        await stop.wait()
    finally:
        await running.stop()
