"""Start one listener per configured upstream inside `blackbox serve`, and report them in `/health`."""

import asyncio
from typing import Any

from blackbox.net import make_client
from blackbox.proxy.core import Proxy
from blackbox.proxy.listener import ListenerApp
from blackbox.server import START_HOOKS, Running, start_server


async def upstream_reachable(target: str) -> bool:
    async with make_client(timeout=1.5) as client:
        try:
            await client.get(target)
        except Exception:
            return False
    return True


async def start_proxy(running: Running) -> None:
    services = running.services
    config = services.settings.proxy
    if not config.enabled or not config.upstreams:
        return
    proxy = Proxy(services, config.upstreams)
    await proxy.seq.preload()
    services.proxy = proxy
    host = services.settings.server.host
    for upstream in config.upstreams:
        server, task, port = await start_server(ListenerApp(proxy, upstream), host, upstream.listen_port, upstream.name)
        proxy.ports[upstream.name] = port
        running.servers.append((f"proxy {upstream.name}: http://{host}:{port} → {upstream.target}", server, task))
    running.stoppers.append(proxy.close)

    async def health() -> dict[str, Any]:
        checks = await asyncio.gather(*(upstream_reachable(u.target) for u in config.upstreams))
        return {
            "proxy": {
                u.name: {"listen": f"http://{host}:{proxy.ports[u.name]}", "target": u.target, "reachable": ok}
                for u, ok in zip(config.upstreams, checks, strict=True)
            }
        }

    services.health_hooks.append(health)


START_HOOKS.append(start_proxy)
