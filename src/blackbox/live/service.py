"""The live parts of a running BlackBox: the alert engine, live patches and markers, and the header banners."""

import html
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.live.alerts import AlertEngine
from blackbox.live.patches import LivePatches
from blackbox.store.models import Marker
from blackbox.util import new_id, now_ms

if TYPE_CHECKING:
    from blackbox.services import Services


class LiveService:
    def __init__(self, services: Services) -> None:
        self.services = services
        self.engine = AlertEngine(services)
        self.patches = LivePatches(services)

    async def start(self) -> None:
        await self.patches.load()
        await self.engine.refresh_banners()
        self.patches.start()
        if self.services.proxy is not None:
            self.services.proxy.request_hooks.append(self.patches.hook)

    async def stop(self) -> None:
        self.patches.stop()

    async def add_marker(self, profile: str, text: str) -> Marker:
        marker = Marker(id=new_id(), profile=profile, text=text, created_ms=now_ms())

        async def op(session: AsyncSession) -> None:
            session.add(marker)

        await self.services.store.write(op)
        self.services.bus.publish("marker.added", marker_id=marker.id, profile=profile)
        return marker

    def banners(self) -> list[dict[str, Any]]:
        out = []
        for alert in self.engine.open_alerts:
            out.append(
                {
                    "kind": "alert",
                    "html": f'Alert: <a href="/alerts/{alert["id"]}">{html.escape(alert["summary"])}</a>',
                }
            )
        now = now_ms()
        for patch in self.patches.active():
            minutes = max(0, round((patch.expires_ms - now) / 60_000))
            out.append(
                {
                    "kind": "patch",
                    "html": (
                        f'Live patch <a href="/live">{html.escape(patch.name)}</a> is changing live traffic '
                        f"({patch.hits} requests so far; expires in {minutes} min)"
                    ),
                }
            )
        return out
