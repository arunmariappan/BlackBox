"""Live patches: `content_regex` request patches (phase 4) applied at the proxy to live traffic, with no session.

A live patch is the deliberate break used to test alerts. It expires after its minutes, so it can't be forgotten;
while one is active every page says so in its header, and every exchange it changed is marked `patched`. Replay
sessions never see live patches (they bring their own).
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.replay.patches import Patch, apply_patches, load_patches
from blackbox.store.models import LivePatch
from blackbox.util import new_id, now_ms

if TYPE_CHECKING:
    from blackbox.proxy.core import IncomingRequest, Recording
    from blackbox.services import Services

log = logging.getLogger(__name__)


class LivePatchError(ValueError):
    pass


def parse_live_patches(text: str) -> list[Patch]:
    """Patches from a patch file's text (`patches: [...]`, a list, or one patch), checked for live use."""
    try:
        data = yaml.safe_load(text)
        if isinstance(data, dict) and "patches" not in data:
            data = [data]
        patches = load_patches(data)
    except yaml.YAMLError as exc:
        raise LivePatchError(f"not YAML: {exc}") from None
    except ValidationError as exc:
        raise LivePatchError(str(exc)) from None
    if not patches:
        raise LivePatchError("expected `patches:` with at least one patch")
    for patch in patches:
        if patch.match.tape_node is not None or patch.match.step is not None:
            raise LivePatchError(f"patch {patch.name}: tape_node and step need a replay; live patches match on content")
        if not patch.match.content_regex:
            raise LivePatchError(f"patch {patch.name}: a live patch needs match.content_regex")
    return patches


@dataclass
class ActivePatch:
    id: str
    name: str
    patch: Patch
    created_ms: int
    expires_ms: int
    hits: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "spec": self.patch.model_dump(mode="json", by_alias=True, exclude_unset=True),
            "created_ms": self.created_ms,
            "expires_ms": self.expires_ms,
            "hits": self.hits,
            "status": "active",
        }


class LivePatches:
    def __init__(self, services: Services, *, clock: Callable[[], int] = now_ms) -> None:
        self.services = services
        self.clock = clock
        self._active: dict[str, ActivePatch] = {}
        self._task: asyncio.Task[None] | None = None

    async def load(self) -> None:
        """Expire what ran out while BlackBox was down, and keep the rest active."""
        await self.expire_due()
        async with self.services.store.read() as s:
            rows = (await s.execute(select(LivePatch).where(LivePatch.status == "active"))).scalars().all()
        for row in rows:
            self._active[row.id] = ActivePatch(
                row.id, row.name, Patch.model_validate(row.spec), row.created_ms, row.expires_ms, row.hits
            )

    def active(self) -> list[ActivePatch]:
        now = self.clock()
        return [p for p in self._active.values() if p.expires_ms > now]

    async def add(self, patches: list[Patch], minutes: float) -> list[ActivePatch]:
        now = self.clock()
        added = [ActivePatch(new_id(), p.name, p, now, now + int(minutes * 60_000)) for p in patches]

        async def op(session: AsyncSession) -> None:
            for patch in added:
                session.add(
                    LivePatch(
                        id=patch.id,
                        name=patch.name,
                        spec=patch.patch.model_dump(mode="json", by_alias=True, exclude_unset=True),
                        status="active",
                        hits=0,
                        created_ms=patch.created_ms,
                        expires_ms=patch.expires_ms,
                    )
                )

        await self.services.store.write(op)
        for patch in added:
            self._active[patch.id] = patch
            log.warning("live patch %s active until %s", patch.name, patch.expires_ms)
        self.services.bus.publish("patch.added", ids=[p.id for p in added])
        return added

    async def remove(self, ref: str) -> list[str]:
        """Remove the active patches whose id or name is `ref`."""
        ids = [p.id for p in self._active.values() if ref in (p.id, p.name)]
        await self._end(ids, "removed")
        return ids

    async def expire_due(self) -> list[str]:
        now = self.clock()
        async with self.services.store.read() as s:
            ids = list(
                (
                    await s.execute(
                        select(LivePatch.id).where(LivePatch.status == "active", LivePatch.expires_ms <= now)
                    )
                ).scalars()
            )
        await self._end(ids, "expired")
        return ids

    async def _end(self, ids: list[str], status: str) -> None:
        if not ids:
            return
        now = self.clock()

        async def op(session: AsyncSession) -> None:
            await session.execute(
                update(LivePatch)
                .where(LivePatch.id.in_(ids), LivePatch.status == "active")
                .values(status=status, ended_ms=now)
            )

        await self.services.store.write(op)
        for patch_id in ids:
            self._active.pop(patch_id, None)
        self.services.bus.publish(f"patch.{status}", ids=ids)

    async def hook(self, req: IncomingRequest, rec: Recording) -> None:
        """Proxy request hook: patch a live request whose content matches."""
        if rec.session_id is not None:
            return
        active = self.active()
        if not active:
            return
        body, applied = apply_patches([p.patch for p in active], rec.upstream, req.body)
        if not applied:
            return
        rec.sent_request_body = body
        rec.served_from = "patched"
        hit = [p for p in active if p.patch.name in applied]
        for patch in hit:
            patch.hits += 1
        self.services.spawn(self._count([p.id for p in hit]), name="live-patch-hits")

    async def _count(self, ids: list[str]) -> None:
        async def op(session: AsyncSession) -> None:
            await session.execute(update(LivePatch).where(LivePatch.id.in_(ids)).values(hits=LivePatch.hits + 1))

        await self.services.store.write(op)

    def start(self, every_seconds: float = 5.0) -> None:
        async def loop() -> None:
            while True:
                await asyncio.sleep(every_seconds)
                with contextlib.suppress(Exception):
                    await self.expire_due()

        self._task = self.services.spawn(loop(), name="live-patch-expiry")

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
