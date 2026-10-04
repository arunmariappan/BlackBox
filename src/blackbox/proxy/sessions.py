"""Replay sessions at the proxy: answer a session's calls from its source run's tape, or send them live.

| Mode | Matched request | First divergence | After going live |
|---|---|---|---|
| `exact` | from the tape | 409 (with `lenient`: the next unused tape exchange on that path, flagged) | never |
| `fork` | from the tape while the matched step is before N; step N or later goes live | goes live | live |
| `auto_fork` | from the tape | goes live | live |

Once a session goes live it stays live: the tape's later steps can no longer be trusted to fit.
"""

import asyncio
import json
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from blackbox.proxy.core import Attribution, IncomingRequest, Outcome, Proxy, Recording, json_outcome
from blackbox.proxy.http import HeaderList, is_hop_by_hop
from blackbox.proxy.matching import Normaliser, is_llm_path, match_key, request_diff
from blackbox.replay.patches import LineUp, Patch, apply_patches
from blackbox.runs.context import ExchangeData
from blackbox.util import now_ms

if TYPE_CHECKING:
    from blackbox.profiles.base import Profile
    from blackbox.services import Services

log = logging.getLogger(__name__)

MODES = ("exact", "fork", "auto_fork")


@dataclass
class TapeEntry:
    exchange: ExchangeData
    step: int
    node: str | None
    key: str
    used: bool = False

    @property
    def upstream(self) -> str:
        return self.exchange.row.upstream


@dataclass
class SessionSpec:
    source_run_id: str
    mode: str = "exact"
    fork_step: int | None = None
    model: str | None = None
    patches: list[Patch] = field(default_factory=list)
    speed: float = 0.0
    lenient: bool = False
    aliasing: bool = True  # stateful upstreams: map ids created on the tape to ids the live sandbox created
    sync_forward: bool = True  # stateful upstreams: replay state-changing calls into the session's own sandbox

    def overrides(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "patches": [p.model_dump(by_alias=True, exclude_none=True) for p in self.patches],
            "speed": self.speed,
            "lenient": self.lenient,
            "aliasing": self.aliasing,
            "sync_forward": self.sync_forward,
        }


@dataclass
class Served:
    """How one request of the session was answered (for the fidelity report)."""

    seq: int
    upstream: str
    method: str
    path: str
    served: str  # tape, live, patched, blocked, lenient
    tape_step: int | None
    exchange_id: str
    patches: list[str] = field(default_factory=list)
    model_override: str | None = None


class ReplaySession:
    def __init__(
        self,
        session_id: str,
        trace_id: str,
        spec: SessionSpec,
        profile: Profile,
        tape: list[TapeEntry],
        *,
        ttl_seconds: float,
    ) -> None:
        self.id = session_id
        self.trace_id = trace_id
        self.spec = spec
        self.profile = profile
        self.tape: dict[str, list[TapeEntry]] = {}
        for entry in sorted(tape, key=lambda e: e.step):
            self.tape.setdefault(entry.upstream, []).append(entry)
        self.live = False
        self.went_live_at: int | None = None  # the seq of the first live request
        self.first_divergence: dict[str, Any] | None = None
        self.served: list[Served] = []
        self.expires_at = time.monotonic() + ttl_seconds
        self.lock = asyncio.Lock()
        self.state: dict[str, Any] = {}  # per-session data for stateful upstreams (sandbox id, aliases)
        self.position: dict[str, int] = {}  # requests seen per upstream

    # Matching -------------------------------------------------------------------------------------------------------

    def normalisers(self, upstream: str) -> list[Normaliser]:
        return self.profile.normalisers(upstream)

    def key(self, req: IncomingRequest, body: bytes) -> str:
        return match_key(req.method, req.path, req.query, body, self.normalisers(req.upstream.name))

    def candidate(self, upstream: str, key: str) -> TapeEntry | None:
        for entry in self.tape.get(upstream, []):
            if not entry.used and entry.key == key:
                return entry
        return None

    def next_unused(self, upstream: str, path: str | None = None) -> TapeEntry | None:
        entries = self.tape.get(upstream, [])
        for entry in entries:
            if not entry.used and (path is None or entry.exchange.row.path == path):
                return entry
        if path is not None:
            return self.next_unused(upstream)
        return None

    def expired(self) -> bool:
        return time.monotonic() > self.expires_at

    # Handling -------------------------------------------------------------------------------------------------------

    async def handle(self, proxy: Proxy, req: IncomingRequest, attribution: Attribution) -> Outcome:
        rec = await proxy.begin(req, attribution, session_id=self.id)
        upstream = req.upstream.name
        async with self.lock:
            position = self.position.get(upstream, 0)
            self.position[upstream] = position + 1
            original_key = self.key(req, req.body)
            candidate = self.candidate(upstream, original_key)
            if self.live:
                # Once live, the k-th request on an upstream lines up with the k-th tape step on it.
                entries = self.tape.get(upstream, [])
                lined_up = entries[position] if position < len(entries) else None
            else:
                lined_up = candidate or self.next_unused(upstream, req.path)
            lineup = LineUp(lined_up.step, lined_up.node) if lined_up is not None else None
            body, applied = apply_patches(self.spec.patches, upstream, req.body, lineup)
            if applied:
                candidate = self.candidate(upstream, self.key(req, body))
            decision, entry = self._decide(rec, req, body, candidate)
        served = Served(rec.seq, upstream, req.method, req.path, decision, entry.step if entry else None, rec.id)
        served.patches = applied
        self.served.append(served)
        if decision in ("tape", "lenient"):
            assert entry is not None
            if decision == "lenient":
                rec.divergence = {**(rec.divergence or {}), "lenient": True}
            return await self.serve_tape(proxy, req, rec, entry)
        if decision == "blocked":
            step = (rec.divergence or {}).get("step")
            rec.status = 409
            rec.served_from = "blocked"
            payload = {"error": "blackbox_divergence", "step": step, "session": self.id, "upstream": upstream}
            rec.chunks += json.dumps(payload).encode()
            rec.content_type = "application/json"
            rec.finish("blackbox_divergence")

            async def finish(_: str | None) -> None:
                await proxy.store_recording(rec)

            return json_outcome(409, payload, finish)
        # live
        sent = body
        if self.spec.model and is_llm_path(req.path):
            sent = _override_model(sent, self.spec.model)
            if sent != body:
                served.model_override = self.spec.model
        if sent != req.body:
            rec.sent_request_body = sent
        rec.served_from = "patched" if applied else "live"
        served.served = rec.served_from
        if req.upstream.stateful:
            from blackbox.proxy.stateful import forward_live_stateful

            return await forward_live_stateful(self, proxy, req, rec, sent)
        return await proxy.record_live(req, attribution, rec, hooks=False)

    def _decide(
        self, rec: Recording, req: IncomingRequest, body: bytes, candidate: TapeEntry | None
    ) -> tuple[str, TapeEntry | None]:
        upstream = req.upstream.name
        if self.live:
            return "live", None
        if candidate is not None:
            if self.spec.mode == "fork" and self.spec.fork_step is not None and candidate.step >= self.spec.fork_step:
                self._go_live(rec)
                return "live", None
            candidate.used = True
            return "tape", candidate
        expected = self.next_unused(upstream, req.path)
        divergence: dict[str, Any] = {
            "step": expected.step if expected is not None else None,
            "node": expected.node if expected is not None else None,
            "tape_exchange": expected.exchange.id if expected is not None else None,
            "upstream": upstream,
            "path": req.path,
            "diff": request_diff(expected.exchange.request_body if expected else None, body),
        }
        rec.divergence = divergence
        if self.first_divergence is None:
            self.first_divergence = {**divergence, "seq": rec.seq}
        if self.spec.mode == "exact":
            if self.spec.lenient and expected is not None:
                expected.used = True
                return "lenient", expected
            return "blocked", None
        self._go_live(rec)
        return "live", None

    def _go_live(self, rec: Recording) -> None:
        self.live = True
        self.went_live_at = rec.seq

    async def serve_tape(self, proxy: Proxy, req: IncomingRequest, rec: Recording, entry: TapeEntry) -> Outcome:
        row = entry.exchange.row
        rec.served_from = f"tape:{row.id}"
        rec.status = row.status or 200
        rec.response_headers = dict(row.response_headers)
        rec.content_type = _first(row.response_headers.get("content-type"))
        rec.stream = row.stream
        body = entry.exchange.response_body or b""
        if req.upstream.stateful and self.spec.sync_forward:
            from blackbox.proxy.stateful import sync_forward

            await sync_forward(self, proxy, req, entry)
        headers = tape_headers(row.response_headers, len(body))
        speed = self.spec.speed
        chunks = split_chunks(body, row.chunk_times)

        async def stream() -> AsyncIterator[bytes]:
            started = time.perf_counter()
            for chunk, at_ms in chunks:
                if speed > 0:
                    delay = started + at_ms / 1000 / speed - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
                rec.add_chunk(chunk)
                yield chunk

        async def finish(error: str | None) -> None:
            rec.finish(error)
            await proxy.store_recording(rec)

        return Outcome(rec.status, headers, stream(), finish)


def _first(value: Any) -> str | None:
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value) if value is not None else None


def tape_headers(stored: dict[str, Any], length: int) -> HeaderList:
    headers: HeaderList = []
    for name, value in stored.items():
        if is_hop_by_hop(name) or name.lower() == "content-length":
            continue
        for item in value if isinstance(value, list) else [value]:
            headers.append((name, str(item)))
    headers.append(("content-length", str(length)))
    return headers


def split_chunks(body: bytes, chunk_times: list[Any]) -> list[tuple[bytes, float]]:
    """The recorded response cut back into its chunks, each with the time it was sent (ms since the start)."""
    if not chunk_times:
        return [(body, 0.0)] if body else []
    points = [(int(offset), float(at)) for offset, at in chunk_times]
    out = []
    for i, (offset, at) in enumerate(points):
        end = points[i + 1][0] if i + 1 < len(points) else len(body)
        out.append((body[offset:end], at))
    return out


def _override_model(body: bytes, model: str) -> bytes:
    try:
        parsed = json.loads(body)
    except ValueError:
        return body
    if not isinstance(parsed, dict) or "model" not in parsed or parsed["model"] == model:
        return body
    parsed["model"] = model
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":")).encode()


class SessionManager:
    """The active sessions, keyed by their new trace id. A call for an expired session's trace gets 410."""

    def __init__(self, services: Services) -> None:
        self.services = services
        self.active: dict[str, ReplaySession] = {}
        self.by_id: dict[str, ReplaySession] = {}
        self._expired: OrderedDict[str, int] = OrderedDict()

    def register(self, session: ReplaySession) -> None:
        self.active[session.trace_id] = session
        self.by_id[session.id] = session

    def lookup(self, trace_id: str) -> ReplaySession | None:
        session = self.active.get(trace_id)
        if session is not None and session.expired():
            self.end(trace_id)
            self.services.spawn(_mark_status(self.services, session.id, "expired"))
            return None
        return session

    def is_expired(self, trace_id: str) -> bool:
        return trace_id in self._expired

    def end(self, trace_id: str) -> None:
        session = self.active.pop(trace_id, None)
        if session is not None:
            self.by_id.pop(session.id, None)
        self._expired[trace_id] = now_ms()
        while len(self._expired) > 10_000:
            self._expired.popitem(last=False)

    def remember_expired(self, trace_ids: list[str]) -> None:
        for trace_id in trace_ids:
            self._expired[trace_id] = now_ms()


async def _mark_status(services: Services, session_id: str, status: str) -> None:
    from sqlalchemy import update
    from sqlalchemy.ext.asyncio import AsyncSession

    from blackbox.store.models import Session

    async def op(session: AsyncSession) -> None:
        await session.execute(
            update(Session)
            .where(Session.id == session_id, Session.status == "active")
            .values(status=status, ended_ms=now_ms())
        )

    await services.store.write(op)
