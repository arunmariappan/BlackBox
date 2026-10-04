"""The recording proxy: forwards every call to its upstream, streams the response back untouched, and records the
calls on `record_paths` as exchanges joined to the span that made them.

Replay sessions (phase 4) and live patches (phase 9) plug in through `Proxy.sessions` and `Proxy.request_hooks`.
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlsplit

import httpx
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.config import UpstreamConfig
from blackbox.net import make_client, parse_traceparent
from blackbox.proxy.http import (
    HeaderList,
    canonical_body,
    forward_request_headers,
    header_value,
    path_matches,
    redact,
    request_key,
    response_headers,
)
from blackbox.store import PreparedBlob, Store, insert_blob
from blackbox.store.models import Exchange, Run
from blackbox.util import new_id, now_ms

if TYPE_CHECKING:
    from blackbox.services import Services

log = logging.getLogger(__name__)

FALLBACK_TRACE_HEADER = "x-blackbox-trace-id"  # risk R1's fallback, for agents that don't propagate traceparent
STREAM_TYPES = ("application/x-ndjson", "text/event-stream", "application/ndjson")


@dataclass
class IncomingRequest:
    upstream: UpstreamConfig
    method: str
    path: str  # decoded, used for matching and recording
    raw_path: str  # as sent, used for forwarding
    query: str
    headers: HeaderList
    body: bytes
    received_ms: int = field(default_factory=now_ms)

    def url(self, path: str | None = None) -> str:
        url = f"{self.upstream.target}{path if path is not None else self.raw_path}"
        return f"{url}?{self.query}" if self.query else url

    def json(self) -> Any:
        try:
            return json.loads(self.body) if self.body else None
        except ValueError:
            return None


@dataclass
class Attribution:
    trace_id: str | None
    parent_span_id: str | None


@dataclass
class Recording:
    """One exchange being recorded while its response streams."""

    upstream: str
    attribution: Attribution
    seq: int
    method: str
    path: str
    query: str
    request_headers: dict[str, Any]
    request_body: bytes
    request_key: str
    id: str = field(default_factory=new_id)
    session_id: str | None = None
    sent_request_body: bytes | None = None
    status: int | None = None
    response_headers: dict[str, Any] = field(default_factory=dict)
    content_type: str | None = None
    chunks: bytearray = field(default_factory=bytearray)
    chunk_times: list[list[float]] = field(default_factory=list)
    started_ms: int = field(default_factory=now_ms)
    first_byte_ms: int | None = None
    ended_ms: int | None = None
    error: str | None = None
    served_from: str = "live"
    divergence: dict[str, Any] | None = None
    stream: bool = False
    _t0: float = field(default_factory=time.perf_counter)

    def add_chunk(self, chunk: bytes) -> None:
        elapsed = round((time.perf_counter() - self._t0) * 1000, 2)
        if self.first_byte_ms is None:
            self.first_byte_ms = self.started_ms + int(elapsed)
        self.chunk_times.append([len(self.chunks), elapsed])
        self.chunks += chunk

    def finish(self, error: str | None = None) -> None:
        if error and not self.error:
            self.error = error
        if self.ended_ms is None:
            self.ended_ms = self.started_ms + int((time.perf_counter() - self._t0) * 1000)


@dataclass
class Outcome:
    """What the listener sends back: status, headers, the body chunk by chunk, and a callback once it's done."""

    status: int
    headers: HeaderList
    body: AsyncIterator[bytes]
    finish: Callable[[str | None], Awaitable[None]]


async def _one(data: bytes) -> AsyncIterator[bytes]:
    yield data


async def _nothing(_: str | None) -> None:
    return None


def json_outcome(
    status: int, payload: dict[str, Any], finish: Callable[[str | None], Awaitable[None]] = _nothing
) -> Outcome:
    data = json.dumps(payload).encode()
    headers = [("content-type", "application/json"), ("content-length", str(len(data)))]
    return Outcome(status, headers, _one(data), finish)


class SessionRouter(Protocol):
    """Replay sessions keyed by trace id (phase 4)."""

    def lookup(self, trace_id: str) -> Any: ...

    def is_expired(self, trace_id: str) -> bool: ...


type RequestHook = Callable[[IncomingRequest, Recording], Awaitable[None]]


class SeqCounter:
    """Per-trace order of request start, from 1, kept in memory so a call never waits for the database. Runs
    still open from before a restart are preloaded, so their numbering continues."""

    def __init__(self, store: Store) -> None:
        self._store = store
        self._next: dict[str | None, int] = {}

    def next(self, trace_id: str | None) -> int:
        value = self._next.get(trace_id, 0) + 1
        self._next[trace_id] = value
        return value

    async def preload(self) -> None:
        async with self._store.read() as s:
            query = (
                select(Exchange.trace_id, func.max(Exchange.seq))
                .join(Run, Run.trace_id == Exchange.trace_id)
                .where(Run.status == "open")
                .group_by(Exchange.trace_id)
            )
            for trace_id, value in (await s.execute(query)).all():
                self._next[trace_id] = max(self._next.get(trace_id, 0), int(value or 0))
            unattributed = select(func.max(Exchange.seq)).where(Exchange.trace_id.is_(None))
            self._next[None] = int((await s.execute(unattributed)).scalar_one_or_none() or 0)

    def forget(self, trace_id: str) -> None:
        self._next.pop(trace_id, None)


class Proxy:
    def __init__(self, services: Services, upstreams: list[UpstreamConfig]) -> None:
        self.services = services
        self.upstreams = {u.name: u for u in upstreams}
        self.clients = {u.name: make_client(timeout=u.timeout_seconds) for u in upstreams}
        self.seq = SeqCounter(services.store)
        self.sessions: SessionRouter | None = None
        self.request_hooks: list[RequestHook] = []
        self.ports: dict[str, int] = {}
        self.redact_headers = list(services.settings.proxy.redact_headers)
        self._pending: set[asyncio.Task[Any]] = set()
        services.assembler.on_complete.append(lambda result: self.seq.forget(result.trace_id))

    async def close(self) -> None:
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)
        for client in self.clients.values():
            await client.aclose()

    # Attribution ----------------------------------------------------------------------------------------------------

    @staticmethod
    def attribute(headers: HeaderList) -> Attribution:
        parsed = parse_traceparent(header_value(headers, "traceparent"))
        if parsed is not None:
            return Attribution(*parsed)
        fallback = (header_value(headers, FALLBACK_TRACE_HEADER) or "").strip().lower()
        if len(fallback) == 32 and all(c in "0123456789abcdef" for c in fallback):
            return Attribution(fallback, None)
        return Attribution(None, None)

    # Pipeline -------------------------------------------------------------------------------------------------------

    async def handle(self, req: IncomingRequest) -> Outcome:
        if not path_matches(req.path, req.upstream.record_paths):
            return await self.passthrough(req)
        attribution = self.attribute(req.headers)
        if self.sessions is not None and attribution.trace_id is not None:
            session = self.sessions.lookup(attribution.trace_id)
            if session is not None:
                outcome: Outcome = await session.handle(self, req, attribution)
                return outcome
            if self.sessions.is_expired(attribution.trace_id):
                return json_outcome(
                    410,
                    {
                        "error": "blackbox_session_expired",
                        "trace_id": attribution.trace_id,
                        "upstream": req.upstream.name,
                    },
                )
        return await self.record_live(req, attribution)

    async def begin(
        self, req: IncomingRequest, attribution: Attribution, *, session_id: str | None = None
    ) -> Recording:
        """Start recording a call: open the run, count it in flight, give it its sequence number."""
        trace_id = attribution.trace_id
        if trace_id is not None:
            self.services.assembler.open_call(trace_id)
        seq = self.seq.next(trace_id)
        return Recording(
            upstream=req.upstream.name,
            attribution=attribution,
            seq=seq,
            method=req.method,
            path=req.path,
            query=req.query,
            request_headers=redact(req.headers, self.redact_headers),
            request_body=req.body,
            request_key=request_key(req.method, req.path, req.query, req.body),
            session_id=session_id,
            started_ms=req.received_ms,
        )

    async def record_live(
        self, req: IncomingRequest, attribution: Attribution, rec: Recording | None = None
    ) -> Outcome:
        if rec is None:
            rec = await self.begin(req, attribution)
        body = req.body
        for hook in self.request_hooks:
            await hook(req, rec)
        if rec.sent_request_body is not None:
            body = rec.sent_request_body
        try:
            response = await self.send(req, body)
        except httpx.HTTPError as exc:
            detail = f"{type(exc).__name__}: {exc}".strip()
            rec.status = 502
            rec.finish(f"upstream_error: {detail}")
            await self.store_recording(rec)
            return json_outcome(
                502, {"error": "blackbox_upstream_error", "upstream": req.upstream.name, "detail": detail}
            )
        return self.stream_live(rec, response)

    def stream_live(self, rec: Recording, response: httpx.Response) -> Outcome:
        rec.status = response.status_code
        headers = response_headers(response.headers.multi_items())
        rec.response_headers = redact(headers, self.redact_headers)
        rec.content_type = response.headers.get("content-type")
        rec.stream = _is_stream(rec.content_type, rec.request_body)

        async def body() -> AsyncIterator[bytes]:
            async for chunk in response.aiter_raw():
                rec.add_chunk(chunk)
                yield chunk

        async def finish(error: str | None) -> None:
            try:
                await response.aclose()
            finally:
                rec.finish(error)
                await self.store_recording(rec)

        return Outcome(response.status_code, headers, body(), finish)

    async def send(self, req: IncomingRequest, body: bytes, *, path: str | None = None) -> httpx.Response:
        client = self.clients[req.upstream.name]
        target = urlsplit(req.upstream.target)
        request = client.build_request(
            req.method,
            req.url(path),
            headers=forward_request_headers(req.headers, target.netloc),
            content=body,
        )
        return await client.send(request, stream=True)

    async def passthrough(self, req: IncomingRequest) -> Outcome:
        try:
            response = await self.send(req, req.body)
        except httpx.HTTPError as exc:
            return json_outcome(
                502, {"error": "blackbox_upstream_error", "upstream": req.upstream.name, "detail": str(exc)}
            )

        async def finish(_: str | None) -> None:
            await response.aclose()

        return Outcome(
            response.status_code, response_headers(response.headers.multi_items()), response.aiter_raw(), finish
        )

    # Storage --------------------------------------------------------------------------------------------------------

    async def store_recording(self, rec: Recording) -> None:
        """Write the exchange, then tell the assembler the call has ended. Never lets a failure pass silently."""
        try:
            await write_exchange(self.services.store, rec)
            self.services.bus.publish(
                "run.updated" if rec.attribution.trace_id else "exchange.unattributed",
                trace_id=rec.attribution.trace_id,
                exchange_id=rec.id,
            )
        except Exception:
            log.exception("could not store exchange %s on %s", rec.id, rec.upstream)
        finally:
            if rec.attribution.trace_id is not None:
                self.services.assembler.end_call(rec.attribution.trace_id)

    def spawn(self, coro: Awaitable[Any]) -> None:
        task = asyncio.ensure_future(coro)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)


def _is_stream(content_type: str | None, request_body: bytes) -> bool:
    if content_type and any(kind in content_type for kind in STREAM_TYPES):
        return True
    parsed = canonical_body(request_body)
    return isinstance(parsed, dict) and parsed.get("stream") is True


async def write_exchange(store: Store, rec: Recording) -> None:
    request_blob = (
        PreparedBlob.of(rec.request_body, rec.request_headers.get("content-type")) if rec.request_body else None
    )
    response_blob = PreparedBlob.of(bytes(rec.chunks), rec.content_type) if rec.chunks else None
    sent_blob = (
        PreparedBlob.of(rec.sent_request_body, rec.request_headers.get("content-type"))
        if rec.sent_request_body is not None and rec.sent_request_body != rec.request_body
        else None
    )
    values: dict[str, Any] = {
        "id": rec.id,
        "trace_id": rec.attribution.trace_id,
        "parent_span_id": rec.attribution.parent_span_id,
        "session_id": rec.session_id,
        "upstream": rec.upstream,
        "seq": rec.seq,
        "method": rec.method,
        "path": rec.path,
        "query": rec.query,
        "request_headers": rec.request_headers,
        "request_key": rec.request_key,
        "status": rec.status,
        "response_headers": rec.response_headers,
        "stream": rec.stream,
        "chunk_times": rec.chunk_times,
        "started_ms": rec.started_ms,
        "first_byte_ms": rec.first_byte_ms,
        "ended_ms": rec.ended_ms,
        "error": rec.error,
        "served_from": rec.served_from,
        "divergence": rec.divergence,
    }

    async def op(session: AsyncSession) -> None:
        if request_blob is not None:
            values["request_blob"] = await insert_blob(session, request_blob)
        if response_blob is not None:
            values["response_blob"] = await insert_blob(session, response_blob)
        if sent_blob is not None and hasattr(Exchange, "sent_request_blob"):
            values["sent_request_blob"] = await insert_blob(session, sent_blob)
        session.add(Exchange(**values))

    await store.write(op)
