"""Run assembly: spans and proxy calls arrive per trace; a run completes when its root span has ended, nothing has
arrived for `quiet_seconds`, and no proxy call is in flight.

The root span is the span with no parent, the span whose parent is the made-up span id in the `traceparent` BlackBox
sent, or a span whose parent is missing and that is either a server (or consumer) span, the entry point of a service,
or flagged by its exporter as having a remote parent. A trace with none of these (an agent called with
someone else's `traceparent` by an old exporter) completes anyway after `orphan_seconds` of quiet.
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.config import RunsConfig
from blackbox.events import EventBus
from blackbox.otlp.decode import SpanData
from blackbox.profiles import ProfileRegistry
from blackbox.runs.complete import CompletionResult, complete_run
from blackbox.store import PreparedBlob, Store, insert_blob
from blackbox.store.models import RecordedValue, Run, Span
from blackbox.util import new_id, now_ms

log = logging.getLogger(__name__)

LARGE_ATTRIBUTE_BYTES = 4096
RECORDED_VALUE_EVENT = "blackbox.recorded_value"


@dataclass
class TraceState:
    trace_id: str
    run_id: str
    remote_parent: str | None
    last_activity_ms: int
    parents: dict[str, str | None] = field(default_factory=dict)
    remote_flagged: set[str] = field(default_factory=set)
    entry_spans: set[str] = field(default_factory=set)  # server and consumer spans: entry points into a service
    in_flight: int = 0
    completing: bool = False
    row_pending: bool = False  # the run row is still being written (a proxy call opened the run)

    def add_span(self, span: SpanData) -> None:
        self.parents[span.span_id] = span.parent_span_id
        if span.parent_is_remote:
            self.remote_flagged.add(span.span_id)
        if span.kind in ("server", "consumer"):
            self.entry_spans.add(span.span_id)

    @property
    def root_ended(self) -> bool:
        for span_id, parent in self.parents.items():
            if parent is None or (self.remote_parent and parent == self.remote_parent):
                return True
            if (span_id in self.remote_flagged or span_id in self.entry_spans) and parent not in self.parents:
                return True
        return False


def _move_large_attributes(attributes: dict[str, Any], blobs: list[PreparedBlob]) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {}
    refs: list[str] = []
    for key, value in attributes.items():
        if isinstance(value, str):
            raw, kind = value.encode("utf-8"), "str"
        else:
            raw, kind = json.dumps(value, ensure_ascii=False).encode("utf-8"), "json"
        if len(raw) > LARGE_ATTRIBUTE_BYTES:
            blob = PreparedBlob.of(raw, "text/plain" if kind == "str" else "application/json")
            blobs.append(blob)
            refs.append(blob.sha256)
            out[key] = {"$blob": blob.sha256, "size": len(raw), "type": kind}
        else:
            out[key] = value
    return out, refs


def recorded_values(span: SpanData) -> list[dict[str, Any]]:
    """Values an SDK agent drew from its clock or random generator, sent as span events."""
    values = []
    for event in span.events:
        if event.get("name") != RECORDED_VALUE_EVENT:
            continue
        attributes = event.get("attributes") or {}
        try:
            values.append(
                {
                    "trace_id": span.trace_id,
                    "seq": int(attributes["blackbox.seq"]),
                    "kind": str(attributes["blackbox.kind"]),
                    "value": str(attributes["blackbox.value"]),
                }
            )
        except KeyError, ValueError:
            log.warning("malformed %s event on span %s", RECORDED_VALUE_EVENT, span.span_id)
    return values


class RunAssembler:
    def __init__(
        self,
        store: Store,
        profiles: ProfileRegistry,
        bus: EventBus,
        config: RunsConfig,
        clock: Callable[[], int] = now_ms,
    ) -> None:
        self._store = store
        self._profiles = profiles
        self._bus = bus
        self._config = config
        self._clock = clock
        self._states: dict[str, TraceState] = {}
        self._task: asyncio.Task[None] | None = None
        self._completions: set[asyncio.Task[CompletionResult | None]] = set()
        self.on_complete: list[Callable[[CompletionResult], Any]] = []
        self._waiters: dict[str, list[asyncio.Future[CompletionResult]]] = {}

    # Lifecycle --------------------------------------------------------------------------------------------------

    async def start(self) -> None:
        await self.recover()
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="blackbox-run-assembler")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._completions:
            await asyncio.gather(*self._completions, return_exceptions=True)

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.tick_seconds)
            try:
                await self.tick(wait=False)
            except Exception:
                log.exception("run assembler tick failed")

    async def recover(self) -> None:
        """Pick up runs still open after a restart; the tick completes them once they are quiet."""
        async with self._store.read() as s:
            runs = (await s.execute(select(Run).where(Run.status == "open"))).scalars().all()
            for run in runs:
                state = TraceState(run.trace_id, run.id, run.remote_parent_span_id, run.updated_ms)
                query = select(Span.span_id, Span.parent_span_id, Span.flags, Span.kind).where(
                    Span.trace_id == run.trace_id
                )
                for span_id, parent, flags, kind in (await s.execute(query)).all():
                    state.parents[span_id] = parent
                    if flags & 0x100 and flags & 0x200:
                        state.remote_flagged.add(span_id)
                    if kind in ("server", "consumer"):
                        state.entry_spans.add(span_id)
                self._states.setdefault(run.trace_id, state)

    # Inputs -----------------------------------------------------------------------------------------------------

    async def ingest(self, spans: list[SpanData]) -> None:
        if not spans:
            return
        now = self._clock()
        blobs: list[PreparedBlob] = []
        rows: list[dict[str, Any]] = []
        values: list[dict[str, Any]] = []
        first_start: dict[str, int] = {}
        for span in spans:
            attributes, refs = _move_large_attributes(span.attributes, blobs)
            rows.append(
                {
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "parent_span_id": span.parent_span_id,
                    "name": span.name,
                    "kind": span.kind,
                    "service": span.service,
                    "scope": span.scope,
                    "start_ns": span.start_ns,
                    "end_ns": span.end_ns,
                    "status_code": span.status_code,
                    "status_message": span.status_message,
                    "flags": span.flags,
                    "attributes": attributes,
                    "events": span.events,
                    "resource": span.resource,
                    "blob_refs": refs,
                    "received_ms": now,
                }
            )
            values.extend(recorded_values(span))
            start_ms = span.start_ns // 1_000_000
            first_start[span.trace_id] = min(first_start.get(span.trace_id, start_ms), start_ms)

        async def op(session: AsyncSession) -> dict[str, tuple[str, str, str | None, bool]]:
            for blob in blobs:
                await insert_blob(session, blob)
            await session.execute(insert(Span).on_conflict_do_nothing(), rows)
            if values:
                await session.execute(insert(RecordedValue).on_conflict_do_nothing(), values)
            result = {}
            for trace_id, start in first_start.items():
                known = self._states.get(trace_id)
                proposed = known.run_id if known is not None else None
                run_id, status, remote, created = await _ensure_run(session, trace_id, now, start, proposed=proposed)
                result[trace_id] = (run_id, status, remote, created)
            return result

        runs = await self._store.write(op)
        for span in spans:
            run_id, status, remote, created = runs[span.trace_id]
            if status != "open":
                continue  # a late span for a completed run is stored, but doesn't reopen it
            state = self._state(span.trace_id, run_id, remote, now)
            state.add_span(span)
            state.last_activity_ms = now
        for trace_id, (run_id, _status, _, created) in runs.items():
            self._bus.publish("run.created" if created else "run.updated", run_id=run_id, trace_id=trace_id)

    async def ensure_run(self, trace_id: str) -> str:
        """Make sure an open run exists for a trace (a proxy call can arrive before any span)."""
        state = self._states.get(trace_id)
        if state is not None:
            return state.run_id
        now = self._clock()

        async def op(session: AsyncSession) -> tuple[str, str, str | None, bool]:
            return await _ensure_run(session, trace_id, now, None)

        run_id, status, remote, created = await self._store.write(op)
        if status == "open":
            self._state(trace_id, run_id, remote, now).run_id = run_id
        if created:
            self._bus.publish("run.created", run_id=run_id, trace_id=trace_id)
        return run_id

    def open_call(self, trace_id: str) -> None:
        """A proxy call for `trace_id` starts: count it in flight, and make sure the run exists without making the
        call wait for a database write (the row is written in the background)."""
        state = self._states.get(trace_id)
        now = self._clock()
        if state is None:
            state = self._state(trace_id, new_id(), None, now)
            state.row_pending = True
            task = asyncio.create_task(self._write_row(state, now))
            self._completions.add(task)
            task.add_done_callback(self._completions.discard)
        state.in_flight += 1
        state.last_activity_ms = now

    async def _write_row(self, state: TraceState, now: int) -> None:
        async def op(session: AsyncSession) -> tuple[str, str, str | None, bool]:
            return await _ensure_run(session, state.trace_id, now, None, proposed=state.run_id)

        try:
            run_id, status, remote, created = await self._store.write(op)
        except Exception:
            log.exception("could not create the run for trace %s", state.trace_id)
            return
        finally:
            state.row_pending = False
        state.run_id = run_id
        state.remote_parent = state.remote_parent or remote
        if status != "open" and self._states.get(state.trace_id) is state:
            del self._states[state.trace_id]  # a late call for a completed run
        if created:
            self._bus.publish("run.created", run_id=run_id, trace_id=state.trace_id)

    def track(self, trace_id: str, run_id: str, remote_parent: str | None) -> None:
        """Register a run BlackBox created itself (its row already exists)."""
        self._state(trace_id, run_id, remote_parent, self._clock())

    def begin_call(self, trace_id: str) -> None:
        state = self._states.get(trace_id)
        if state is not None:
            state.in_flight += 1
            state.last_activity_ms = self._clock()

    def end_call(self, trace_id: str) -> None:
        state = self._states.get(trace_id)
        if state is not None:
            state.in_flight = max(0, state.in_flight - 1)
            state.last_activity_ms = self._clock()

    def in_flight(self) -> int:
        return sum(state.in_flight for state in self._states.values())

    def completion(self, trace_id: str) -> asyncio.Future[CompletionResult]:
        """A future that resolves when the run of `trace_id` completes (or is deleted, or fails assembly)."""
        future: asyncio.Future[CompletionResult] = asyncio.get_running_loop().create_future()
        self._waiters.setdefault(trace_id, []).append(future)
        return future

    def open_count(self) -> int:
        return len(self._states)

    def is_open(self, trace_id: str) -> bool:
        return trace_id in self._states

    def _state(self, trace_id: str, run_id: str, remote: str | None, now: int) -> TraceState:
        state = self._states.get(trace_id)
        if state is None:
            state = self._states[trace_id] = TraceState(trace_id, run_id, remote, now)
        return state

    # Completion -------------------------------------------------------------------------------------------------

    def _ready(self, state: TraceState, now: int) -> bool:
        if state.completing or state.in_flight > 0 or state.row_pending:
            return False
        quiet = now - state.last_activity_ms
        if quiet < self._config.quiet_seconds * 1000:
            return False
        return state.root_ended or quiet >= self._config.orphan_seconds * 1000

    async def tick(self, *, wait: bool = True) -> list[CompletionResult]:
        now = self._clock()
        tasks = []
        for state in list(self._states.values()):
            if self._ready(state, now):
                state.completing = True
                task = asyncio.create_task(self._complete(state))
                self._completions.add(task)
                task.add_done_callback(self._completions.discard)
                tasks.append(task)
        if not wait or not tasks:
            return []
        return [r for r in await asyncio.gather(*tasks) if r is not None]

    async def _complete(self, state: TraceState) -> CompletionResult | None:
        result: CompletionResult | None = None
        try:
            result = await complete_run(
                self._store, self._profiles, state.trace_id, keep_unmatched=self._config.keep_unmatched
            )
        except Exception as exc:
            log.exception("completing run %s failed", state.run_id)
            await self._mark_failed(state, exc)
            result = CompletionResult(state.trace_id, state.run_id, "failed_assembly")
        finally:
            if self._states.get(state.trace_id) is state:
                del self._states[state.trace_id]
        if result.outcome != "skipped":
            self._bus.publish("run.completed", run_id=result.run_id, trace_id=state.trace_id, outcome=result.outcome)
        for callback in self.on_complete:
            callback(result)
        for future in self._waiters.pop(state.trace_id, []):
            if not future.done():
                future.set_result(result)
        return result

    async def _mark_failed(self, state: TraceState, exc: Exception) -> None:
        async def op(session: AsyncSession) -> None:
            run = await session.get(Run, state.run_id)
            if run is not None and run.status == "open":
                await session.execute(
                    update(Run)
                    .where(Run.id == state.run_id)
                    .values(status="failed_assembly", tags={**run.tags, "assembly_error": repr(exc)[:500]})
                )

        try:
            await self._store.write(op)
        except Exception:
            log.exception("could not mark run %s failed", state.run_id)


async def _ensure_run(
    session: AsyncSession, trace_id: str, now: int, start_ms: int | None, proposed: str | None = None
) -> tuple[str, str, str | None, bool]:
    run = (await session.execute(select(Run).where(Run.trace_id == trace_id))).scalar_one_or_none()
    if run is None:
        run_id = proposed or new_id()
        session.add(Run(id=run_id, trace_id=trace_id, status="open", updated_ms=now, started_ms=start_ms, tags={}))
        return run_id, "open", None, True
    values: dict[str, Any] = {"updated_ms": now}
    if start_ms is not None and (run.started_ms is None or start_ms < run.started_ms):
        values["started_ms"] = start_ms
    await session.execute(update(Run).where(Run.id == run.id).values(**values))
    return run.id, run.status, run.remote_parent_span_id, False
