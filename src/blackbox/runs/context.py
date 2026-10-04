"""Everything known about one run, loaded once and handed to profiles, step builders, metrics and judges."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from blackbox.otlp.decode import SpanData
from blackbox.otlp.genai import SpanView, view_of
from blackbox.store import Store
from blackbox.store.models import Exchange, Run, Span, Step


@dataclass
class ExchangeData:
    """An exchange row with its bodies loaded."""

    row: Exchange
    request_body: bytes | None
    response_body: bytes | None
    sent_request_body: bytes | None = None

    @property
    def id(self) -> str:
        return self.row.id

    def request_json(self) -> Any:
        return _json_or_none(self.request_body)


@dataclass
class StepDraft:
    idx: int
    kind: str
    node: str | None = None
    exchange_id: str | None = None
    span_id: str | None = None
    model: str | None = None
    tool_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    started_ms: int | None = None
    latency_ms: int | None = None
    status: str = "ok"
    view: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: Step) -> StepDraft:
        return cls(
            idx=row.idx,
            kind=row.kind,
            node=row.node,
            exchange_id=row.exchange_id,
            span_id=row.span_id,
            model=row.model,
            tool_name=row.tool_name,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            started_ms=row.started_ms,
            latency_ms=row.latency_ms,
            status=row.status,
            view=row.view,
        )

    def to_row(self, run_id: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "idx": self.idx,
            "kind": self.kind,
            "node": self.node,
            "exchange_id": self.exchange_id,
            "span_id": self.span_id,
            "model": self.model,
            "tool_name": self.tool_name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "started_ms": self.started_ms,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "view": self.view,
        }


def _json_or_none(body: bytes | None) -> Any:
    if not body:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


@dataclass
class RunContext:
    run: Run
    spans: list[SpanData]
    exchanges: list[ExchangeData] = field(default_factory=list)
    steps: list[StepDraft] = field(default_factory=list)
    entry_request: dict[str, Any] | None = None  # {"method", "url", "headers", "body"} when BlackBox started the run
    output_body: bytes | None = None  # the raw response, when BlackBox started the run

    @cached_property
    def by_id(self) -> dict[str, SpanData]:
        return {span.span_id: span for span in self.spans}

    @cached_property
    def _views(self) -> dict[str, SpanView]:
        return {}

    def view(self, span: SpanData) -> SpanView:
        cached = self._views.get(span.span_id)
        if cached is None:
            cached = self._views[span.span_id] = view_of(span)
        return cached

    @cached_property
    def children(self) -> dict[str | None, list[SpanData]]:
        result: dict[str | None, list[SpanData]] = {}
        for span in sorted(self.spans, key=lambda s: (s.start_ns, s.span_id)):
            parent = span.parent_span_id if span.parent_span_id in self.by_id else None
            result.setdefault(parent, []).append(span)
        return result

    @cached_property
    def root(self) -> SpanData | None:
        return find_root(self.spans, self.run.remote_parent_span_id)

    def ancestors(self, span_id: str | None) -> Iterator[SpanData]:
        """The span itself, then its parent, grandparent and so on (only spans present in the trace)."""
        seen: set[str] = set()
        while span_id and span_id in self.by_id and span_id not in seen:
            seen.add(span_id)
            span = self.by_id[span_id]
            yield span
            span_id = span.parent_span_id

    def descendants(self, span_id: str) -> Iterator[SpanData]:
        stack = list(self.children.get(span_id, []))
        while stack:
            span = stack.pop()
            yield span
            stack.extend(self.children.get(span.span_id, []))

    def spans_named(self, name: str) -> list[SpanData]:
        return [span for span in self.spans if span.name == name]

    @property
    def entry_body(self) -> Any:
        return self.entry_request.get("body") if self.entry_request else None

    @property
    def output_json(self) -> Any:
        return _json_or_none(self.output_body)


def find_root(spans: Sequence[SpanData], remote_parent: str | None = None) -> SpanData | None:
    """The span with no parent, or whose parent is not in the trace (earliest first)."""
    ids = {span.span_id for span in spans}
    candidates = [
        span
        for span in spans
        if not span.parent_span_id or span.parent_span_id == remote_parent or span.parent_span_id not in ids
    ]
    if not candidates:
        return None
    return min(
        candidates, key=lambda s: (s.parent_span_id is not None and s.parent_span_id != remote_parent, s.start_ns)
    )


def span_data(row: Span, attributes: dict[str, Any] | None = None) -> SpanData:
    return SpanData(
        trace_id=row.trace_id,
        span_id=row.span_id,
        parent_span_id=row.parent_span_id,
        name=row.name,
        kind=row.kind,
        service=row.service,
        scope=row.scope,
        start_ns=row.start_ns,
        end_ns=row.end_ns,
        status_code=row.status_code,
        status_message=row.status_message,
        flags=row.flags,
        attributes=attributes if attributes is not None else dict(row.attributes),
        events=list(row.events),
        resource=dict(row.resource),
    )


async def resolve_attributes(store: Store, attributes: dict[str, Any]) -> dict[str, Any]:
    """Replace `{"$blob": sha}` placeholders (large values moved to blobs) with the values themselves."""
    resolved: dict[str, Any] = {}
    for key, value in attributes.items():
        if isinstance(value, dict) and "$blob" in value:
            raw = await store.blobs.get_optional(value["$blob"])
            if raw is None:
                resolved[key] = value
            elif value.get("type") == "json":
                resolved[key] = json.loads(raw)
            else:
                resolved[key] = raw.decode("utf-8", errors="replace")
        else:
            resolved[key] = value
    return resolved


async def load_spans(store: Store, trace_id: str) -> list[SpanData]:
    rows = await store.reader.spans(trace_id)
    return [span_data(row, await resolve_attributes(store, row.attributes)) for row in rows]


async def load_exchanges(store: Store, trace_id: str) -> list[ExchangeData]:
    out = []
    for row in await store.reader.exchanges(trace_id):
        sent = row.sent_request_blob
        out.append(
            ExchangeData(
                row,
                await store.blobs.get_optional(row.request_blob),
                await store.blobs.get_optional(row.response_blob),
                await store.blobs.get_optional(sent),
            )
        )
    return out


async def load_run_context(store: Store, run: Run, *, with_steps: bool = True) -> RunContext:
    entry: dict[str, Any] | None = None
    raw_entry = await store.blobs.get_optional(run.entry_request_blob)
    if raw_entry is not None:
        entry = json.loads(raw_entry)
    ctx = RunContext(
        run=run,
        spans=await load_spans(store, run.trace_id),
        exchanges=await load_exchanges(store, run.trace_id),
        entry_request=entry,
        output_body=await store.blobs.get_optional(run.output_blob),
    )
    if with_steps:
        ctx.steps = [StepDraft.from_row(row) for row in await store.reader.steps(run.id)]
    return ctx
