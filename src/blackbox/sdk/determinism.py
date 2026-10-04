"""Determinism shims: `now()`, `uuid4()` and `random()` for agents.

Inside an `agent_run`, every value drawn is recorded as a `blackbox.recorded_value` event on the run's root span, in
order. When the run belongs to a replay session (W3C baggage `blackbox.session=<id>`), the values are fetched from
BlackBox and returned in the recorded order instead, so a time- or randomness-dependent prompt replays exactly. If
the agent asks for more values than were recorded, or for another kind, it gets live values and a
`blackbox.value_divergence` event says so.
"""

import contextlib
import datetime as dt
import json
import random as _random
import urllib.request
import uuid as _uuid
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.trace import Span

RECORDED_VALUE_EVENT = "blackbox.recorded_value"
DIVERGENCE_EVENT = "blackbox.value_divergence"

_endpoint = {"url": "http://127.0.0.1:8200"}


def configure(endpoint: str) -> None:
    _endpoint["url"] = endpoint.rstrip("/")


@dataclass
class _Scope:
    trace_id: str
    session: str | None
    span: Span
    seq: int = 0
    replay: list[dict[str, Any]] | None = None
    errors: list[str] = field(default_factory=list)

    def recorded(self) -> list[dict[str, Any]]:
        if self.replay is None:
            self.replay = _fetch_values(self.session) if self.session else []
        return self.replay


_scope: ContextVar[_Scope | None] = ContextVar("blackbox_sdk_scope", default=None)


def _fetch_values(session: str | None) -> list[dict[str, Any]]:
    if not session:
        return []
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"{_endpoint['url']}/api/sessions/{session}/values", timeout=10) as response:
        values: list[dict[str, Any]] = json.loads(response.read())
    return sorted(values, key=lambda v: int(v["seq"]))


@contextlib.contextmanager
def run_scope(*, trace_id: str, session: str | None, span: Span) -> Iterator[None]:
    token = _scope.set(_Scope(trace_id=trace_id, session=session, span=span))
    try:
        yield
    finally:
        _scope.reset(token)


def _draw[T](kind: str, live: Callable[[], T], encode: Callable[[T], str], decode: Callable[[str], T]) -> T:
    scope = _scope.get()
    if scope is None:
        return live()
    scope.seq += 1
    value: T | None = None
    if scope.session:
        try:
            recorded = scope.recorded()
        except OSError as exc:
            recorded = []
            scope.errors.append(str(exc))
        if scope.seq <= len(recorded) and recorded[scope.seq - 1].get("kind") == kind:
            value = decode(str(recorded[scope.seq - 1]["value"]))
        else:
            scope.span.add_event(
                DIVERGENCE_EVENT,
                {"blackbox.seq": scope.seq, "blackbox.kind": kind, "blackbox.recorded": len(recorded)},
            )
    if value is None:
        value = live()
    scope.span.add_event(
        RECORDED_VALUE_EVENT, {"blackbox.seq": scope.seq, "blackbox.kind": kind, "blackbox.value": encode(value)}
    )
    return value


def now() -> dt.datetime:
    """The current UTC time, recorded (or replayed)."""
    return _draw("now", lambda: dt.datetime.now(dt.UTC), lambda v: v.isoformat(), dt.datetime.fromisoformat)


def uuid4() -> _uuid.UUID:
    """A random UUID, recorded (or replayed)."""
    return _draw("uuid", _uuid.uuid4, str, _uuid.UUID)


def random() -> float:
    """A random float in [0, 1), recorded (or replayed)."""
    return _draw("random", _random.random, repr, float)
