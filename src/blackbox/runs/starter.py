"""Starting runs: BlackBox chooses the trace id, sends the agent's entry request with a `traceparent` carrying it,
and stores the exact request and the response."""

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.net import new_span_id, new_trace_id, traceparent
from blackbox.profiles.base import StartRequest
from blackbox.store import PreparedBlob, insert_blob
from blackbox.store.models import Run
from blackbox.util import new_id, now_ms

if TYPE_CHECKING:
    from blackbox.services import Services

log = logging.getLogger(__name__)


@dataclass
class StartedRun:
    run_id: str
    trace_id: str
    traceparent: str
    status: int | None = None
    error: str | None = None
    response: Any = None
    tags: dict[str, Any] = field(default_factory=dict)


async def create_run(
    services: Services,
    profile_name: str,
    request: StartRequest,
    *,
    source: str = "live",
    trace_id: str | None = None,
    session_id: str | None = None,
    replay_of: str | None = None,
    tags: dict[str, Any] | None = None,
) -> StartedRun:
    """Create the run row and register it with the assembler; `send_entry_request` then starts it."""
    trace_id = trace_id or new_trace_id()
    span_id = new_span_id()
    run_id = new_id()
    now = now_ms()
    envelope = PreparedBlob.of(json.dumps(request.envelope(), ensure_ascii=False).encode(), "application/json")
    body = request.body if isinstance(request.body, dict) else {}
    text = next((body[k] for k in ("query", "instruction", "input", "question") if isinstance(body.get(k), str)), None)
    run_tags = dict(tags or {})

    async def op(session: AsyncSession) -> None:
        await insert_blob(session, envelope)
        session.add(
            Run(
                id=run_id,
                trace_id=trace_id,
                profile=profile_name,
                status="open",
                source=source,
                started_ms=now,
                updated_ms=now,
                entry_request_blob=envelope.sha256,
                input_text=text[:500] if text else None,
                model=body.get("model") if isinstance(body.get("model"), str) else None,
                session_id=session_id,
                replay_of=replay_of,
                remote_parent_span_id=span_id,
                tags=run_tags,
            )
        )

    await services.store.write(op)
    services.assembler.track(trace_id, run_id, span_id)
    services.bus.publish("run.created", run_id=run_id, trace_id=trace_id)
    return StartedRun(run_id, trace_id, traceparent(trace_id, span_id), tags=run_tags)


async def send_entry_request(
    services: Services, started: StartedRun, request: StartRequest, *, extra_headers: dict[str, str] | None = None
) -> StartedRun:
    """Send the entry request and store its response. The run stays open (a call is in flight) until then."""
    headers = {**request.headers, "traceparent": started.traceparent, **(extra_headers or {})}
    services.assembler.begin_call(started.trace_id)
    raw: bytes | None = None
    content_type: str | None = None
    try:
        response = await services.http.request(
            request.method,
            request.url,
            json=request.body if request.body is not None else None,
            headers=headers,
            timeout=request.timeout_seconds,
        )
        started.status = response.status_code
        raw = response.content
        content_type = response.headers.get("content-type")
        try:
            started.response = response.json()
        except ValueError:
            started.response = response.text
    except httpx.HTTPError as exc:
        started.error = f"{type(exc).__name__}: {exc}"
        log.warning("starting run %s failed: %s", started.run_id, started.error)
    finally:
        try:
            await _store_response(services, started, raw, content_type)
        finally:
            services.assembler.end_call(started.trace_id)
    return started


async def _store_response(services: Services, started: StartedRun, raw: bytes | None, content_type: str | None) -> None:
    blob = PreparedBlob.of(raw, content_type) if raw is not None else None
    tags = dict(started.tags)
    if started.status is not None:
        tags["entry_status"] = started.status
    if started.error is not None:
        tags["start_error"] = started.error

    async def op(session: AsyncSession) -> None:
        values: dict[str, Any] = {"tags": tags}
        if blob is not None:
            values["output_blob"] = await insert_blob(session, blob)
        await session.execute(update(Run).where(Run.id == started.run_id).values(**values))

    await services.store.write(op)


async def start_run(
    services: Services,
    profile_name: str,
    run_input: dict[str, Any],
    *,
    source: str = "live",
    wait: bool = False,
    tags: dict[str, Any] | None = None,
) -> StartedRun:
    """Start a run of `profile_name`. With `wait=False` the entry request is sent in the background."""
    profile = services.profiles.get(profile_name)
    run_input = await profile.prepare_input(run_input, services)
    request = profile.build_request(run_input)
    started = await create_run(services, profile_name, request, source=source, tags=tags)
    if wait:
        return await send_entry_request(services, started, request)
    services.spawn(send_entry_request(services, started, request), name=f"start-run-{started.run_id}")
    return started
