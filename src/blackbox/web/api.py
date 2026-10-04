"""The REST API (`/api/...`) and the SSE event stream."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from blackbox.runs.context import load_spans
from blackbox.runs.starter import start_run
from blackbox.services import Services
from blackbox.store.models import Run
from blackbox.web.format import row_dict

router = APIRouter(prefix="/api")


def services_of(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


def run_json(run: Run, base_url: str | None = None) -> dict[str, Any]:
    data = row_dict(run)
    if base_url:
        data["url"] = f"{base_url}/runs/{run.id}"
    return data


async def get_run_or_404(services: Services, ref: str) -> Run:
    run = await services.store.reader.find_run(ref)
    if run is None:
        raise HTTPException(404, f"no run {ref}")
    return run


@router.get("/runs")
async def list_runs(
    request: Request,
    profile: str | None = None,
    ending: str | None = None,
    source: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    services = services_of(request)
    runs = await services.store.reader.runs(
        profile=profile, ending=ending, source=source, status=status, limit=min(limit, 500), offset=offset
    )
    return [run_json(run) for run in runs]


@router.get("/runs/{ref}")
async def get_run(request: Request, ref: str) -> dict[str, Any]:
    services = services_of(request)
    run = await get_run_or_404(services, ref)
    data = run_json(run, services.settings.server.base_url)
    entry = await services.store.blobs.get_optional(run.entry_request_blob)
    output = await services.store.blobs.get_optional(run.output_blob)
    data["entry_request"] = json.loads(entry) if entry else None
    if output is not None:
        try:
            data["output"] = json.loads(output)
        except ValueError:
            data["output"] = output.decode("utf-8", errors="replace")
    else:
        data["output"] = None
    data["scores"] = [row_dict(score) for score in await services.store.reader.scores(run.id)]
    return data


@router.get("/runs/{ref}/steps")
async def get_steps(request: Request, ref: str) -> list[dict[str, Any]]:
    services = services_of(request)
    run = await get_run_or_404(services, ref)
    return [row_dict(step) for step in await services.store.reader.steps(run.id)]


@router.get("/runs/{ref}/spans")
async def get_spans(request: Request, ref: str) -> list[dict[str, Any]]:
    services = services_of(request)
    run = await get_run_or_404(services, ref)
    return [span.__dict__ for span in await load_spans(services.store, run.trace_id)]


class StartRunBody(BaseModel):
    profile: str
    input: str | dict[str, Any]
    options: dict[str, Any] = Field(default_factory=dict)
    source: str = "live"
    wait: bool = False
    tags: dict[str, Any] = Field(default_factory=dict)


@router.post("/runs")
async def post_run(request: Request, body: StartRunBody) -> dict[str, Any]:
    services = services_of(request)
    try:
        profile = services.profiles.get(body.profile)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None
    run_input = profile.parse_input(body.input, **body.options) if isinstance(body.input, str) else body.input
    started = await start_run(services, body.profile, run_input, source=body.source, wait=body.wait, tags=body.tags)
    return {
        "run_id": started.run_id,
        "trace_id": started.trace_id,
        "url": f"{services.settings.server.base_url}/runs/{started.run_id}",
        "status": started.status,
        "error": started.error,
    }


@router.get("/events")
async def events(request: Request) -> EventSourceResponse:
    """Server-sent events: `run` for run changes, plus the other event families later phases publish."""
    services = services_of(request)

    async def stream() -> AsyncIterator[dict[str, str]]:
        async with services.bus.subscribe() as queue:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    continue
                family = event.type.split(".", 1)[0]
                yield {"event": family, "data": json.dumps({"type": event.type, **event.data}, default=str)}

    return EventSourceResponse(stream(), ping=None)
