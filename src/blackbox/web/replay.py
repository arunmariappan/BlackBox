"""Replay and fork: the sessions API, the session page and the fork form."""

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select

from blackbox.net import make_client
from blackbox.proxy.sessions import SessionSpec
from blackbox.replay.patches import Patch, load_patches
from blackbox.replay.runner import ReplayError, start_replay
from blackbox.services import Services
from blackbox.store.models import Session
from blackbox.web.format import row_dict

router = APIRouter()


def services_of(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


def render(request: Request, name: str, **context: Any) -> HTMLResponse:
    response: HTMLResponse = request.app.state.templates.TemplateResponse(request, name, context)
    return response


class SessionBody(BaseModel):
    source_run_id: str
    mode: str = "exact"
    fork_step: int | None = None
    model: str | None = None
    patches: list[dict[str, Any]] | str = Field(default_factory=list)  # parsed patches, or the YAML text of a file
    speed: float = 0.0
    lenient: bool = False
    aliasing: bool = True
    sync_forward: bool = True
    wait: bool = False


def spec_of(body: SessionBody) -> SessionSpec:
    try:
        patches: list[Patch] = load_patches(body.patches) if body.patches else []
    except (ValidationError, ValueError) as exc:
        raise HTTPException(422, f"invalid patches: {exc}") from None
    return SessionSpec(
        source_run_id=body.source_run_id,
        mode=body.mode,
        fork_step=body.fork_step,
        model=body.model or None,
        patches=patches,
        speed=body.speed,
        lenient=body.lenient,
        aliasing=body.aliasing,
        sync_forward=body.sync_forward,
    )


@router.post("/api/sessions")
async def create_session(request: Request, body: SessionBody) -> dict[str, Any]:
    services = services_of(request)
    try:
        prepared, report = await start_replay(services, spec_of(body), wait=body.wait)
    except ReplayError as exc:
        raise HTTPException(400, str(exc)) from None
    session = prepared.session
    return {
        "session_id": session.id,
        "trace_id": session.trace_id,
        "source_run_id": prepared.source.id,
        "rebuilt_input": prepared.rebuilt,
        "url": f"{services.settings.server.base_url}/sessions/{session.id}",
        "report": report,
    }


@router.get("/api/sessions")
async def list_sessions(request: Request, source_run_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    services = services_of(request)
    query = select(Session).order_by(Session.created_ms.desc()).limit(min(limit, 500))
    if source_run_id:
        query = query.where(Session.source_run_id == source_run_id)
    async with services.store.read() as s:
        return [row_dict(row) for row in (await s.execute(query)).scalars()]


@router.get("/api/sessions/{session_id}")
async def get_session(request: Request, session_id: str) -> dict[str, Any]:
    services = services_of(request)
    row = await services.store.reader.session(session_id)
    if row is None:
        raise HTTPException(404, f"no session {session_id}")
    data = row_dict(row)
    replay = await services.store.reader.run_by_trace(row.trace_id)
    data["replay_run_id"] = replay.id if replay else None
    return data


@router.get("/api/sessions/{session_id}/values")
async def session_values(request: Request, session_id: str) -> list[dict[str, Any]]:
    """The clock and random values the source run's SDK recorded, in order, for the SDK's determinism shims."""
    services = services_of(request)
    row = await services.store.reader.session(session_id)
    if row is None:
        raise HTTPException(404, f"no session {session_id}")
    source = await services.store.reader.run(row.source_run_id)
    if source is None:
        raise HTTPException(404, f"source run {row.source_run_id} is gone")
    values = await services.store.reader.recorded_values(source.trace_id)
    return [{"seq": v.seq, "kind": v.kind, "value": v.value} for v in values]


# Pages -------------------------------------------------------------------------------------------------------------


@router.get("/sessions", response_class=HTMLResponse, include_in_schema=False)
async def sessions_page(request: Request) -> HTMLResponse:
    services = services_of(request)
    async with services.store.read() as s:
        rows = (await s.execute(select(Session).order_by(Session.created_ms.desc()).limit(200))).scalars().all()
    return render(request, "sessions.html", sessions=rows)


@router.get("/sessions/{session_id}", response_class=HTMLResponse, include_in_schema=False)
async def session_page(request: Request, session_id: str) -> HTMLResponse:
    services = services_of(request)
    row = await services.store.reader.session(session_id)
    if row is None:
        raise HTTPException(404, f"no session {session_id}")
    source = await services.store.reader.run(row.source_run_id)
    replay = await services.store.reader.run_by_trace(row.trace_id)
    source_steps = list(await services.store.reader.steps(source.id)) if source else []
    replay_steps = list(await services.store.reader.steps(replay.id)) if replay else []
    return render(
        request,
        "session.html",
        session=row,
        report=row.result or {},
        source=source,
        replay=replay,
        pairs=_pairs(source_steps, replay_steps),
    )


def _pairs(source: list[Any], replay: list[Any]) -> list[tuple[Any, Any]]:
    return [
        (source[i] if i < len(source) else None, replay[i] if i < len(replay) else None)
        for i in range(max(len(source), len(replay)))
    ]


@router.post("/runs/{ref}/replay", include_in_schema=False)
async def replay_from_form(request: Request, ref: str) -> Response:
    services = services_of(request)
    form = await request.form()
    run = await services.store.reader.find_run(ref)
    if run is None:
        raise HTTPException(404, f"no run {ref}")
    fork_step = str(form.get("fork_step") or "")
    body = SessionBody(
        source_run_id=run.id,
        mode=str(form.get("mode") or "exact"),
        fork_step=int(fork_step) if fork_step.isdigit() else None,
        model=str(form.get("model") or "") or None,
        speed=float(str(form.get("speed") or 0)),
        patches=_message_patches(form, run_step=int(fork_step) if fork_step.isdigit() else None),
    )
    try:
        prepared, _ = await start_replay(services, spec_of(body))
    except ReplayError as exc:
        raise HTTPException(400, str(exc)) from None
    return RedirectResponse(f"/sessions/{prepared.session.id}", status_code=303)


def _message_patches(form: Any, run_step: int | None) -> list[dict[str, Any]]:
    """Edited messages in the fork form become `set` patches for that step."""
    patches = []
    upstream = str(form.get("upstream") or "") or None
    for key in form:
        if not key.startswith("message_"):
            continue
        index = int(key.removeprefix("message_"))
        new, old = str(form.get(key)), str(form.get(f"original_{index}") or "")
        if new.replace("\r\n", "\n") != old.replace("\r\n", "\n"):
            patches.append(
                {
                    "name": f"edit step {run_step} message {index}",
                    "match": {"step": run_step, **({"upstream": upstream} if upstream else {})},
                    "edit": {"path": f"$.messages[{index}].content", "set": new.replace("\r\n", "\n")},
                }
            )
    return patches


@router.get("/runs/{ref}/fork", response_class=HTMLResponse, include_in_schema=False)
async def fork_form(request: Request, ref: str, step: int) -> HTMLResponse:
    services = services_of(request)
    run = await services.store.reader.find_run(ref)
    if run is None:
        raise HTTPException(404, f"no run {ref}")
    steps = await services.store.reader.steps(run.id)
    target = next((s for s in steps if s.idx == step), None)
    if target is None:
        raise HTTPException(404, f"run {run.id} has no step {step}")
    messages: list[dict[str, Any]] = []
    upstream = None
    if target.exchange_id:
        exchange = await services.store.reader.exchange(target.exchange_id)
        if exchange is not None:
            upstream = exchange.upstream
            raw = await services.store.blobs.get_optional(exchange.request_blob)
            try:
                body = json.loads(raw) if raw else {}
            except ValueError:
                body = {}
            if isinstance(body, dict) and isinstance(body.get("messages"), list):
                messages = [m for m in body["messages"] if isinstance(m, dict)]
    return render(
        request,
        "fork.html",
        run=run,
        step=target,
        messages=messages,
        upstream=upstream,
        models=await _models(services),
    )


async def _models(services: Services) -> list[str]:
    """Models Ollama has (its `/api/tags`), for the fork form; empty when Ollama can't be reached."""
    try:
        target = services.settings.proxy.upstream("ollama").target
    except KeyError:
        target = services.settings.ollama.base_url
    try:
        async with make_client(timeout=2) as client:
            response = await client.get(f"{target}/api/tags")
            return sorted(str(m.get("name")) for m in response.json().get("models", []))
    except Exception:
        return []
