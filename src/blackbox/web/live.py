"""Live scoring in the UI and API: the live page, alerts, markers, live patches and the failed-jobs list."""

from datetime import UTC, datetime
from typing import Any

import plotly.graph_objects as go
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.live.alerts import summary
from blackbox.live.detectors import beta_interval
from blackbox.live.patches import LivePatchError, parse_live_patches
from blackbox.live.service import LiveService
from blackbox.live.signals import alert_periods, load_points, signals
from blackbox.services import Services
from blackbox.store.models import Alert, Job, LivePatch, Marker
from blackbox.util import now_ms
from blackbox.web.format import row_dict
from blackbox.web.overview import _figure

router = APIRouter()

ROLLING = 20


def _services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


def _live(request: Request) -> LiveService:
    live = _services(request).live
    if live is None:
        raise HTTPException(503, "live scoring isn't running")
    return live


def _when(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


# API ------------------------------------------------------------------------------------------------------------------


class MarkerBody(BaseModel):
    profile: str
    text: str = Field(min_length=1, max_length=500)


@router.post("/api/markers")
async def post_marker(request: Request, body: MarkerBody) -> dict[str, Any]:
    marker = await _live(request).add_marker(body.profile, body.text)
    return row_dict(marker)


@router.get("/api/markers")
async def get_markers(request: Request, profile: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    query = select(Marker).order_by(Marker.created_ms.desc()).limit(min(limit, 1000))
    if profile:
        query = query.where(Marker.profile == profile)
    async with _services(request).store.read() as s:
        return [row_dict(m) for m in (await s.execute(query)).scalars()]


@router.get("/api/alerts")
async def get_alerts(
    request: Request, status: str | None = None, profile: str | None = None, limit: int = 100
) -> list[dict[str, Any]]:
    query = select(Alert).order_by(Alert.opened_ms.desc()).limit(min(limit, 1000))
    if status:
        query = query.where(Alert.status == status)
    if profile:
        query = query.where(Alert.profile == profile)
    async with _services(request).store.read() as s:
        return [{**row_dict(a), "summary": summary(a)} for a in (await s.execute(query)).scalars()]


@router.post("/api/alerts/{alert_id}/resolve")
async def api_resolve_alert(request: Request, alert_id: str) -> dict[str, Any]:
    alert = await _live(request).engine.resolve(alert_id, reason="by hand")
    if alert is None:
        raise HTTPException(404, f"no open alert {alert_id}")
    return row_dict(alert)


class LivePatchBody(BaseModel):
    yaml: str
    minutes: float | None = Field(default=None, gt=0, le=24 * 60)


@router.get("/api/live-patches")
async def get_live_patches(request: Request, all: bool = False) -> list[dict[str, Any]]:
    if not all:
        return [p.as_dict() for p in _live(request).patches.active()]
    async with _services(request).store.read() as s:
        rows = (await s.execute(select(LivePatch).order_by(LivePatch.created_ms.desc()).limit(200))).scalars()
        return [row_dict(r) for r in rows]


@router.post("/api/live-patches")
async def post_live_patch(request: Request, body: LivePatchBody) -> list[dict[str, Any]]:
    live = _live(request)
    try:
        patches = parse_live_patches(body.yaml)
    except LivePatchError as exc:
        raise HTTPException(400, str(exc)) from None
    minutes = body.minutes or _services(request).settings.live.patch_minutes
    return [p.as_dict() for p in await live.patches.add(patches, minutes)]


@router.delete("/api/live-patches/{ref}")
async def delete_live_patch(request: Request, ref: str) -> dict[str, Any]:
    removed = await _live(request).patches.remove(ref)
    if not removed:
        raise HTTPException(404, f"no active live patch {ref}")
    return {"removed": removed}


@router.get("/api/jobs")
async def get_jobs(request: Request, status: str = "failed", limit: int = 100) -> list[dict[str, Any]]:
    async with _services(request).store.read() as s:
        rows = (
            await s.execute(
                select(Job).where(Job.status == status).order_by(Job.updated_ms.desc()).limit(min(limit, 1000))
            )
        ).scalars()
        return [row_dict(j) for j in rows]


async def retry_job(services: Services, job_id: str) -> bool:
    async def op(session: AsyncSession) -> int:
        result = await session.execute(
            update(Job)
            .where(Job.id == job_id, Job.status == "failed")
            .values(status="queued", attempts=0, available_ms=now_ms(), updated_ms=now_ms())
        )
        return int(result.rowcount or 0)  # type: ignore[attr-defined]

    changed = await services.store.write(op)
    if changed and services.worker is not None:
        services.worker.wake()
    return bool(changed)


@router.post("/api/jobs/{job_id}/retry")
async def api_retry_job(request: Request, job_id: str) -> dict[str, Any]:
    if not await retry_job(_services(request), job_id):
        raise HTTPException(404, f"no failed job {job_id}")
    return {"retried": job_id}


# Pages ----------------------------------------------------------------------------------------------------------------


def _signal_chart(signal: Any, markers: list[Marker], periods: list[tuple[int, int]]) -> str:
    points = signal.points
    x = [_when(p.t_ms) for p, _ in points]
    if signal.kind == "rate":
        bad = signal.key != "pass_rate"
        values = [(1 - v) if bad else v for _, v in points]
        y = [
            sum(values[max(0, i - ROLLING + 1) : i + 1]) / len(values[max(0, i - ROLLING + 1) : i + 1])
            for i in range(len(values))
        ]
        trace = go.Scatter(x=x, y=y, name=f"{signal.label} (last {ROLLING} runs)", mode="lines")
        html = _figure(signal.label, [trace], "share of runs", percent=True, markers=markers, periods=periods)
    else:
        trace = go.Scatter(x=x, y=[v for _, v in points], name=signal.label, mode="markers")
        html = _figure(signal.label, [trace], signal.label, markers=markers, periods=periods)
    return html


@router.get("/live", response_class=HTMLResponse, include_in_schema=False)
async def live_page(request: Request, profile: str | None = None) -> HTMLResponse:
    services = _services(request)
    live = _live(request)
    profiles = services.profiles.names()
    profile = profile or (profiles[0] if profiles else "")
    now = now_ms()
    points = await load_points(services, profile)
    periods = await alert_periods(services, profile, now)
    sigs = signals(points)
    readings = [live.engine.evaluate_signal(signal, periods) for signal in sigs]
    async with services.store.read() as s:
        markers = list(
            (await s.execute(select(Marker).where(Marker.profile == profile).order_by(Marker.created_ms))).scalars()
        )
        alerts = list(
            (
                await s.execute(
                    select(Alert).where(Alert.profile == profile, Alert.status == "open").order_by(Alert.opened_ms)
                )
            ).scalars()
        )
        queued = (
            await s.execute(
                select(func.count(), func.min(Job.created_ms)).where(Job.lane == "model", Job.status == "queued")
            )
        ).one()
        failed_jobs = (
            await s.execute(select(func.count()).select_from(Job).where(Job.status == "failed"))
        ).scalar_one()
    last_hour = sum(1 for p in points if p.t_ms >= now - 3_600_000)
    last_day = sum(1 for p in points if p.t_ms >= now - 86_400_000)
    pass_rate = next((sig for sig in sigs if sig.key == "pass_rate"), None)
    main: dict[str, Any] | None = None
    if pass_rate is not None and pass_rate.points:
        recent = [v for _, v in pass_rate.points[-ROLLING:]]
        good = int(sum(recent))
        rolling = [
            sum(v for _, v in pass_rate.points[max(0, i - ROLLING + 1) : i + 1])
            / len(pass_rate.points[max(0, i - ROLLING + 1) : i + 1])
            for i in range(max(0, len(pass_rate.points) - 60), len(pass_rate.points))
        ]
        main = {
            "rate": good / len(recent),
            "n": len(recent),
            "interval": beta_interval(good, len(recent)),
            "spark": rolling,
        }
    charts = [_signal_chart(signal, markers, periods) for signal in sigs if len(signal.points) >= 2]
    response: HTMLResponse = request.app.state.templates.TemplateResponse(
        request,
        "live.html",
        {
            "profile": profile,
            "profiles": profiles,
            "runs_last_hour": last_hour,
            "runs_per_hour_day": last_day / 24,
            "main": main,
            "queue_depth": queued[0],
            "queue_lag_ms": (now - queued[1]) if queued[1] else None,
            "failed_jobs": failed_jobs,
            "alerts": [{"row": a, "summary": summary(a)} for a in alerts],
            "patches": live.patches.active(),
            "readings": [r for r in readings if r.state != "insufficient"],
            "insufficient": [r.reading.get("label") or r.rule for r in readings if r.state == "insufficient"],
            "charts": charts,
            "markers": markers[-10:],
            "now": now,
        },
    )
    return response


@router.get("/alerts", response_class=HTMLResponse, include_in_schema=False)
async def alerts_page(request: Request, status: str | None = None, profile: str | None = None) -> HTMLResponse:
    rows = await get_alerts(request, status=status, profile=profile, limit=500)
    response: HTMLResponse = request.app.state.templates.TemplateResponse(
        request, "alerts.html", {"alerts": rows, "status": status, "profile": profile}
    )
    return response


@router.get("/alerts/{alert_id}", response_class=HTMLResponse, include_in_schema=False)
async def alert_page(request: Request, alert_id: str) -> HTMLResponse:
    async with _services(request).store.read() as s:
        alert = await s.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(404, f"no alert {alert_id}")
    response: HTMLResponse = request.app.state.templates.TemplateResponse(
        request,
        "alert.html",
        {
            "alert": alert,
            "summary": summary(alert),
            "d": alert.details,
            "delta": _services(request).settings.live.rate_drop.delta,
        },
    )
    return response


@router.post("/alerts/{alert_id}/resolve", include_in_schema=False)
async def resolve_alert_form(request: Request, alert_id: str) -> RedirectResponse:
    await _live(request).engine.resolve(alert_id, reason="by hand")
    return RedirectResponse(f"/alerts/{alert_id}", status_code=303)


@router.post("/live/markers", include_in_schema=False)
async def marker_form(request: Request, profile: str = Form(), text: str = Form()) -> RedirectResponse:
    if text.strip():
        await _live(request).add_marker(profile, text.strip()[:500])
    return RedirectResponse(f"/live?profile={profile}", status_code=303)


@router.post("/live/patches/{ref}/remove", include_in_schema=False)
async def remove_patch_form(request: Request, ref: str) -> RedirectResponse:
    await _live(request).patches.remove(ref)
    return RedirectResponse(request.headers.get("referer") or "/live", status_code=303)


@router.get("/jobs", response_class=HTMLResponse, include_in_schema=False)
async def jobs_page(request: Request) -> HTMLResponse:
    services = _services(request)
    async with services.store.read() as s:
        failed = list(
            (
                await s.execute(select(Job).where(Job.status == "failed").order_by(Job.updated_ms.desc()).limit(200))
            ).scalars()
        )
        counts = (await s.execute(select(Job.lane, Job.status, func.count()).group_by(Job.lane, Job.status))).all()
    table: dict[str, dict[str, int]] = {}
    for lane, status, count in counts:
        table.setdefault(lane, {})[status] = count
    response: HTMLResponse = request.app.state.templates.TemplateResponse(
        request, "jobs.html", {"failed": failed, "counts": table}
    )
    return response


@router.post("/jobs/{job_id}/retry", include_in_schema=False)
async def retry_job_form(request: Request, job_id: str) -> RedirectResponse:
    await retry_job(_services(request), job_id)
    return RedirectResponse("/jobs", status_code=303)
