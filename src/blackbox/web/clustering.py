"""Failure clusters: the clusters page per profile, a cluster's page, and the failure panel on run pages."""

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.clusters.service import failure_and_cluster
from blackbox.runs.context import RunContext
from blackbox.services import Services
from blackbox.store.models import Cluster, ClusterSpace, Failure, Run
from blackbox.util import now_ms

router = APIRouter()
STATUSES = ("new", "known", "fixed")
DAY_MS = 86_400_000


def services_of(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


def render(request: Request, name: str, **context: Any) -> HTMLResponse:
    response: HTMLResponse = request.app.state.templates.TemplateResponse(request, name, context)
    return response


def sparkline(created: list[int], days: int = 14) -> list[int]:
    """New members per day over the last `days` days (oldest first)."""
    today = now_ms() // DAY_MS
    counts = Counter(ms // DAY_MS for ms in created)
    return [counts.get(day, 0) for day in range(today - days + 1, today + 1)]


@router.get("/clusters", response_class=HTMLResponse, include_in_schema=False)
async def clusters_page(request: Request, profile: str | None = None) -> HTMLResponse:
    services = services_of(request)
    async with services.store.read() as s:
        profiles = sorted(
            set((await s.execute(select(Failure.profile).distinct())).scalars()) | set(services.profiles.names())
        )
        profile = profile or next((p for p in profiles if p), "")
        clusters = list(
            (
                await s.execute(
                    select(Cluster)
                    .where(Cluster.profile == profile, Cluster.status != "retired")
                    .order_by(Cluster.size.desc())
                )
            ).scalars()
        )
        failures = list((await s.execute(select(Failure).where(Failure.profile == profile))).scalars())
        space = await s.get(ClusterSpace, profile)
    created: dict[str, list[int]] = {}
    for failure in failures:
        if failure.cluster_id:
            created.setdefault(failure.cluster_id, []).append(failure.created_ms)
    lines = {c.id: sparkline(created.get(c.id, [])) for c in clusters}
    unclustered = sorted((f for f in failures if f.cluster_id is None), key=lambda f: -f.created_ms)
    return render(
        request,
        "clusters.html",
        profile=profile,
        profiles=profiles,
        clusters=clusters,
        lines=lines,
        unclustered=unclustered,
        space=space,
        statuses=STATUSES,
    )


@router.get("/clusters/{cluster_id}", response_class=HTMLResponse, include_in_schema=False)
async def cluster_page(request: Request, cluster_id: str) -> HTMLResponse:
    services = services_of(request)
    async with services.store.read() as s:
        cluster = await s.get(Cluster, cluster_id)
        if cluster is None:
            raise HTTPException(404, f"no cluster {cluster_id}")
        members = list(
            (
                await s.execute(
                    select(Failure).where(Failure.cluster_id == cluster_id).order_by(Failure.created_ms.desc())
                )
            ).scalars()
        )
        runs = {
            r.id: r for r in (await s.execute(select(Run).where(Run.id.in_([m.run_id for m in members])))).scalars()
        }
    n = max(len(members), 1)
    stats = {
        "endings": Counter(str(runs[m.run_id].ending) for m in members if m.run_id in runs).most_common(),
        "categories": Counter(m.category or "undescribed" for m in members).most_common(),
        "flags": Counter(f for m in members for f in (m.signature.get("flags") or [])).most_common(),
        "nodes": Counter(
            str(m.signature.get("first_problem_node")) for m in members if m.signature.get("first_problem_node")
        ).most_common(),
    }
    return render(
        request, "cluster.html", cluster=cluster, members=members, runs=runs, stats=stats, n=n, statuses=STATUSES
    )


@router.post("/clusters/{cluster_id}/status", include_in_schema=False)
async def set_status(request: Request, cluster_id: str) -> Response:
    services = services_of(request)
    status = str((await request.form()).get("status"))
    if status not in STATUSES:
        raise HTTPException(422, f"status is one of {', '.join(STATUSES)}")

    async def op(session: AsyncSession) -> None:
        await session.execute(
            update(Cluster).where(Cluster.id == cluster_id).values(status=status, updated_ms=now_ms())
        )

    await services.store.write(op)
    return RedirectResponse(request.headers.get("referer") or f"/clusters/{cluster_id}", status_code=303)


@router.get("/api/clusters")
async def clusters_api(request: Request, profile: str | None = None) -> list[dict[str, Any]]:
    services = services_of(request)
    query = select(Cluster).where(Cluster.status != "retired").order_by(Cluster.size.desc())
    if profile:
        query = query.where(Cluster.profile == profile)
    async with services.store.read() as s:
        rows = list((await s.execute(query)).scalars())
    return [
        {
            "id": c.id,
            "profile": c.profile,
            "title": c.title,
            "likely_cause": c.likely_cause,
            "suggested_fix": c.suggested_fix,
            "evidence": c.evidence,
            "cause_status": c.cause_status,
            "size": c.size,
            "status": c.status,
            "updated": datetime.fromtimestamp(c.updated_ms / 1000, tz=UTC).isoformat(),
        }
        for c in rows
    ]


async def failure_panel(services: Services, run: Run, ctx: RunContext) -> str | None:
    """The run page's failure section: the description and the cluster it belongs to."""
    failure, cluster = await failure_and_cluster(services, run.id)
    if failure is None:
        return None
    from markupsafe import escape

    parts = ['<h2>Failure</h2><div class="card">']
    if failure.category:
        parts.append(f'<span class="badge bad">{escape(failure.category)}</span> ')
    parts.append(escape(failure.description or "not described yet"))
    if failure.where_step:
        parts.append(f' <a href="#step-{failure.where_step}">step #{failure.where_step}</a>')
    reasons = failure.signature.get("reasons") or []
    if reasons:
        parts.append(f'<div class="small muted">{escape("; ".join(reasons))}</div>')
    if cluster is not None:
        parts.append(
            f'<div class="small">cluster: <a href="/clusters/{cluster.id}">{escape(cluster.title or cluster.id)}</a>'
            f" ({cluster.size} runs)</div>"
        )
    else:
        parts.append('<div class="small muted">not in a cluster yet</div>')
    parts.append("</div>")
    return "".join(str(p) for p in parts)
