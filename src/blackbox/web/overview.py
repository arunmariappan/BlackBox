"""The profile overview: rates and distributions over time, as Plotly charts drawn server-side."""

import statistics
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import plotly.graph_objects as go
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select

from blackbox.services import Services
from blackbox.store.models import Marker, Run, Score
from blackbox.util import now_ms, parse_duration_ms

router = APIRouter()

RATES = [
    ("checker_pass", "checker pass"),
    ("loop", "loop"),
    ("wrong_tool", "wrong tool"),
    ("unrecovered_errors", "unrecovered error"),
    ("fallbacks", "fallback"),
    ("policy_violations", "policy violation"),
    ("citations_valid", "valid citations"),
    ("scope_mismatch", "scope mismatch"),
]


def _when(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _figure(
    title: str,
    traces: list[Any],
    y_title: str,
    percent: bool = False,
    *,
    markers: Sequence[Marker] = (),
    periods: Sequence[tuple[int, int]] = (),
) -> str:
    """A chart; markers are dotted vertical lines and alert periods shaded bands."""
    fig = go.Figure(traces)
    for start, end in periods:
        fig.add_vrect(x0=_when(start), x1=_when(end), fillcolor="red", opacity=0.08, line_width=0)
    for marker in markers:
        fig.add_vline(x=_when(marker.created_ms).timestamp() * 1000, line_dash="dot", line_color="gray")
        fig.add_annotation(
            x=_when(marker.created_ms), y=1, yref="paper", text=marker.text[:40], showarrow=False, font={"size": 10}
        )
    fig.update_layout(
        title={"text": title, "font": {"size": 14}},
        height=260,
        margin={"l": 40, "r": 10, "t": 40, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": -0.25},
        yaxis={"title": y_title, "tickformat": ".0%" if percent else None, "rangemode": "tozero"},
    )
    html: str = fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
    return html


async def overview_data(services: Services, profile: str, window_ms: int, bucket_ms: int) -> dict[str, Any]:
    since = now_ms() - window_ms
    async with services.store.read() as s:
        runs = list(
            (
                await s.execute(
                    select(Run)
                    .where(Run.profile == profile, Run.status == "complete", Run.started_ms >= since)
                    .order_by(Run.started_ms)
                )
            ).scalars()
        )
        ids = [r.id for r in runs]
        scores = (
            list(
                (
                    await s.execute(select(Score).where(Score.run_id.in_(ids), Score.kind.in_(("metric", "checker"))))
                ).scalars()
            )
            if ids
            else []
        )
    by_run: dict[str, dict[str, Score]] = defaultdict(dict)
    for score in scores:
        by_run[score.run_id][score.name] = score
    buckets: dict[int, list[Run]] = defaultdict(list)
    for run in runs:
        buckets[(run.started_ms or 0) // bucket_ms * bucket_ms].append(run)
    keys = sorted(buckets)
    rates: dict[str, list[float | None]] = {name: [] for name, _ in RATES}
    totals: dict[str, float | None] = {}
    for name, _ in RATES:
        hits = [_hit(name, by_run[r.id].get(name)) for r in runs]
        known = [h for h in hits if h is not None]
        totals[name] = sum(known) / len(known) if known else None
        for key in keys:
            values = [_hit(name, by_run[r.id].get(name)) for r in buckets[key]]
            bucket_known = [v for v in values if v is not None]
            rates[name].append(sum(bucket_known) / len(bucket_known) if bucket_known else None)
    series: dict[str, dict[str, list[float | None]]] = {}
    for label in ("steps", "tokens", "latency_ms"):
        p50: list[float | None] = []
        p95: list[float | None] = []
        for key in keys:
            measured: list[float] = [v for r in buckets[key] if (v := _measure(r, label)) is not None]
            p50.append(statistics.median(measured) if measured else None)
            p95.append(percentile(measured, 0.95))
        series[label] = {"p50": p50, "p95": p95}
    return {"runs": len(runs), "keys": keys, "rates": rates, "totals": totals, "series": series}


def _measure(run: Run, label: str) -> float | None:
    if label == "steps":
        return float(run.step_count)
    if label == "tokens":
        return float(run.input_tokens + run.output_tokens)
    return float(run.duration_ms) if run.duration_ms is not None else None


def _hit(name: str, score: Score | None) -> float | None:
    """1 when the run shows the thing the rate counts, 0 when it doesn't, None when the metric doesn't apply."""
    if score is None or score.value is None:
        return None
    if name in ("checker_pass", "citations_valid"):
        return 1.0 if score.value >= 1 else 0.0
    return 1.0 if score.value > 0 else 0.0


@router.get("/overview", response_class=HTMLResponse, include_in_schema=False)
async def overview_page(request: Request, profile: str | None = None, window: str = "7d") -> HTMLResponse:
    services: Services = request.app.state.services
    profiles = services.profiles.names()
    profile = profile or (profiles[0] if profiles else "")
    window_ms = parse_duration_ms(window)
    bucket_ms = 3_600_000 if window_ms <= 2 * 86_400_000 else 86_400_000
    data = await overview_data(services, profile, window_ms, bucket_ms)
    x = [datetime.fromtimestamp(k / 1000, tz=UTC) for k in data["keys"]]
    async with services.store.read() as s:
        markers = list(
            (
                await s.execute(
                    select(Marker)
                    .where(Marker.profile == profile, Marker.created_ms >= now_ms() - window_ms)
                    .order_by(Marker.created_ms)
                )
            ).scalars()
        )
    charts = []
    rate_traces = [
        go.Scatter(x=x, y=data["rates"][name], name=label, mode="lines+markers", connectgaps=True)
        for name, label in RATES
        if any(v is not None for v in data["rates"][name])
    ]
    if rate_traces:
        charts.append(_figure("Rates", rate_traces, "share of runs", percent=True, markers=markers))
    for label, unit in (("steps", "steps"), ("tokens", "tokens"), ("latency_ms", "ms")):
        series = data["series"][label]
        charts.append(
            _figure(
                label.replace("_ms", " (ms)"),
                [
                    go.Scatter(x=x, y=series["p50"], name="p50", mode="lines+markers"),
                    go.Scatter(x=x, y=series["p95"], name="p95", mode="lines+markers"),
                ],
                unit,
                markers=markers,
            )
        )
    for hook in getattr(request.app.state, "overview_hooks", []):
        charts.extend(await hook(services, profile, window_ms, x))
    rates = [(label, data["totals"][name]) for name, label in RATES if data["totals"][name] is not None]
    response: HTMLResponse = request.app.state.templates.TemplateResponse(
        request,
        "overview.html",
        {"profile": profile, "profiles": profiles, "window": window, "data": data, "charts": charts, "rates": rates},
    )
    return response
