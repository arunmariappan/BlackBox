"""HTML pages (Jinja2 + HTMX)."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from blackbox.runs.context import ExchangeData, load_run_context
from blackbox.services import Services
from blackbox.web.format import waterfall

router = APIRouter()


def services_of(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


def render(request: Request, name: str, **context: Any) -> HTMLResponse:
    response: HTMLResponse = request.app.state.templates.TemplateResponse(request, name, context)
    return response


RUN_FILTERS = ("profile", "ending", "source", "status")


async def _runs(request: Request) -> tuple[list[Any], dict[str, str]]:
    services = services_of(request)
    filters = {key: request.query_params.get(key, "") for key in RUN_FILTERS}
    runs = await services.store.reader.runs(
        **{key: value or None for key, value in filters.items()},  # type: ignore[arg-type]
        limit=int(request.query_params.get("limit", 100)),
    )
    return list(runs), filters


@router.get("/", include_in_schema=False)
async def index() -> Response:
    return RedirectResponse("/runs", status_code=307)


@router.get("/runs", response_class=HTMLResponse, include_in_schema=False)
async def runs_page(request: Request) -> HTMLResponse:
    runs, filters = await _runs(request)
    services = services_of(request)
    return render(request, "runs.html", runs=runs, filters=filters, profiles=services.profiles.names())


@router.get("/runs/rows", response_class=HTMLResponse, include_in_schema=False)
async def runs_rows(request: Request) -> HTMLResponse:
    runs, _ = await _runs(request)
    return render(request, "_runs_rows.html", runs=runs)


@router.get("/runs/{ref}", response_class=HTMLResponse, include_in_schema=False)
async def run_page(request: Request, ref: str) -> HTMLResponse:
    services = services_of(request)
    run = await services.store.reader.find_run(ref)
    if run is None:
        raise HTTPException(404, f"no run {ref}")
    ctx = await load_run_context(services.store, run)
    profile = services.profiles.find(run.profile)
    output = profile.read_output(ctx) if profile is not None else ctx.output_json
    exchanges = {exchange.id: exchange for exchange in ctx.exchanges}
    span_by_id = ctx.by_id
    scores = await services.store.reader.scores(run.id)
    context: dict[str, Any] = {
        "run": run,
        "ctx": ctx,
        "output": output,
        "steps": ctx.steps,
        "exchanges": exchanges,
        "span_by_id": span_by_id,
        "waterfall": waterfall(ctx, profile.node_spans if profile else frozenset()),
        "scores": scores,
        "panels": [],
        "tab": request.query_params.get("tab", "steps"),
    }
    for provider in getattr(request.app.state, "run_panels", []):
        panel = await provider(services, run, ctx)
        if panel:
            context["panels"].append(panel)
    return render(request, "run.html", **context)


@router.get("/unattributed", response_class=HTMLResponse, include_in_schema=False)
async def unattributed_page(request: Request) -> HTMLResponse:
    services = services_of(request)
    exchanges = await services.store.reader.unattributed_exchanges()
    return render(request, "unattributed.html", exchanges=exchanges)


@router.get("/exchanges/{exchange_id}", response_class=HTMLResponse, include_in_schema=False)
async def exchange_page(request: Request, exchange_id: str) -> HTMLResponse:
    services = services_of(request)
    row = await services.store.reader.exchange(exchange_id)
    if row is None:
        raise HTTPException(404, f"no exchange {exchange_id}")
    blobs = services.store.blobs
    exchange = ExchangeData(
        row,
        await blobs.get_optional(row.request_blob),
        await blobs.get_optional(row.response_blob),
        await blobs.get_optional(getattr(row, "sent_request_blob", None)),
    )
    template = "_exchange.html" if request.headers.get("hx-request") else "exchange.html"
    return render(request, template, exchange=exchange)
