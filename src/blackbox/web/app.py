"""The FastAPI application: UI pages, REST API, SSE events, static files and the OTLP receiver, on one port."""

from pathlib import Path
from typing import Any

import plotly
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from blackbox import __version__
from blackbox.otlp.receiver import otlp_router
from blackbox.services import Services
from blackbox.web import format as fmt

WEB_DIR = Path(__file__).parent
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"


def _context(request: Request) -> dict[str, Any]:
    services: Services | None = getattr(request.app.state, "services", None)
    extra: dict[str, Any] = {
        "theme": request.cookies.get("theme", "auto"),
        "version": __version__,
        "banners": [],
        "nav": getattr(request.app.state, "nav", []),
    }
    if services is not None:
        for provider in getattr(request.app.state, "banner_providers", []):
            extra["banners"].extend(provider(services))
    return extra


def make_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=str(WEB_DIR / "templates"), context_processors=[_context])
    env = templates.env
    env.filters["time"] = fmt.fmt_time
    env.filters["duration"] = fmt.fmt_duration
    env.filters["pretty"] = fmt.fmt_json
    env.filters["num"] = fmt.fmt_num
    env.filters["ending_class"] = fmt.ending_class
    return templates


def create_app(services: Services) -> FastAPI:
    from blackbox.web import api, pages

    app = FastAPI(title="BlackBox", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.services = services
    app.state.templates = make_templates()
    app.state.banner_providers = []
    app.state.run_panels = []
    app.state.nav = [("/runs", "Runs"), ("/unattributed", "Unattributed")]
    app.include_router(otlp_router(services.assembler.ingest))
    app.include_router(api.router)
    app.include_router(pages.router)
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

    @app.get("/static-vendor/plotly.min.js", include_in_schema=False)
    async def plotly_js() -> FileResponse:
        return FileResponse(PLOTLY_JS, media_type="application/javascript")

    @app.get("/theme/{name}", include_in_schema=False)
    async def set_theme(name: str, request: Request) -> RedirectResponse:
        response = RedirectResponse(request.headers.get("referer") or "/", status_code=303)
        response.set_cookie("theme", name if name in ("light", "dark", "auto") else "auto", max_age=365 * 86400)
        return response

    @app.get("/banners", include_in_schema=False)
    async def banners(request: Request) -> Any:
        return app.state.templates.TemplateResponse(request, "_banners.html", {})

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(await health_report(services))

    return app


async def health_report(services: Services) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "ok",
        "version": __version__,
        "store": {"path": str(services.store.path), "writer_running": services.store.writer.running},
        "open_runs": services.assembler.open_count(),
        "calls_in_flight": services.assembler.in_flight(),
    }
    for hook in services.health_hooks:
        report.update(await hook())
    if not services.store.writer.running:
        report["status"] = "degraded"
    return report
