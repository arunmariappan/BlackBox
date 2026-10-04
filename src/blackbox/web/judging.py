"""Judges and labels: the judges page with agreement statistics, and the blind label page."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import select

from blackbox.judges import calibrate
from blackbox.judges.inputs import last_search_hits
from blackbox.judges.labels import VALUES, label_of, questions, queue, save_label
from blackbox.llm.client import LLMUnavailable
from blackbox.runs.context import load_run_context
from blackbox.services import Services
from blackbox.store.models import Score

router = APIRouter()


def services_of(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


def render(request: Request, name: str, **context: Any) -> HTMLResponse:
    response: HTMLResponse = request.app.state.templates.TemplateResponse(request, name, context)
    return response


@router.get("/judges", response_class=HTMLResponse, include_in_schema=False)
async def judges_page(request: Request) -> HTMLResponse:
    services = services_of(request)
    rows = []
    for judge in services.judges.values():
        rows.append({"judge": judge, "versions": await calibrate.versions(services.store, judge.name)})
    return render(request, "judges.html", rows=rows)


@router.get("/judges/{name}", response_class=HTMLResponse, include_in_schema=False)
async def judge_page(request: Request, name: str) -> HTMLResponse:
    services = services_of(request)
    judge = services.judges.get(name)
    if judge is None:
        raise HTTPException(404, f"no judge {name}")
    versions = await calibrate.versions(services.store, name)
    stats = {
        row.version: row.agreement or await calibrate.compute(services.store, judge, row.version) for row in versions
    }
    return render(
        request, "judge.html", judge=judge, versions=versions, stats=stats, error=request.query_params.get("error")
    )


@router.post("/judges/{name}/calibrate", include_in_schema=False)
async def calibrate_judge(request: Request, name: str) -> Response:
    services = services_of(request)
    judge = services.judges.get(name)
    if judge is None:
        raise HTTPException(404, f"no judge {name}")
    try:
        await calibrate.calibrate(services.store, services.judge_runner, judge)
    except LLMUnavailable as exc:
        await calibrate.calibrate(services.store, services.judge_runner, judge, judge_missing=False)
        return RedirectResponse(f"/judges/{name}?error=model+unavailable:+{str(exc)[:120]}", status_code=303)
    return RedirectResponse(f"/judges/{name}", status_code=303)


@router.get("/api/judges")
async def judges_api(request: Request) -> list[dict[str, Any]]:
    services = services_of(request)
    out = []
    for judge in services.judges.values():
        versions = await calibrate.versions(services.store, judge.name)
        out.append(
            {
                "name": judge.name,
                "profile": judge.profile,
                "version": judge.version,
                "label_question": judge.label_question,
                "versions": [{"version": v.version, "trusted": v.trusted, "agreement": v.agreement} for v in versions],
            }
        )
    return out


# Labelling ------------------------------------------------------------------------------------------------------------


async def _card(services: Services, question_name: str, run_id: str | None) -> dict[str, Any]:
    known = questions(services.judges)
    if question_name not in known:
        raise HTTPException(404, f"no label question {question_name!r} (known: {', '.join(known)})")
    question = known[question_name]
    judge = services.judges[question.judge]
    todo = await queue(services.store, question, judge)
    if run_id is None:
        run_id = todo[0] if todo else None
    context: dict[str, Any] = {"question": question, "remaining": len(todo), "questions": list(known.values())}
    if run_id is None:
        return context
    run = await services.store.reader.find_run(run_id)
    if run is None:
        raise HTTPException(404, f"no run {run_id}")
    ctx = await load_run_context(services.store, run)
    profile = services.profiles.find(run.profile)
    output = profile.read_output(ctx) if profile else ctx.output_json
    position = todo.index(run.id) if run.id in todo else -1
    context.update(
        run=run,
        hits=last_search_hits(ctx) or [],
        answer=(output.get("answer") if isinstance(output, dict) else output) or run.output_text,
        next_id=todo[position + 1]
        if 0 <= position < len(todo) - 1
        else (todo[0] if todo and todo[0] != run.id else None),
        previous_id=todo[position - 1] if position > 0 else None,
        existing=await label_of(services.store, run.id, question.name),
    )
    return context


async def _verdict(services: Services, run_id: str, judge_name: str) -> Score | None:
    judge = services.judges[judge_name]
    async with services.store.read() as s:
        query = (
            select(Score)
            .where(
                Score.run_id == run_id, Score.kind == "judge", Score.name == judge_name, Score.version == judge.version
            )
            .order_by(Score.created_ms.desc())
        )
        return (await s.execute(query.limit(1))).scalar_one_or_none()


@router.get("/label", response_class=HTMLResponse, include_in_schema=False)
async def label_page(request: Request, question: str = "faithful", run: str | None = None) -> HTMLResponse:
    services = services_of(request)
    context = await _card(services, question, run)
    template = "_label_card.html" if request.headers.get("hx-request") else "label.html"
    return render(request, template, revealed=False, **context)


@router.post("/label", response_class=HTMLResponse, include_in_schema=False)
async def label_save(request: Request) -> HTMLResponse:
    services = services_of(request)
    form = await request.form()
    run_id, question, value = str(form.get("run_id")), str(form.get("question")), str(form.get("value"))
    if value not in VALUES:
        raise HTTPException(422, f"value must be one of {', '.join(VALUES)}")
    note = str(form.get("note") or "") or None
    existing = await label_of(services.store, run_id, question)
    if note is None and existing is not None and existing.value == value:
        note = existing.note
    await save_label(services.store, run_id, question, value, note)
    context = await _card(services, question, run_id)
    verdict = await _verdict(services, run_id, context["question"].judge)
    return render(request, "_label_card.html", revealed=True, verdict=verdict, saved=value, **context)


@router.get("/label/note", response_class=HTMLResponse, include_in_schema=False)
async def label_note(request: Request, run: str, question: str) -> HTMLResponse:
    services = services_of(request)
    existing = await label_of(services.store, run, question)
    return render(request, "_label_note.html", run_id=run, question=question, existing=existing)
