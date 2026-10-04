"""Live scoring against a real store: the alert lifecycle, sampling and its budget, the model lane's wait, durable
jobs, live patches at the proxy, markers, and the whole fan-out from a completed run to an alert."""

import asyncio
import inspect
import json
import shutil
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.config import (
    AlertsConfig,
    LiveConfig,
    OllamaConfig,
    RateDropConfig,
    SamplingConfig,
    UpstreamConfig,
)
from blackbox.live.pipeline import ModelLaneGate, live_after_complete
from blackbox.live.sampling import is_sampled, plan_judges, sample_point
from blackbox.live.telegram import Telegram
from blackbox.live.worker import JobView, Worker, enqueue
from blackbox.net import new_span_id, new_trace_id, traceparent
from blackbox.services import Services
from blackbox.store.models import Alert, Exchange, Job, LivePatch, Run, Score
from blackbox.store.models import Judge as JudgeRow
from blackbox.util import new_id, now_ms
from blackbox.web.app import create_app
from tests.factories import trace_id
from tests.fakes import FakeUpstream
from tests.harness import make_settings, running_blackbox

BASE = 1_800_000_000_000


@pytest.fixture
async def services(tmp_path: Path, migrated_db: Path, request: pytest.FixtureRequest) -> AsyncIterator[Services]:
    db = tmp_path / "blackbox.db"
    shutil.copy(migrated_db, db)
    extra = getattr(request, "param", {})
    s = await Services.create(make_settings(db, **extra), migrate=False)
    try:
        yield s
    finally:
        if s.worker is not None:
            await s.worker.stop()
        await s.close()


async def add_scored_run(services: Services, t_ms: int, passed: bool, **fields: Any) -> Run:
    run = Run(
        id=new_id(),
        trace_id=fields.pop("trace_id", None) or trace_id(),
        profile="opsdesk",
        status="complete",
        source="live",
        started_ms=t_ms - 1000,
        ended_ms=t_ms,
        updated_ms=t_ms,
        duration_ms=1000,
        step_count=3,
        ending=fields.pop("ending", "finished"),
        input_text=fields.pop("input_text", "restart the search indexer"),
        **fields,
    )

    async def op(session: AsyncSession) -> None:
        session.add(run)
        session.add(
            Score(
                id=new_id(),
                run_id=run.id,
                kind="checker",
                name="checker",
                version="1",
                label="pass" if passed else "fail",
                value=1.0 if passed else 0.0,
                details={},
                created_ms=t_ms,
            )
        )

    await services.store.write(op)
    return run


async def trust(services: Services, name: str) -> None:
    judge = services.judges[name]

    async def op(session: AsyncSession) -> None:
        session.add(
            JudgeRow(
                name=judge.name,
                version=judge.version,
                prompt_hash=judge.prompt_hash,
                model=judge.model,
                options=judge.options,
                created_ms=now_ms(),
                trusted=True,
            )
        )

    await services.store.write(op)


async def wait_for(predicate: Any, within: float = 10.0) -> None:
    """Poll `predicate` (sync or async) until it holds."""
    async with asyncio.timeout(within):
        while True:
            result = predicate()
            if inspect.isawaitable(result):
                result = await result
            if result:
                return
            await asyncio.sleep(0.02)


# Alerts ---------------------------------------------------------------------------------------------------------------


async def test_alert_lifecycle(services: Services) -> None:
    assert services.live is not None
    engine = services.live.engine
    sent: list[dict[str, Any]] = []

    def telegram(request: httpx.Request) -> httpx.Response:
        sent.append({"url": str(request.url), **json.loads(request.content)})
        return httpx.Response(200, json={"ok": True})

    config = AlertsConfig(telegram_bot_token="123:secret-token", telegram_chat_id="42")  # type: ignore[arg-type]
    engine.telegram = Telegram(config, httpx.AsyncClient(transport=httpx.MockTransport(telegram)))
    clock = [BASE]
    engine.clock = lambda: clock[0]
    n = [0]

    async def run(passed: bool, gap_ms: int = 60_000) -> list[str]:
        n[0] += 1
        clock[0] += gap_ms
        await add_scored_run(services, clock[0], passed)
        return [f"{t.change}:{t.rule}" for t in await engine.evaluate("opsdesk")]

    for _ in range(37):
        assert await run(True) == []
    for _ in range(30):
        assert await run(True) == []
    opened = []
    for _ in range(12):
        opened += [c for c in await run(False) if c.startswith("opened")]
    assert opened == ["opened:rate_drop:pass_rate"]  # once, however many evaluations stay in alarm
    async with services.store.read() as s:
        [alert] = (await s.execute(select(Alert))).scalars().all()
    assert alert.status == "open" and alert.details["shown"]["baseline"] > 0.9
    assert alert.details["shown"]["current"] < alert.details["shown"]["baseline"]
    assert alert.details["window"]["drop_started_run"] and alert.details["latest"]["shown"]["current"] < 0.7
    banners = services.live.banners()
    assert banners and banners[0]["kind"] == "alert" and "pass rate dropped" in banners[0]["html"]

    changes: list[str] = []
    for _ in range(40):
        changes += [c for c in await run(True) if "pass_rate" in c]
        if any(c.startswith("resolved") for c in changes):
            break
    assert changes[-2:] == ["cleared_once:rate_drop:pass_rate", "resolved:rate_drop:pass_rate"]
    assert services.live.banners() == []

    for _ in range(12):  # the same drop again, within the 30-minute cooldown: no new alert
        assert not [c for c in await run(False) if c.startswith("opened")]
    reopened = await run(False, gap_ms=31 * 60_000)
    assert "opened:rate_drop:pass_rate" in reopened
    async with services.store.read() as s:
        alerts = (await s.execute(select(Alert).order_by(Alert.opened_ms))).scalars().all()
    assert [a.status for a in alerts] == ["resolved", "open"]

    await asyncio.sleep(0.2)  # Telegram sends in the background
    assert sent and sent[0]["chat_id"] == "42" and "pass rate dropped" in sent[0]["text"]
    assert any("resolved" in m["text"] for m in sent)
    async with services.store.read() as s:
        assert (await s.get(Alert, alerts[0].id)).notified is True  # type: ignore[union-attr]


async def test_short_history_raises_nothing(services: Services) -> None:
    assert services.live is not None
    for i in range(20):
        await add_scored_run(services, BASE + i * 60_000, passed=i < 5)
    assert await services.live.engine.evaluate("opsdesk") == []


async def test_telegram_token_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG")
    config = AlertsConfig(telegram_bot_token="123456:AAH-very-secret", telegram_chat_id="7")  # type: ignore[arg-type]
    ok = Telegram(config, httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))))
    assert await ok.send("hello") is True
    down = Telegram(config, httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(401))))
    assert await down.send("hello") is False

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url}")

    broken = Telegram(config, httpx.AsyncClient(transport=httpx.MockTransport(refuse)))
    assert await broken.send("hello") is False
    assert "AAH-very-secret" not in caplog.text and "HTTP 401" in caplog.text
    assert Telegram(AlertsConfig()).enabled is False


# Sampling -------------------------------------------------------------------------------------------------------------


def test_sampling_is_reproducible() -> None:
    ids = [trace_id() for _ in range(4000)]
    assert [is_sampled(t, 0.3) for t in ids] == [is_sampled(t, 0.3) for t in ids]
    share = sum(is_sampled(t, 0.3) for t in ids) / len(ids)
    assert 0.27 < share < 0.33
    assert not any(is_sampled(t, 0.0) for t in ids) and all(is_sampled(t, 1.0) for t in ids)
    assert sample_point("ab" * 16) == sample_point("ab" * 16)


@pytest.mark.parametrize(
    "services",
    [{"live": LiveConfig(sampling={"opsdesk": SamplingConfig(judge_rate=0.5, max_judge_calls_per_hour=10)})}],
    indirect=True,
)
async def test_sampling_respects_the_hourly_budget(services: Services) -> None:
    run = await add_scored_run(services, BASE, True)
    assert (await plan_judges(services, run)).reason == "no_trusted_judges"  # untrusted judges never run live
    await trust(services, "ops_task_success")
    unsampled = next(t for t in (trace_id() for _ in range(100)) if not is_sampled(t, 0.5))
    flagged = await add_scored_run(services, BASE, True, trace_id=unsampled, ending="max_steps")
    assert (await plan_judges(services, flagged)).reason == "flagged"
    runs = [await add_scored_run(services, BASE + i, True) for i in range(40)]
    for r in runs:
        await live_after_complete(services, r)
    async with services.store.read() as s:
        jobs = (await s.execute(select(Job).where(Job.kind == "judge").order_by(Job.created_ms))).scalars().all()
    sampled = [r.id for r in runs if is_sampled(r.trace_id, 0.5)]
    assert len(sampled) > 10
    assert [j.run_id for j in jobs] == sampled[:10]  # one call each, ten calls an hour
    assert all(j.lane == "model" and j.payload["judges"] == ["ops_task_success"] for j in jobs)
    assert (await plan_judges(services, runs[-1])).reason in ("over_budget", "not_sampled")


# Worker ---------------------------------------------------------------------------------------------------------------


async def test_model_lane_waits_for_calls_in_flight(services: Services) -> None:
    worker = Worker(services, poll_seconds=0.02)
    services.worker = worker
    ran: list[float] = []

    async def probe(_: Services, job: JobView) -> None:
        ran.append(time.monotonic())

    worker.register("probe", probe)
    in_flight = [1]
    gate = ModelLaneGate(services, in_flight=lambda: in_flight[0], max_wait_seconds=0.6)
    worker.gates["model"] = gate
    await worker.start()
    start = time.monotonic()
    await enqueue(services, "probe", lane="model")
    await enqueue(services, "probe", lane="cpu")  # the CPU lane never waits
    await asyncio.sleep(0.35)
    assert len(ran) == 1
    await wait_for(lambda: len(ran) >= 2, within=5)
    assert ran[1] - start >= 0.55 and gate.forced == 1  # ran anyway after the time limit
    in_flight[0] = 0
    queued = time.monotonic()
    await enqueue(services, "probe", lane="model")
    await wait_for(lambda: len(ran) >= 3, within=5)
    assert ran[2] - queued < 0.4


async def test_a_job_failing_three_times_ends_up_failed(services: Services) -> None:
    worker = Worker(services, poll_seconds=0.02, backoff_ms=10)
    services.worker = worker
    attempts: list[int] = []
    gave_up: list[str] = []

    async def boom(_: Services, job: JobView) -> None:
        attempts.append(job.attempts)
        raise RuntimeError("model unreachable")

    async def give_up(_: Services, job: JobView) -> None:
        gave_up.append(job.id)

    worker.register("boom", boom)
    worker.on_give_up["boom"] = give_up
    await worker.start()
    job_id = await enqueue(services, "boom", lane="cpu")

    async def failed() -> bool:
        async with services.store.read() as s:
            job = await s.get(Job, job_id)
        return job is not None and job.status == "failed"

    await wait_for(failed)
    assert attempts == [1, 2, 3] and gave_up == [job_id]
    app = create_app(services)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://bb") as client:
        page = await client.get("/jobs")
        assert "boom" in page.text and "model unreachable" in page.text
        assert (await client.post(f"/jobs/{job_id}/retry")).status_code == 303
    await wait_for(lambda: len(attempts) >= 4, within=5)
    assert attempts[3] == 1  # retried from the list: a fresh set of attempts


async def test_jobs_survive_a_restart(services: Services) -> None:
    started = asyncio.Event()

    async def hang(_: Services, job: JobView) -> None:
        started.set()
        await asyncio.sleep(3600)

    first = Worker(services, poll_seconds=0.02)
    services.worker = first
    first.register("work", hang)
    await first.start()
    job_id = await enqueue(services, "work")
    await asyncio.wait_for(started.wait(), 5)
    await first.stop()  # BlackBox stops mid-job
    async with services.store.read() as s:
        assert (await s.get(Job, job_id)).status == "running"  # type: ignore[union-attr]

    done: list[str] = []

    async def finish(_: Services, job: JobView) -> None:
        done.append(job.id)

    second = Worker(services, poll_seconds=0.02)
    services.worker = second
    second.register("work", finish)
    await second.start()

    async def finished() -> bool:
        async with services.store.read() as s:
            job = await s.get(Job, job_id)
        return job is not None and job.status == "done"

    await wait_for(finished)
    assert done == [job_id]


# Live patches and markers ---------------------------------------------------------------------------------------------

PATCH = """
patches:
  - name: strict-guardrail
    match: { upstream: ollama, content_regex: "relevance of the question" }
    edit:
      path: "$.messages[0].content"
      replace: { find: "Score from 0 to 100", with: "Be very strict. Score from 0 to 100" }
"""


async def test_live_patch_changes_only_matching_requests_until_it_expires(tmp_path: Path, migrated_db: Path) -> None:
    upstream = await FakeUpstream().start()
    upstream.release.set()
    config = UpstreamConfig(name="ollama", listen_port=0, target=upstream.url, record_paths=["/api/chat"])
    try:
        async with running_blackbox(tmp_path, migrated_db, upstreams=[config], keep_unmatched=True) as bb:
            assert bb.services.proxy is not None and bb.services.live is not None
            proxy = f"http://127.0.0.1:{bb.services.proxy.ports['ollama']}"

            async def chat(client: httpx.AsyncClient, content: str) -> str:
                tid = new_trace_id()
                body = {"model": "m", "messages": [{"role": "user", "content": content}], "stream": False}
                response = await client.post(
                    f"{proxy}/api/chat", json=body, headers={"traceparent": traceparent(tid, new_span_id())}
                )
                assert response.status_code == 200
                return tid

            async with httpx.AsyncClient(base_url=bb.base_url) as client:
                bad = await client.post("/api/live-patches", json={"yaml": PATCH.replace("content_regex", "tape_node")})
                assert bad.status_code == 400
                added = await client.post("/api/live-patches", json={"yaml": PATCH, "minutes": 0.02})
                assert added.status_code == 200 and added.json()[0]["name"] == "strict-guardrail"
                page = await client.get("/runs")
                assert "Live patch" in page.text and "strict-guardrail" in page.text
                matching = await chat(client, "Judge the relevance of the question. Score from 0 to 100.")
                other = await chat(client, "Summarise the paper. Score from 0 to 100.")
                seen = [c["messages"][0]["content"] for c in upstream.chat_calls()]
                assert seen == [
                    "Judge the relevance of the question. Be very strict. Score from 0 to 100.",
                    "Summarise the paper. Score from 0 to 100.",
                ]

                async def recorded(tid: str) -> Exchange:
                    rows: list[Exchange] = []

                    async def found() -> bool:
                        rows[:] = await bb.services.store.reader.exchanges(tid)
                        return bool(rows)

                    await wait_for(found, within=5)
                    return rows[0]

                patched, untouched = await recorded(matching), await recorded(other)
                assert patched.served_from == "patched" and patched.sent_request_blob is not None
                assert untouched.served_from == "live" and untouched.sent_request_blob is None
                listed = (await client.get("/api/live-patches")).json()
                assert listed[0]["hits"] == 1

                await asyncio.sleep(1.3)  # past its 0.02 minutes
                await chat(client, "Judge the relevance of the question. Score from 0 to 100.")
                assert "Be very strict" not in upstream.chat_calls()[-1]["messages"][0]["content"]
                assert (await client.get("/api/live-patches")).json() == []
                await bb.services.live.patches.expire_due()
                async with bb.services.store.read() as s:
                    [row] = (await s.execute(select(LivePatch))).scalars().all()
                assert row.status == "expired" and row.hits == 1
                assert "Live patch" not in (await client.get("/runs")).text

                added = await client.post("/api/live-patches", json={"yaml": PATCH})
                assert (await client.delete("/api/live-patches/strict-guardrail")).status_code == 200
                assert (await client.get("/api/live-patches")).json() == []
    finally:
        await upstream.stop()


async def test_markers_api_and_alert_note(services: Services) -> None:
    assert services.live is not None
    app = create_app(services)
    engine = services.live.engine
    clock = [BASE]
    engine.clock = lambda: clock[0]
    for _ in range(60):
        clock[0] += 60_000
        await add_scored_run(services, clock[0], True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://bb") as client:
        marker = await client.post("/api/markers", json={"profile": "opsdesk", "text": "new guardrail prompt"})
        assert marker.status_code == 200
        await _move_marker(services, marker.json()["id"], clock[0] + 30_000)  # onto the test's timeline
        for i in range(12):
            clock[0] += 60_000
            await add_scored_run(services, clock[0], passed=i < 3)
            await engine.evaluate("opsdesk")
        listed = (await client.get("/api/markers?profile=opsdesk")).json()
        assert [m["text"] for m in listed] == ["new guardrail prompt"]
        alerts = (await client.get("/api/alerts?status=open")).json()
        assert alerts and alerts[0]["rule"] == "rate_drop:pass_rate"
        notes = [m.get("note", "") for m in alerts[0]["details"]["markers"]]
        assert notes and notes[0].startswith("the drop started 3 runs after marker")
        detail = await client.get(f"/alerts/{alerts[0]['id']}")
        assert "new guardrail prompt" in detail.text and "Baseline" in detail.text
        live = await client.get("/live?profile=opsdesk")
        assert live.status_code == 200 and "pass rate" in live.text and "alarm" in live.text
        assert (await client.get("/alerts")).status_code == 200
        resolved = await client.post(f"/api/alerts/{alerts[0]['id']}/resolve")
        assert resolved.status_code == 200 and resolved.json()["status"] == "resolved"


async def _move_marker(services: Services, marker_id: str, created_ms: int) -> None:
    from blackbox.store.models import Marker

    async def op(session: AsyncSession) -> None:
        marker = await session.get(Marker, marker_id)
        assert marker is not None
        marker.created_ms = created_ms

    await services.store.write(op)


# The whole fan-out ----------------------------------------------------------------------------------------------------


def verdicts(body: dict[str, Any]) -> dict[str, Any]:
    prompt = body["messages"][-1]["content"]
    if "Categories:" in prompt:  # a failure description
        content = {"what_went_wrong": "The agent broke the task", "where_step": 1, "category": "other"}
    else:
        verdict = "fail" if "BREAK" in prompt else "pass"
        content = {"problems": [], "rationale": "checked", "verdict": verdict}
    return {"role": "assistant", "content": json.dumps(content)}


async def test_completed_runs_are_judged_and_raise_an_alert(tmp_path: Path, migrated_db: Path) -> None:
    model = await FakeUpstream(reply=verdicts).start()
    model.release.set()
    live = LiveConfig(
        sampling={"opsdesk": SamplingConfig(judge_rate=1.0, max_judge_calls_per_hour=1000)},
        rate_drop=RateDropConfig(baseline_runs=30, min_baseline=12, window=8, min_current=4, threshold=4.0),
    )
    try:
        async with running_blackbox(tmp_path, migrated_db, ollama=OllamaConfig(base_url=model.url), live=live) as bb:
            services = bb.services
            await trust(services, "ops_task_success")
            runs = []
            for i in range(20):
                text = "BREAK the payments database" if i >= 14 else "restart the search indexer"
                run = Run(
                    id=new_id(),
                    trace_id=trace_id(),
                    profile="opsdesk",
                    status="complete",
                    source="traffic",
                    started_ms=BASE + i * 60_000,
                    ended_ms=BASE + i * 60_000 + 1000,
                    updated_ms=BASE + i * 60_000 + 1000,
                    duration_ms=1000,
                    ending="finished",
                    input_text=text,
                )

                async def op(session: AsyncSession, run: Run = run) -> None:
                    session.add(run)

                await services.store.write(op)
                runs.append(run)
                await enqueue(services, "run_completed", run_id=run.id)

            async def judged_and_detected() -> bool:
                async with services.store.read() as s:
                    scores = (await s.execute(select(Score).where(Score.kind == "judge"))).scalars().all()
                    busy = (await s.execute(select(Job).where(Job.status.in_(("queued", "running"))))).scalars().all()
                return len(scores) == 20 and not busy

            await wait_for(judged_and_detected, within=60)
            async with services.store.read() as s:
                jobs = (await s.execute(select(Job))).scalars().all()
                alerts = (await s.execute(select(Alert))).scalars().all()
                labels = {
                    sc.run_id: sc.label
                    for sc in (await s.execute(select(Score).where(Score.kind == "judge"))).scalars()
                }
            kinds = {(j.kind, j.lane) for j in jobs}
            assert {("run_completed", "cpu"), ("judge", "model"), ("detect", "cpu")} <= kinds
            assert ("describe_failure", "model") in kinds  # trusted judges said fail
            assert not [j for j in jobs if j.status == "failed"]
            assert [labels[r.id] for r in runs] == ["pass"] * 14 + ["fail"] * 6
            assert {a.rule for a in alerts} >= {"rate_drop:pass_rate"}
            async with httpx.AsyncClient(base_url=bb.base_url) as client:
                page = await client.get("/runs")
                assert "Alert:" in page.text
    finally:
        await model.stop()
