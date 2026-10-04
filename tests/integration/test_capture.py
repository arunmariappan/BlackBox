"""A fake agent using the SDK, run against a whole in-process BlackBox over real sockets."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request

from blackbox import sdk
from blackbox.server import Running, start_server
from blackbox.store.models import Run
from tests.harness import running_blackbox
from tests.profiles import FakeAgentProfile


class FakeChatClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, *, model: str, messages: list[Any], tools: Any = None, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls % 2 == 1:
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "get_logs", "arguments": {"service": "cache"}}}],
            }
        else:
            message = {"role": "assistant", "content": "The cache is down."}
        return {"model": model, "message": message, "prompt_eval_count": 50, "eval_count": 5}


def fake_agent_app() -> FastAPI:
    app = FastAPI()
    client = FakeChatClient()

    @sdk.tool
    def get_logs(service: str) -> dict[str, Any]:
        return {"service": service, "lines": ["connection refused to cache:6379"]}

    @app.post("/run")
    async def run(request: Request) -> dict[str, Any]:
        body = await request.json()
        with sdk.agent_run("fake-agent", input=body, headers=dict(request.headers)) as run:
            messages: list[dict[str, Any]] = [{"role": "user", "content": body["instruction"]}]
            sdk.traced_chat(client, model="fake-model", messages=messages)
            result = get_logs(service="cache")
            messages.append({"role": "tool", "content": str(result)})
            reply = sdk.traced_chat(client, model="fake-model", messages=messages)
            output = {"final_answer": reply["message"]["content"], "ending": "finished"}
            run.set_output(output)
            run.set_ending("finished")
        return output

    return app


@pytest.fixture
async def stack(tmp_path: Path, migrated_db: Path) -> AsyncIterator[tuple[Running, str]]:
    agent_server, agent_task, agent_port = await start_server(fake_agent_app(), "127.0.0.1", 0, "fake-agent")
    agent_url = f"http://127.0.0.1:{agent_port}"
    async with running_blackbox(tmp_path, migrated_db, profiles=[FakeAgentProfile(agent_url)]) as running:
        sdk.init("fake-agent", endpoint=running.base_url, instrument_httpx=False, batch_delay_ms=50)
        try:
            yield running, agent_url
        finally:
            await asyncio.to_thread(sdk.shutdown)
            agent_server.should_exit = True
            await agent_task


async def wait_complete(running: Running, run_id: str, within: float = 10) -> Run:
    async with asyncio.timeout(within):
        while True:
            await asyncio.to_thread(sdk.flush)
            run = await running.services.store.reader.run(run_id)
            if run is not None and run.status != "open":
                return run
            await asyncio.sleep(0.1)


async def test_started_run_is_recorded(stack: tuple[Running, str]) -> None:
    running, _ = stack
    async with httpx.AsyncClient(base_url=running.base_url) as client:
        response = await client.post(
            "/api/runs",
            json={"profile": "fake-agent", "input": {"instruction": "Why is checkout failing?"}, "wait": True},
        )
        assert response.status_code == 200, response.text
        started = response.json()
        assert started["status"] == 200
        run = await wait_complete(running, started["run_id"])
        assert run.trace_id == started["trace_id"]
        assert (run.profile, run.source, run.status, run.ending) == ("fake-agent", "live", "complete", "finished")
        assert run.input_text == "Why is checkout failing?"
        assert run.output_text == "The cache is down."
        assert (run.step_count, run.input_tokens, run.output_tokens) == (3, 100, 10)
        spans = await running.services.store.reader.spans(run.trace_id)
        root = next(s for s in spans if s.name == "invoke_agent fake-agent")
        assert root.parent_span_id == run.remote_parent_span_id  # the agent continued BlackBox's traceparent
        steps = await running.services.store.reader.steps(run.id)
        assert [(s.kind, s.tool_name) for s in steps] == [("llm", None), ("tool", "get_logs"), ("llm", None)]

        api_run = (await client.get(f"/api/runs/{run.id}")).json()
        assert api_run["output"] == {"final_answer": "The cache is down.", "ending": "finished"}
        assert api_run["entry_request"]["body"] == {"instruction": "Why is checkout failing?"}
        assert len((await client.get(f"/api/runs/{run.id}/steps")).json()) == 3
        assert len((await client.get(f"/api/runs/{run.id}/spans")).json()) == 4
        assert [r["id"] for r in (await client.get("/api/runs?profile=fake-agent")).json()] == [run.id]

        for tab in ("steps", "waterfall", "spans"):
            page = await client.get(f"/runs/{run.id}?tab={tab}")
            assert page.status_code == 200
            assert "The cache is down." in page.text
        assert "get_logs" in (await client.get(f"/runs/{run.id}?tab=steps")).text
        listing = await client.get("/runs")
        assert listing.status_code == 200 and run.id in listing.text
        assert run.id in (await client.get("/runs/rows?profile=fake-agent")).text
        health = (await client.get("/health")).json()
        assert health["status"] == "ok"


async def test_run_the_agent_was_asked_directly_also_appears(stack: tuple[Running, str]) -> None:
    running, agent_url = stack
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{agent_url}/run", json={"instruction": "Is search healthy?"})
    assert response.status_code == 200
    async with asyncio.timeout(10):
        while True:
            await asyncio.to_thread(sdk.flush)
            runs = await running.services.store.reader.runs(status="complete")
            if runs:
                break
            await asyncio.sleep(0.1)
    run = runs[0]
    assert (run.profile, run.ending, run.step_count) == ("fake-agent", "finished", 3)
    assert run.input_text is not None and "Is search healthy?" in run.input_text
    assert run.output_text == "The cache is down."


async def test_sse_stream_reports_runs(stack: tuple[Running, str]) -> None:
    running, _ = stack
    received: list[str] = []
    async with (
        httpx.AsyncClient(base_url=running.base_url, timeout=10) as client,
        client.stream("GET", "/api/events") as response,
    ):

        async def read() -> None:
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    received.append(line.split(":", 1)[1].strip())
                    if "run" in received:
                        return

        reader = asyncio.create_task(read())
        await asyncio.sleep(0.2)
        await client.post("/api/runs", json={"profile": "fake-agent", "input": {"instruction": "x"}})
        await asyncio.wait_for(reader, 5)
    assert "run" in received
