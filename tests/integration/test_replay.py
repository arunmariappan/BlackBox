"""Replays, forks and auto-forks of a fake agent whose model calls go through the proxy to a counting fake model."""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from blackbox import sdk
from blackbox.config import UpstreamConfig
from blackbox.net import make_client, new_span_id, traceparent
from blackbox.proxy.sessions import SessionSpec
from blackbox.replay.runner import prepare_replay, session_manager
from blackbox.server import Running, start_server
from blackbox.store.models import Run
from tests.fake_agent import AgentConfig, deterministic_reply, make_agent_app
from tests.fakes import FakeUpstream
from tests.harness import running_blackbox
from tests.profiles import FakeAgentProfile


@dataclass
class Stack:
    bb: Running
    model: FakeUpstream
    agent: AgentConfig
    profile: FakeAgentProfile

    @property
    def proxy_url(self) -> str:
        assert self.bb.services.proxy is not None
        return f"http://127.0.0.1:{self.bb.services.proxy.ports['ollama']}"


@pytest.fixture
async def stack(tmp_path: Path, migrated_db: Path) -> AsyncIterator[Stack]:
    model = await FakeUpstream(reply=deterministic_reply).start()
    model.release.set()
    profile = FakeAgentProfile()
    upstream = UpstreamConfig(name="ollama", listen_port=0, target=model.url, record_paths=["/api/chat"])
    async with running_blackbox(tmp_path, migrated_db, profiles=[profile], upstreams=[upstream]) as bb:
        assert bb.services.proxy is not None
        config = AgentConfig(model_url=f"http://127.0.0.1:{bb.services.proxy.ports['ollama']}")
        server, task, port = await start_server(make_agent_app(config), "127.0.0.1", 0, "fake-agent")
        profile.options["base_url"] = f"http://127.0.0.1:{port}"
        sdk.init("fake-agent", endpoint=bb.base_url, batch_delay_ms=50)
        try:
            yield Stack(bb, model, config, profile)
        finally:
            await asyncio.to_thread(sdk.shutdown)
            server.should_exit = True
            await task
    await model.stop()


async def wait_run(stack: Stack, run_id: str, within: float = 15) -> Run:
    async with asyncio.timeout(within):
        while True:
            run = await stack.bb.services.store.reader.run(run_id)
            if run is not None and run.status != "open":
                return run
            await asyncio.sleep(0.05)


async def record(stack: Stack, question: str = "Why is checkout failing?") -> Run:
    async with httpx.AsyncClient(base_url=stack.bb.base_url, timeout=30) as client:
        response = await client.post(
            "/api/runs", json={"profile": "fake-agent", "input": {"question": question}, "wait": True}
        )
    assert response.status_code == 200, response.text
    run = await wait_run(stack, response.json()["run_id"])
    assert run.status == "complete" and run.replayable and run.step_count == 4, (run.status, run.step_count)
    return run


async def replay(stack: Stack, run_id: str, **spec: Any) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=stack.bb.base_url, timeout=60) as client:
        response = await client.post("/api/sessions", json={"source_run_id": run_id, "wait": True, **spec})
    assert response.status_code == 200, response.text
    report: dict[str, Any] = response.json()["report"]
    assert "steps" in report, report
    return report


async def test_exact_replay_is_identical_and_never_calls_the_model(stack: Stack) -> None:
    source = await record(stack)
    calls = len(stack.model.chat_calls())
    report = await replay(stack, source.id)
    assert report["exact"] is True
    assert [s["served"] for s in report["steps"]] == ["tape"] * 4
    assert [s["tape_step"] for s in report["steps"]] == [1, 2, 3, 4]
    assert report["outputs"]["equal"] and report["live_calls"] == 0
    assert len(stack.model.chat_calls()) == calls  # nothing reached the model
    replay_run = await stack.bb.services.store.reader.run(report["replay_run"])
    assert replay_run is not None
    assert (replay_run.source, replay_run.replay_of, replay_run.ending) == ("replay", source.id, "finished")
    async with httpx.AsyncClient(base_url=stack.bb.base_url) as client:
        page = await client.get(f"/sessions/{report['session_id']}")
        assert page.status_code == 200 and "EXACT" in page.text
        assert (await client.get("/sessions")).status_code == 200
        assert (await client.get(f"/runs/{source.id}")).text.count("fork from here") == 4
        form = await client.get(f"/runs/{source.id}/fork?step=2")
        assert form.status_code == 200 and "You are step 2." in form.text


async def test_changed_prompt_diverges_in_exact_and_auto_forks(stack: Stack) -> None:
    source = await record(stack)
    stack.agent.prompts[2] = "You are step 3. Be careful."  # the agent's code changed at its third call
    exact = await replay(stack, source.id)
    assert exact["exact"] is False
    divergence = exact["first_divergence"]
    assert divergence["step"] == 3 and divergence["node"] == "act"
    [message] = divergence["diff"]["messages"]
    assert message["role"] == "system" and "+You are step 3. Be careful." in message["diff"]
    assert [s["served"] for s in exact["steps"]] == ["tape", "tape", "blocked"]  # the agent failed at the 409

    calls = len(stack.model.chat_calls())
    forked = await replay(stack, source.id, mode="auto_fork")
    assert [s["served"] for s in forked["steps"]] == ["tape", "tape", "live", "live"]
    live = stack.model.chat_calls()[calls:]
    assert [c["messages"][0]["content"] for c in live] == ["You are step 3. Be careful.", "You are step 4."]
    assert forked["live_calls"] == 2 and forked["outputs"]["equal"] is False


async def test_fork_with_model_override(stack: Stack) -> None:
    source = await record(stack)
    calls = len(stack.model.chat_calls())
    report = await replay(stack, source.id, mode="fork", fork_step=2, model="qwen3.5:9b")
    assert [s["served"] for s in report["steps"]] == ["tape", "live", "live", "live"]
    live = stack.model.chat_calls()[calls:]
    assert [c["model"] for c in live] == ["qwen3.5:9b"] * 3
    assert report["steps"][1]["differs_from_tape"] is True
    replay_run = await stack.bb.services.store.reader.run(report["replay_run"])
    assert replay_run is not None
    exchanges = await stack.bb.services.store.reader.exchanges(replay_run.trace_id)
    assert exchanges[1].sent_request_blob is not None  # the agent's own body and the overridden one both kept
    assert json.loads(await stack.bb.services.store.blobs.get(exchanges[1].request_blob or ""))["model"] == "qwen3.5:4b"


@pytest.mark.parametrize(
    "match",
    [{"tape_node": "answer"}, {"step": 4}, {"content_regex": "You are step 4"}],
    ids=["tape_node", "step", "content_regex"],
)
async def test_patches_change_only_the_intended_request(stack: Stack, match: dict[str, Any]) -> None:
    source = await record(stack)
    patch = {
        "name": "rewrite-answer-prompt",
        "match": {"upstream": "ollama", **match},
        "edit": {
            "path": "$.messages[?(@.role == 'system')].content",
            "replace": {"find": "step 4", "with": "the final step"},
        },
    }
    calls = len(stack.model.chat_calls())
    report = await replay(stack, source.id, mode="auto_fork", patches=[patch])
    assert [s["served"] for s in report["steps"]] == ["tape", "tape", "tape", "patched"]
    [live] = stack.model.chat_calls()[calls:]
    assert live["messages"][0]["content"] == "You are the final step."


async def test_exact_with_patches_is_refused(stack: Stack) -> None:
    source = await record(stack)
    patch = {"name": "p", "edit": {"path": "$.model", "set": "x"}}
    async with httpx.AsyncClient(base_url=stack.bb.base_url) as client:
        response = await client.post("/api/sessions", json={"source_run_id": source.id, "patches": [patch]})
    assert response.status_code == 400 and "auto_fork" in response.text


async def test_timestamp_needs_a_normaliser(stack: Stack) -> None:
    stack.agent.clock = "wall"
    source = await record(stack)
    await asyncio.sleep(0.01)
    without = await replay(stack, source.id)
    assert without["exact"] is False and without["first_divergence"]["step"] == 1
    stack.profile.options["normalisers"] = {"ollama": [{"mask": r"\d{4}-\d{2}-\d{2}T[0-9:.+]+", "with": "<ts>"}]}
    try:
        with_normaliser = await replay(stack, source.id)
    finally:
        stack.profile.options.pop("normalisers")
    assert with_normaliser["exact"] is True


async def test_two_sessions_at_once_keep_their_own_tapes(stack: Stack) -> None:
    first = await record(stack, "Why is checkout failing?")
    second = await record(stack, "Is search healthy?")
    a, b = await asyncio.gather(replay(stack, first.id), replay(stack, second.id))
    assert a["exact"] and b["exact"]
    assert a["source_run"] == first.id and b["source_run"] == second.id


async def send_tape_requests(stack: Stack, trace_id: str, bodies: list[bytes]) -> list[httpx.Response]:
    out = []
    # BlackBox's untraced client: the SDK instrumented httpx in this process and would replace our traceparent.
    async with make_client(timeout=10) as client:
        for body in bodies:
            out.append(
                await client.post(
                    f"{stack.proxy_url}/api/chat",
                    content=body,
                    headers={"traceparent": traceparent(trace_id, new_span_id()), "content-type": "application/json"},
                )
            )
    return out


async def test_expired_session_gets_410(stack: Stack) -> None:
    source = await record(stack)
    prepared = await prepare_replay(stack.bb.services, SessionSpec(source_run_id=source.id))
    session_manager(stack.bb.services).end(prepared.session.trace_id)
    [response] = await send_tape_requests(stack, prepared.session.trace_id, [b'{"model":"m","messages":[]}'])
    assert response.status_code == 410 and response.json()["error"] == "blackbox_session_expired"


async def test_streamed_tape_speed(stack: Stack) -> None:
    stack.agent.stream = True
    stack.model.chunks, stack.model.chunk_delay = 20, 0.02
    source = await record(stack)
    exchanges = await stack.bb.services.store.reader.exchanges(source.trace_id)
    recorded = exchanges[0]
    assert recorded.stream and len(recorded.chunk_times) >= 10
    recorded_ms = recorded.chunk_times[-1][1]
    request = await stack.bb.services.store.blobs.get(recorded.request_blob or "")
    original = await stack.bb.services.store.blobs.get(recorded.response_blob or "")
    for speed, check in ((1.0, lambda ms: abs(ms - recorded_ms) <= 0.2 * recorded_ms), (0.0, lambda ms: ms < 100)):
        prepared = await prepare_replay(stack.bb.services, SessionSpec(source_run_id=source.id, speed=speed))
        started = time.perf_counter()
        [response] = await send_tape_requests(stack, prepared.session.trace_id, [request])
        elapsed = (time.perf_counter() - started) * 1000
        assert response.content == original
        assert check(elapsed), (speed, elapsed, recorded_ms)
        assert [served.served for served in prepared.session.served] == ["tape"]


async def test_paperpilot_tape_serves_every_recorded_request(tmp_path: Path, migrated_db: Path) -> None:
    """Matching against PaperPilot-shaped bodies without running .NET: re-send the run's recorded requests."""
    from tests.fixtures import paperpilot

    model = await FakeUpstream().start()
    upstreams = [
        UpstreamConfig(name=name, listen_port=0, target=model.url, record_paths=["/*"])
        for name in ("ollama", "opensearch", "jina")
    ]
    try:
        async with running_blackbox(tmp_path, migrated_db, upstreams=upstreams) as bb:
            run_id = await paperpilot.insert_run(bb.services.store)
            prepared = await prepare_replay(bb.services, SessionSpec(source_run_id=run_id))
            assert prepared.rebuilt is False
            exchanges = await bb.services.store.reader.exchanges(paperpilot.TRACE_ID)
            assert bb.services.proxy is not None
            async with make_client(timeout=10) as client:
                for exchange in exchanges:
                    port = bb.services.proxy.ports[exchange.upstream]
                    body = await bb.services.store.blobs.get(exchange.request_blob or "")
                    response = await client.post(
                        f"http://127.0.0.1:{port}{exchange.path}",
                        content=body,
                        headers={"traceparent": traceparent(prepared.session.trace_id, new_span_id())},
                    )
                    assert response.status_code == 200
                    assert response.content == await bb.services.store.blobs.get(exchange.response_blob or "")
            assert all(s.served == "tape" for s in prepared.session.served)
            assert len(prepared.session.served) == len(exchanges) == 9
            assert model.calls == []
    finally:
        await model.stop()


async def test_rebuilt_input_for_a_run_paperpilot_was_asked_directly(tmp_path: Path, migrated_db: Path) -> None:
    from blackbox.profiles.paperpilot import PaperPilotProfile
    from blackbox.runs.context import load_run_context
    from tests.fixtures import paperpilot

    async with running_blackbox(tmp_path, migrated_db) as bb:
        run_id = await paperpilot.insert_run(bb.services.store, started_by_blackbox=False)
        run = await bb.services.store.reader.run(run_id)
        assert run is not None and run.profile == "paperpilot" and run.ending == "answered"
        ctx = await load_run_context(bb.services.store, run)
        request = PaperPilotProfile().rebuild_input(ctx)
        assert request is not None and request.url.endswith("/api/v1/ask-agentic")
        assert request.body == {"query": paperpilot.QUESTION, "top_k": 3, "use_hybrid": True, "model": "qwen3.5:4b"}


async def test_fork_form_turns_edited_messages_into_patches(stack: Stack) -> None:
    source = await record(stack)
    calls = len(stack.model.chat_calls())
    form = {
        "mode": "fork",
        "fork_step": "2",
        "upstream": "ollama",
        "original_0": "You are step 2.",
        "message_0": "You are step 2, and very thorough.",
        "original_1": "unchanged",
        "message_1": "unchanged",
    }
    async with httpx.AsyncClient(base_url=stack.bb.base_url, timeout=30) as client:
        response = await client.post(f"/runs/{source.id}/replay", data=form)
        assert response.status_code == 303
        session_id = response.headers["location"].rsplit("/", 1)[1]
        async with asyncio.timeout(20):
            while True:
                session = (await client.get(f"/api/sessions/{session_id}")).json()
                if session["status"] != "active":
                    break
                await asyncio.sleep(0.1)
    assert session["status"] == "complete", session
    assert [p["name"] for p in session["overrides"]["patches"]] == ["edit step 2 message 0"]
    live = stack.model.chat_calls()[calls:]
    assert live[0]["messages"][0]["content"] == "You are step 2, and very thorough."
    assert [s["served"] for s in session["result"]["steps"]] == ["tape", "patched", "live", "live"]


async def test_replay_command(stack: Stack, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from blackbox.cli import app

    source = await record(stack)
    env = {"BLACKBOX__SERVER__PORT": str(stack.bb.port), "BLACKBOX_CONFIG": str(tmp_path / "none.toml")}
    result = await asyncio.to_thread(CliRunner().invoke, app, ["replay", source.id], env=env)
    assert result.exit_code == 0, result.output
    assert "EXACT" in result.output and "0 live calls" in result.output
    patch = tmp_path / "patch.yaml"
    patch.write_text("patches:\n  - name: p\n    edit: {path: '$.model', set: other}\n")
    refused = await asyncio.to_thread(CliRunner().invoke, app, ["replay", source.id, "--patch", str(patch)], env=env)
    assert refused.exit_code == 2 and "--auto-fork" in refused.output
    forked = await asyncio.to_thread(
        CliRunner().invoke, app, ["replay", source.id, "--from-step", "3", "--times", "2"], env=env
    )
    assert forked.exit_code == 0, forked.output
    assert forked.output.count("2 live calls") == 2


async def test_sdk_clock_replays_exactly_with_the_shims(stack: Stack) -> None:
    """The prompt holds `sdk.now()` (to the microsecond): the replay gets the recorded value back, no normaliser."""
    stack.agent.clock = "sdk"
    source = await record(stack)
    values = await stack.bb.services.store.reader.recorded_values(source.trace_id)
    assert [v.kind for v in values] == ["now"] * 4
    await asyncio.sleep(0.05)
    report = await replay(stack, source.id)
    assert report["exact"] is True and "value_divergences" not in report
    replay_run = await stack.bb.services.store.reader.run(report["replay_run"])
    assert replay_run is not None
    replayed = await stack.bb.services.store.reader.recorded_values(replay_run.trace_id)
    assert [v.value for v in replayed] == [v.value for v in values]  # a replay is itself replayable
