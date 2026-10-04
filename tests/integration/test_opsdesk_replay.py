"""OpsDesk end to end with a scripted fake model (no GPU): the agent, its environment and BlackBox in one process,
talking over real sockets through the proxy, exactly as they would on the PC."""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from blackbox import sdk
from blackbox.config import UpstreamConfig
from blackbox.profiles.opsdesk import OpsDeskProfile
from blackbox.server import Running, start_server
from blackbox.store.models import Run, Score
from opsdesk.agent.app import make_agent_app
from opsdesk.env.app import make_env_app
from opsdesk.tasks import load_tasks
from tests.fakes import FakeUpstream
from tests.harness import running_blackbox


class Script:
    """Plays a fixed list of tool calls, then `finish`. `$ticket` is the id of the last ticket the agent created,
    read from the tool results in the conversation, as a model would."""

    def __init__(self, steps: list[tuple[str, dict[str, Any]]], answer: str = "Done with $ticket.") -> None:
        self.steps = steps
        self.answer = answer

    @staticmethod
    def last_ticket(messages: list[dict[str, Any]]) -> str:
        ticket = ""
        for message in messages:
            if message.get("role") != "tool":
                continue
            try:
                content = json.loads(message.get("content") or "")
            except ValueError:
                continue
            if isinstance(content, dict) and str(content.get("id", "")).startswith("TCK-"):
                ticket = content["id"]
        return ticket

    def reply(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body["messages"]
        turn = sum(1 for m in messages if m.get("role") == "assistant")
        ticket = self.last_ticket(messages)

        def fill(value: Any) -> Any:
            return value.replace("$ticket", ticket) if isinstance(value, str) else value

        if turn < len(self.steps):
            name, args = self.steps[turn]
            call = {"function": {"name": name, "arguments": {k: fill(v) for k, v in args.items()}}}
        else:
            call = {"function": {"name": "finish", "arguments": {"answer": fill(self.answer)}}}
        return {"role": "assistant", "content": "", "tool_calls": [call]}


FIX_01 = Script(
    [
        ("get_service", {"name": "checkout-api"}),
        ("get_logs", {"service": "checkout-api"}),
        ("create_ticket", {"title": "checkout-api 2.3.1 is down", "service": "checkout-api", "priority": 1}),
        ("rollback_service", {"name": "checkout-api", "ticket_id": "$ticket"}),
    ],
    "Rolled checkout-api back to 2.3.0 under $ticket.",
)
COMMENT_LATER = Script(
    [
        ("create_ticket", {"title": "look at search", "service": "search-indexer", "priority": 2}),
        ("get_service", {"name": "search-indexer"}),
        ("add_comment", {"ticket_id": "$ticket", "text": "search-indexer is down; restarting next."}),
    ],
    "Commented on $ticket.",
)


@dataclass
class Ops:
    bb: Running
    model: FakeUpstream
    env_url: str

    async def env_state(self, sandbox: str) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            state: dict[str, Any] = (await client.get(f"{self.env_url}/_sandboxes/{sandbox}/state")).json()
        return state


@pytest.fixture
async def ops(tmp_path: Path, migrated_db: Path) -> AsyncIterator[Ops]:
    model = await FakeUpstream(reply=FIX_01.reply).start()
    model.release.set()
    env_server, env_task, env_port = await start_server(make_env_app(load_tasks()), "127.0.0.1", 0, "opsdesk-env")
    env_url = f"http://127.0.0.1:{env_port}"
    profile = OpsDeskProfile({"env_url": env_url})
    upstreams = [
        UpstreamConfig(name="ollama", listen_port=0, target=model.url, record_paths=["/api/chat"]),
        UpstreamConfig(name="opsdesk-env", listen_port=0, target=env_url, record_paths=["/*"], stateful=True),
    ]
    async with running_blackbox(tmp_path, migrated_db, profiles=[profile], upstreams=upstreams) as bb:
        assert bb.services.proxy is not None
        ports = bb.services.proxy.ports
        agent = make_agent_app(f"http://127.0.0.1:{ports['ollama']}", f"http://127.0.0.1:{ports['opsdesk-env']}")
        agent_server, agent_task, agent_port = await start_server(agent, "127.0.0.1", 0, "opsdesk-agent")
        profile.options["agent_url"] = f"http://127.0.0.1:{agent_port}"
        sdk.init("opsdesk-agent", endpoint=bb.base_url, batch_delay_ms=50)
        try:
            yield Ops(bb, model, env_url)
        finally:
            await asyncio.to_thread(sdk.shutdown)
            agent_server.should_exit = True
            await agent_task
    env_server.should_exit = True
    await env_task
    await model.stop()


async def wait_for(ops: Ops, run_id: str, *, checker: bool = True, within: float = 20) -> tuple[Run, Score | None]:
    async with asyncio.timeout(within):
        while True:
            run = await ops.bb.services.store.reader.run(run_id)
            if run is not None and run.status != "open":
                if not checker:
                    return run, None
                scores = [s for s in await ops.bb.services.store.reader.scores(run_id) if s.kind == "checker"]
                if scores:
                    return run, scores[0]
            await asyncio.sleep(0.05)


async def record(ops: Ops, task: str, mode: str = "seeded") -> tuple[Run, Score | None]:
    async with httpx.AsyncClient(base_url=ops.bb.base_url, timeout=60) as client:
        response = await client.post(
            "/api/runs", json={"profile": "opsdesk", "input": "", "options": {"task": task, "mode": mode}, "wait": True}
        )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == 200, response.json()
    return await wait_for(ops, response.json()["run_id"])


async def replay(ops: Ops, run_id: str, **spec: Any) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=ops.bb.base_url, timeout=60) as client:
        response = await client.post("/api/sessions", json={"source_run_id": run_id, "wait": True, **spec})
    assert response.status_code == 200, response.text
    report: dict[str, Any] = response.json()["report"]
    assert "steps" in report, report
    return report


async def test_seeded_run_replays_exactly_and_the_checker_agrees(ops: Ops) -> None:
    run, checker = await record(ops, "fix-01-bad-deploy")
    assert run.profile == "opsdesk" and run.ending == "finished"
    assert checker is not None and checker.label == "pass", checker.details if checker else None
    steps = await ops.bb.services.store.reader.steps(run.id)
    assert [(s.kind, s.node) for s in steps] == [
        ("llm", "plan"),
        ("tool", "get_service"),
        ("llm", "plan"),
        ("tool", "get_logs"),
        ("llm", "plan"),
        ("tool", "create_ticket"),
        ("llm", "plan"),
        ("tool", "rollback_service"),
        ("llm", "plan"),
    ]
    assert steps[7].view["tool"]["arguments"]["ticket_id"].startswith("TCK-")
    values = await ops.bb.services.store.reader.recorded_values(run.trace_id)
    assert [v.kind for v in values] == ["now"]  # the system prompt's clock, from sdk.now()
    calls = len(ops.model.chat_calls())
    report = await replay(ops, run.id)
    assert report["exact"] is True and report["live_calls"] == 0
    assert len(ops.model.chat_calls()) == calls
    _, replay_checker = await wait_for(ops, report["replay_run"])
    assert replay_checker is not None and replay_checker.label == "pass"  # the session's sandbox followed the tape
    assert replay_checker.details["sandbox"] == report["sandbox"]
    async with httpx.AsyncClient(base_url=ops.bb.base_url) as client:
        page = await client.get(f"/runs/{run.id}")
        assert "checker" in page.text and "Action log" in page.text


async def test_chaotic_run_replays_exactly_although_ids_and_times_change(ops: Ops) -> None:
    run, checker = await record(ops, "fix-01-bad-deploy", mode="chaotic")
    assert checker is not None and checker.label == "pass"
    report = await replay(ops, run.id)
    assert report["exact"] is True, report["first_divergence"]
    [(tape_ticket, live_ticket)] = report["aliases"].items()
    assert tape_ticket != live_ticket and tape_ticket.startswith("TCK-")
    state = await ops.env_state(report["sandbox"])
    assert state["services"]["checkout-api"]["version"] == "2.3.0"  # the rollback reached the session's sandbox
    assert live_ticket in state["tickets"]
    _, replay_checker = await wait_for(ops, report["replay_run"])
    assert replay_checker is not None and replay_checker.label == "pass"
    async with httpx.AsyncClient(base_url=ops.bb.base_url) as client:
        page = await client.get(f"/sessions/{report['session_id']}")
        assert tape_ticket in page.text and live_ticket in page.text


@pytest.mark.parametrize("aliasing", [True, False], ids=["aliasing", "no-aliasing"])
async def test_fork_after_a_ticket_was_created(ops: Ops, aliasing: bool) -> None:
    ops.model.reply = COMMENT_LATER.reply
    run, _ = await record(ops, "fix-06-indexer-oom", mode="chaotic")
    steps = await ops.bb.services.store.reader.steps(run.id)
    assert [s.tool_name for s in steps if s.kind == "tool"] == ["create_ticket", "get_service", "add_comment"]
    report = await replay(ops, run.id, mode="fork", fork_step=4, aliasing=aliasing)
    assert [s["served"] for s in report["steps"]][:4] == ["tape", "tape", "tape", "live"]
    state = await ops.env_state(report["sandbox"])
    tickets = state["tickets"]
    replay_steps = await ops.bb.services.store.reader.steps(report["replay_run"])
    comment_step = next(s for s in replay_steps if s.tool_name == "add_comment")
    if aliasing:
        assert comment_step.status == "ok"
        [ticket] = [t for t in tickets.values() if t["title"] == "look at search"]
        assert ticket["comments"][0]["text"].startswith("search-indexer is down")
        # The agent still sees the tape's ticket id in the live response.
        assert comment_step.view["tool"]["result"]["id"] == next(iter(report["aliases"]))
    else:
        assert comment_step.status == "error" and comment_step.view["tool"]["result"]["error"].startswith("no ticket")
        assert all(not t["comments"] for t in tickets.values())


async def test_fork_sandbox_has_the_state_of_the_steps_before_it(ops: Ops) -> None:
    run, _ = await record(ops, "fix-01-bad-deploy", mode="chaotic")
    report = await replay(ops, run.id, mode="fork", fork_step=9)  # only the final model call goes live
    assert [s["served"] for s in report["steps"]] == ["tape"] * 8 + ["live"]
    state = await ops.env_state(report["sandbox"])
    assert state["services"]["checkout-api"]["version"] == "2.3.0"
    assert len([t for t in state["tickets"].values() if t["service"] == "checkout-api"]) == 1
    assert len(report["synced"]) == 2  # create_ticket and rollback_service were forwarded; the reads weren't
