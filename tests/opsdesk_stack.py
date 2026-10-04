"""The OpsDesk stack for integration tests: environment, agent and BlackBox in one process, with a scripted fake
model, plus helpers to record runs and wait for their checker and metrics."""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

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


async def ops_stack(tmp_path: Path, migrated_db: Path) -> AsyncIterator[Ops]:
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


async def wait_for_metrics(ops: Ops, run_id: str, within: float = 10) -> dict[str, Score]:
    async with asyncio.timeout(within):
        while True:
            scores = [s for s in await ops.bb.services.store.reader.scores(run_id) if s.kind == "metric"]
            if scores:
                return {s.name: s for s in scores}
            await asyncio.sleep(0.05)
