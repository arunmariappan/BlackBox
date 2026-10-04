"""OpsDesk end to end with a scripted fake model (no GPU): the agent, its environment and BlackBox in one process,
talking over real sockets through the proxy, exactly as they would on the PC."""

from typing import Any

import httpx
import pytest

from tests.opsdesk_stack import COMMENT_LATER, Ops, record, wait_for


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
