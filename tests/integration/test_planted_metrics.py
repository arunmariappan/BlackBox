"""Planted runs: scripted transcripts against the real OpsDesk environment, each with known behaviour. Every metric
must flag exactly what was planted, and nothing on the clean run."""

from typing import Any

import pytest

from tests.opsdesk_stack import Ops, Script, record, wait_for_metrics

TICKET = {"title": "search is down", "service": "search-indexer", "priority": 2}
RESTART = ("restart_service", {"name": "search-indexer", "ticket_id": "$ticket"})

PLANTED: dict[str, tuple[str, Script, dict[str, Any]]] = {
    "clean": (
        "fix-06-indexer-oom",
        Script(
            [
                ("get_service", {"name": "search-indexer"}),
                ("get_logs", {"service": "search-indexer"}),
                ("create_ticket", TICKET),
                RESTART,
            ]
        ),
        {},
    ),
    "three-times restart loop": (
        "dep-01-frontend-cache",
        Script(
            [("create_ticket", {"title": "cache", "service": "cache", "priority": 2})]
            + [("restart_service", {"name": "cache", "ticket_id": "$ticket"})] * 3
        ),
        {"loop": "repeat"},
    ),
    "two-step cycle": (
        "fix-06-indexer-oom",
        Script(
            [
                ("create_ticket", TICKET),
                ("get_service", {"name": "search-indexer"}),
                RESTART,
                ("get_service", {"name": "search-indexer"}),
                RESTART,
            ]
        ),
        {"loop": "cycle of 2"},
    ),
    "state change in an information-only task": (
        "info-02-version",
        Script(
            [
                ("get_service", {"name": "checkout-api"}),
                ("create_ticket", {"title": "noted", "service": "checkout-api", "priority": 4}),
            ],
            "2.3.0",
        ),
        {"wrong_tool": 1, "checker_pass": "fail"},
    ),
    "503 then a successful retry": (
        "rec-03-transient-restart",
        Script([("get_service", {"name": "search-indexer"}), ("create_ticket", TICKET), RESTART, RESTART]),
        {"recovered_errors": 1},
    ),
    "503 then giving up": (
        "rec-03-transient-restart",
        Script(
            [("get_service", {"name": "search-indexer"}), ("create_ticket", TICKET), RESTART],
            "Restart failed; giving up.",
        ),
        {"unrecovered_errors": 1, "checker_pass": "fail"},
    ),
    "duplicate reads": (
        "fix-06-indexer-oom",
        Script(
            [
                ("get_service", {"name": "search-indexer"}),
                ("get_service", {"name": "search-indexer"}),
                ("create_ticket", TICKET),
                RESTART,
            ]
        ),
        {"repeated_calls": 1},
    ),
    "invalid arguments": (
        "fix-06-indexer-oom",
        Script([("create_ticket", {**TICKET, "priority": "urgent"}), ("create_ticket", TICKET), RESTART]),
        {"invalid_tool_calls": 1, "recovered_errors": 1},
    ),
}


def summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """The values a planted run is checked on; everything not planted must be zero (or pass)."""
    out: dict[str, Any] = {}
    if metrics["loop"].value:
        out["loop"] = metrics["loop"].label
    for name in (
        "wrong_tool",
        "policy_violations",
        "unrecovered_errors",
        "recovered_errors",
        "repeated_calls",
        "invalid_tool_calls",
    ):
        if metrics[name].value:
            out[name] = int(metrics[name].value)
    if metrics["checker_pass"].label != "pass":
        out["checker_pass"] = metrics["checker_pass"].label
    return out


@pytest.mark.parametrize("name", list(PLANTED))
async def test_planted_run(ops: Ops, name: str) -> None:
    task, script, expected = PLANTED[name]
    ops.model.reply = script.reply
    run, _ = await record(ops, task)
    metrics = await wait_for_metrics(ops, run.id)
    assert summary(metrics) == expected, {k: (v.value, v.label, v.details) for k, v in metrics.items() if v.value}


async def test_overview_flags_and_metrics_panel(ops: Ops) -> None:
    import httpx

    clean_task, clean_script, _ = PLANTED["clean"]
    ops.model.reply = clean_script.reply
    clean, _ = await record(ops, clean_task)
    loop_task, loop_script, _ = PLANTED["three-times restart loop"]
    ops.model.reply = loop_script.reply
    looping, _ = await record(ops, loop_task)
    await wait_for_metrics(ops, clean.id)
    await wait_for_metrics(ops, looping.id)
    async with httpx.AsyncClient(base_url=ops.bb.base_url) as client:
        flagged = await client.get("/runs/rows?flag=loop")
        assert looping.id in flagged.text and clean.id not in flagged.text
        listing = await client.get("/runs")
        assert "loop" in listing.text
        page = await client.get(f"/runs/{looping.id}")
        assert "Metrics" in page.text and "repeat from step" in page.text
        overview = await client.get("/overview?profile=opsdesk&window=7d")
        assert overview.status_code == 200
        assert "plotly" in overview.text.lower() and "checker pass" in overview.text
        assert "2 completed opsdesk runs" in overview.text
