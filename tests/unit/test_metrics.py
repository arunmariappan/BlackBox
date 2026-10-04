import json
from dataclasses import replace
from typing import Any

from sqlalchemy import select

from blackbox.metrics import MODULES, MetricInput, compute, compute_and_store
from blackbox.metrics.generic import detect_loop
from blackbox.otlp.decode import SpanData
from blackbox.profiles import default_registry
from blackbox.profiles.opsdesk import OpsDeskProfile
from blackbox.proxy.http import request_key
from blackbox.runs.context import ExchangeData, RunContext, StepDraft
from blackbox.store import Store
from blackbox.store.models import Exchange, Run, Score
from tests.fixtures import paperpilot

OPS = OpsDeskProfile()


def run_row(**fields: Any) -> Run:
    values: dict[str, Any] = {
        "id": "R1",
        "trace_id": "t" * 32,
        "profile": "opsdesk",
        "status": "complete",
        "updated_ms": 0,
    }
    values.update(fields)
    return Run(**values)


class Builder:
    def __init__(self) -> None:
        self.steps: list[StepDraft] = []
        self.exchanges: list[ExchangeData] = []

    def tool(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        *,
        status: int = 200,
        read_only: bool | None = None,
        sent: bytes | None = None,
    ) -> Builder:
        idx = len(self.steps) + 1
        body = json.dumps(args or {}).encode()
        row = Exchange(
            id=f"E{idx}",
            upstream="opsdesk-env",
            seq=idx,
            method="GET" if read_only else "POST",
            path=f"/{name}",
            request_key=request_key("POST", f"/{name}", "", body),
            status=status,
            started_ms=idx,
            ended_ms=idx + 1,
            served_from="live",
        )
        self.exchanges.append(ExchangeData(row, body, b"{}", sent))
        if read_only is None:
            read_only = name.startswith(("get_", "list_", "search_", "read_"))
        self.steps.append(
            StepDraft(
                idx=idx,
                kind="tool",
                tool_name=name,
                exchange_id=row.id,
                status="ok" if status < 400 else "error",
                view={"tool": {"name": name, "arguments": args or {}}, "read_only": read_only},
            )
        )
        return self

    def llm(
        self,
        messages: list[Any],
        *,
        reply_calls: list[str] = (),
        tools: list[str] = (),
        content: str = "",
        fmt: Any = None,
    ) -> Builder:  # type: ignore[assignment]
        idx = len(self.steps) + 1
        body = json.dumps({"messages": messages}).encode()
        row = Exchange(
            id=f"E{idx}",
            upstream="ollama",
            seq=idx,
            method="POST",
            path="/api/chat",
            request_key=request_key("POST", "/api/chat", "", body),
            status=200,
            started_ms=idx,
            ended_ms=idx + 1,
            served_from="live",
        )
        self.exchanges.append(ExchangeData(row, body, b"{}"))
        view: dict[str, Any] = {
            "messages": messages,
            "reply": [
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [{"name": n, "arguments": {}} for n in reply_calls],
                }
            ],
        }
        if tools:
            view["tools"] = [{"type": "function", "function": {"name": t}} for t in tools]
        if fmt is not None:
            view["format"] = fmt
        self.steps.append(
            StepDraft(idx=idx, kind="llm", exchange_id=row.id, view=view, input_tokens=10, output_tokens=2)
        )
        return self

    def metrics(self, spans: list[SpanData] | None = None, scores: list[Score] | None = None) -> dict[str, Any]:
        ctx = RunContext(run=run_row(), spans=spans or [], exchanges=self.exchanges, steps=self.steps)
        return {value.name: value for _, value in compute(MetricInput(ctx, OPS, scores or []))}


def test_cycle_broken_by_a_different_call_is_not_a_loop() -> None:
    sigs = [(1, "A"), (2, "B"), (3, "X"), (4, "A"), (5, "B")]
    assert detect_loop(sigs, []) is None
    assert detect_loop([(1, "A"), (2, "B"), (3, "A"), (4, "B")], [])["pattern"] == "cycle of 2"  # type: ignore[index]
    assert detect_loop([(1, "A"), (2, "B"), (3, "C"), (4, "A"), (5, "B"), (6, "C")], [])["pattern"] == "cycle of 3"  # type: ignore[index]
    repeat = detect_loop([(1, "A"), (2, "B"), (3, "A"), (4, "C"), (5, "A")], [])
    assert repeat is not None and repeat["pattern"] == "repeat" and repeat["steps"] == [1, 3, 5]
    same_request = detect_loop([], [(1, "k1"), (3, "k2"), (5, "k1")])
    assert same_request is not None and same_request["first_step"] == 1


def test_repeat_after_a_state_change_is_not_a_repeat() -> None:
    m = (
        Builder()
        .tool("get_service", {"name": "cache"})
        .tool("restart_service", {"name": "cache"})
        .tool("get_service", {"name": "cache"})
        .metrics()
    )
    assert m["repeated_calls"].value == 0
    m = Builder().tool("get_service", {"name": "cache"}).tool("get_service", {"name": "cache"}).metrics()
    assert m["repeated_calls"].value == 1 and m["repeated_calls"].details["steps"] == [2]
    assert m["wasted_steps"].value == 1 and m["wasted_ratio"].value == 0.5


def test_error_recovery() -> None:
    m = Builder().tool("get_logs", {"service": "a"}, status=503).tool("get_logs", {"service": "a"}).metrics()
    assert (m["tool_errors"].value, m["recovered_errors"].value, m["unrecovered_errors"].value) == (1, 1, 0)
    m = Builder().tool("get_service", {"name": "a"}).tool("restart_service", {"name": "a"}, status=503).metrics()
    assert m["unrecovered_errors"].value == 1 and m["unrecovered_errors"].flag  # an error on the last step
    m = (
        Builder()
        .tool("run_action", {"action": "clear_cache"}, status=500)
        .tool("restart_service", {"name": "cache"})
        .metrics()
    )
    assert m["recovered_errors"].value == 1  # restart_service is in run_action's group


def test_aliased_ids_keep_the_same_signature() -> None:
    """In a fork, the body sent to the sandbox carries the live id; the agent's own request keeps the tape's."""
    b = Builder().tool("get_service", {"name": "cache", "ticket": "TCK-1"})
    b.tool("get_service", {"name": "cache", "ticket": "TCK-1"}, sent=b'{"name": "cache", "ticket": "TCK-9"}')
    assert b.metrics()["repeated_calls"].value == 1


def test_invalid_tool_calls_and_fallbacks() -> None:
    b = Builder().llm([{"role": "user", "content": "x"}], reply_calls=["reboot_everything"], tools=["get_service"])
    b.tool("create_ticket", {"priority": "urgent"}, status=422)
    m = b.metrics()
    assert m["invalid_tool_calls"].value == 2 and m["invalid_tool_calls"].details["steps"] == [1, 2]
    degraded = SpanData(
        "t" * 32, "s" * 16, None, "document_grading", attributes={"langfuse.observation.level": "WARNING"}
    )
    m = (
        Builder()
        .llm([{"role": "user", "content": "y"}], fmt={"type": "object"}, content="not json")
        .metrics(spans=[degraded])
    )
    assert m["fallbacks"].value == 2


def test_loop_flag_and_model_request_loop() -> None:
    same = [{"role": "user", "content": "same"}]
    m = Builder().llm(same).llm(same).metrics()
    assert m["loop"].flag and m["loop"].label == "identical model request"
    m = Builder().tool("get_service", {"name": "x"}).tool("get_logs", {"service": "x"}).metrics()
    assert m["loop"].value == 0 and not m["loop"].flag


def test_opsdesk_metrics_read_the_task_from_the_checker() -> None:
    task = {
        "category": "dependencies",
        "expected_tools": ["get_service", "create_ticket", "restart_service", "finish"],
        "optimal_steps": 3,
        "wrong_targets": [{"action": "restart", "service": "web-frontend"}],
    }
    checker = Score(
        id="S",
        run_id="R1",
        kind="checker",
        name="checker",
        version="1",
        value=0.0,
        label="fail",
        details={"task_spec": task, "violations": {"P1": ["step 2: restart without a ticket"]}},
        created_ms=0,
    )
    b = Builder().tool("get_service", {"name": "web-frontend"}).tool("restart_service", {"name": "web-frontend"})
    b.tool("rollback_service", {"name": "cache"}).tool("get_logs", {"service": "cache"})
    m = b.metrics(scores=[checker])
    assert m["wrong_tool"].value == 2 and m["wrong_tool"].details["steps"] == [2, 3]
    assert m["extra_read_tools"].value == 1 and m["extra_read_tools"].details["steps"] == [4]
    assert m["policy_violations"].value == 1 and m["policy_violations"].flag
    assert m["steps_over_optimal"].value == 1
    assert m["checker_pass"].value == 0.0 and m["checker_pass"].flag
    info = {**task, "category": "information", "expected_tools": ["get_service", "create_ticket", "finish"]}
    checker.details = {"task_spec": info}
    m = Builder().tool("get_service", {"name": "x"}).tool("create_ticket", {"title": "x"}).metrics(scores=[checker])
    assert m["wrong_tool"].value == 1 and "information-only" in m["wrong_tool"].details["calls"][0]["reasons"][0]


async def test_paperpilot_metrics_on_the_fixture(store: Store) -> None:
    run_id = await paperpilot.insert_run(store)
    run = await store.reader.run(run_id)
    assert run is not None
    values = {v.name: v for v in await compute_and_store(store, default_registry(), run)}
    assert values["retrieval_attempts"].value == 2
    assert values["rewrites"].value == 1 and values["rewrites"].details["steps"] == [5]
    assert values["relevant_gradings"].value == 1
    assert values["guardrail_score"].value == 92
    assert values["empty_retrievals"].value == 0
    assert values["citations_present"].value == 1 and values["citations_valid"].value == 1
    assert values["wasted_rewrite"].value == 0
    assert values["steps"].value == 9 and values["loop"].value == 0


async def test_recompute_replaces_old_scores(store: Store) -> None:
    run_id = await paperpilot.insert_run(store)
    run = await store.reader.run(run_id)
    assert run is not None
    await compute_and_store(store, default_registry(), run)
    original = list(MODULES)
    try:
        MODULES[:] = [replace(m, version="2") if m.name == "generic" else m for m in MODULES]
        await compute_and_store(store, default_registry(), run)
    finally:
        MODULES[:] = original
    async with store.read() as s:
        versions = set(
            (await s.execute(select(Score.version).where(Score.run_id == run_id, Score.kind == "metric"))).scalars()
        )
    assert versions == {"generic-2", "paperpilot-1"}
