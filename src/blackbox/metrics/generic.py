"""Metrics for every profile."""

import json
from collections import Counter
from typing import Any

from blackbox.metrics.framework import MetricInput, MetricValue, http_status, module, step_signature
from blackbox.runs.context import StepDraft

INVALID_STATUSES = (400, 422)


def is_read_only(inp: MetricInput, step: StepDraft) -> bool:
    if inp.profile is not None:
        decided = inp.profile.read_only(step)
        if decided is not None:
            return decided
    exchange = inp.exchange(step)
    return exchange is not None and exchange.row.method.upper() == "GET"


def tool_groups(inp: MetricInput) -> list[frozenset[str]]:
    return [frozenset(group) for group in (inp.profile.tool_groups if inp.profile else [])]


def same_group(inp: MetricInput, a: str | None, b: str | None) -> bool:
    if a == b:
        return True
    return any(a in group and b in group for group in tool_groups(inp))


def detect_loop(signatures: list[tuple[int, str]], model_requests: list[tuple[int, str]]) -> dict[str, Any] | None:
    """(a) one signature 3 or more times, (b) a cycle of 2 or 3 signatures repeated twice in a row, or (c) an
    identical model request sent twice. Returns the pattern and its first step."""
    counts = Counter(sig for _, sig in signatures)
    for idx, sig in signatures:
        if counts[sig] >= 3:
            steps = [i for i, s in signatures if s == sig]
            return {"pattern": "repeat", "signature": sig, "count": counts[sig], "first_step": idx, "steps": steps}
    sigs = [sig for _, sig in signatures]
    for length in (2, 3):
        for start in range(len(sigs) - 2 * length + 1):
            window = sigs[start : start + length]
            if len(set(window)) == length and window == sigs[start + length : start + 2 * length]:
                steps = [idx for idx, _ in signatures[start : start + 2 * length]]
                return {"pattern": f"cycle of {length}", "signatures": window, "first_step": steps[0], "steps": steps}
    seen: dict[str, int] = {}
    for idx, key in model_requests:
        if key in seen:
            return {"pattern": "identical model request", "first_step": seen[key], "steps": [seen[key], idx]}
        seen[key] = idx
    return None


def _defined_tools(step: StepDraft) -> set[str] | None:
    tools = step.view.get("tools")
    if not isinstance(tools, list):
        return None
    names = set()
    for tool in tools:
        if isinstance(tool, dict):
            function = tool.get("function") or tool
            if isinstance(function, dict) and function.get("name"):
                names.add(str(function["name"]))
    return names or None


def _reply_calls(step: StepDraft) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in step.view.get("reply") or []:
        if isinstance(message, dict):
            calls.extend(c for c in message.get("tool_calls") or [] if isinstance(c, dict))
    return calls


def _schema_reply_failed(step: StepDraft) -> bool:
    """A model reply that should have been JSON (a schema in `format`) but isn't."""
    if not isinstance(step.view.get("format"), dict | str) or not step.view.get("reply"):
        return False
    content = str(step.view["reply"][0].get("content") or "")
    if step.view.get("format") == "json" or isinstance(step.view.get("format"), dict):
        try:
            json.loads(content)
        except ValueError:
            return True
    return False


@module("generic", "1")
def generic(inp: MetricInput) -> list[MetricValue]:
    steps = inp.ctx.steps
    llm = [s for s in steps if s.kind == "llm"]
    tools = [s for s in steps if s.kind == "tool"]
    out = [
        MetricValue("steps", len(steps)),
        MetricValue("llm_calls", len(llm)),
        MetricValue("tool_calls", len(tools)),
        MetricValue("input_tokens", sum(s.input_tokens or 0 for s in llm)),
        MetricValue("output_tokens", sum(s.output_tokens or 0 for s in llm)),
        MetricValue("duration_ms", inp.run.duration_ms),
        MetricValue("llm_ms", sum(s.latency_ms or 0 for s in llm)),
        MetricValue("tool_ms", sum(s.latency_ms or 0 for s in tools)),
    ]
    errors = [s for s in tools if s.status == "error"]
    recovered: list[int] = []
    unrecovered: list[int] = []
    for error in errors:
        later = [
            s for s in tools if s.idx > error.idx and s.status == "ok" and same_group(inp, s.tool_name, error.tool_name)
        ]
        (recovered if later else unrecovered).append(error.idx)
    out.append(MetricValue("tool_errors", len(errors), details={"steps": [s.idx for s in errors]}))
    out.append(MetricValue("recovered_errors", len(recovered), details={"steps": recovered}))
    out.append(
        MetricValue("unrecovered_errors", len(unrecovered), details={"steps": unrecovered}, flag=bool(unrecovered))
    )

    repeated: list[int] = []
    seen: set[str] = set()
    for step in tools:
        signature = step_signature(step, inp.exchange(step))
        if is_read_only(inp, step):
            if signature in seen:
                repeated.append(step.idx)
            seen.add(signature)
        else:
            seen.clear()  # a change makes earlier reads stale; reading again isn't waste
    out.append(MetricValue("repeated_calls", len(repeated), details={"steps": repeated}))

    invalid: list[dict[str, Any]] = []
    for step in llm:
        defined = _defined_tools(step)
        if defined is None:
            continue
        for call in _reply_calls(step):
            if call.get("name") not in defined:
                invalid.append({"step": step.idx, "reason": f"unknown tool {call.get('name')!r}"})
    for step in tools:
        status = http_status(inp, step)
        if status in INVALID_STATUSES:
            invalid.append({"step": step.idx, "reason": f"rejected with {status}: {step.view.get('error', '')}"[:200]})
    out.append(
        MetricValue(
            "invalid_tool_calls", len(invalid), details={"calls": invalid, "steps": [c["step"] for c in invalid]}
        )
    )

    signatures = [(s.idx, step_signature(s, inp.exchange(s))) for s in tools]
    model_requests = [(s.idx, step_signature(s, inp.exchange(s))) for s in llm]
    loop = detect_loop(signatures, model_requests)
    out.append(
        MetricValue(
            "loop", 1.0 if loop else 0.0, label=loop["pattern"] if loop else None, details=loop or {}, flag=bool(loop)
        )
    )

    wasted = len(repeated) + len(invalid)
    out.append(
        MetricValue("wasted_steps", wasted, details={"steps": sorted(set(repeated) | {c["step"] for c in invalid})})
    )
    out.append(MetricValue("wasted_ratio", wasted / len(steps) if steps else 0.0))

    fallbacks: list[dict[str, Any]] = []
    for span in inp.ctx.spans:
        level = span.attributes.get("langfuse.observation.level")
        if level in ("WARNING", "ERROR"):
            fallbacks.append(
                {
                    "span": span.name,
                    "level": level,
                    "message": span.attributes.get("langfuse.observation.status_message"),
                }
            )
    for step in llm:
        if _schema_reply_failed(step):
            fallbacks.append({"step": step.idx, "reason": "reply failed the agent's output schema"})
    out.append(MetricValue("fallbacks", len(fallbacks), details={"items": fallbacks}))
    out.append(MetricValue("ending", None, label=inp.run.ending))
    return out
