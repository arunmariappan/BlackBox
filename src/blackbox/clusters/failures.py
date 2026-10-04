"""Which runs failed, the deterministic facts about each failure, and a compact digest of the run for the model.

A run has failed if its OpsDesk checker failed, a **trusted** judge said `fail`, its ending is one the profile marks
as a failure, or a metric flag is set (`loop`, `wrong_tool`, `policy_violations`, `unrecovered_errors`). Untrusted
judges never decide failure.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from blackbox.profiles.base import Profile
from blackbox.runs.context import RunContext
from blackbox.store.models import Score

FAILURE_FLAGS = ("loop", "wrong_tool", "policy_violations", "unrecovered_errors")


@dataclass
class FailureFacts:
    failed: bool
    reasons: list[str] = field(default_factory=list)
    ending: str | None = None
    failed_checks: list[str] = field(default_factory=list)
    failed_judges: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    first_problem_step: int | None = None
    first_problem_node: str | None = None
    tool_errors: list[str] = field(default_factory=list)
    divergence: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reasons": self.reasons,
            "ending": self.ending,
            "failed_checks": self.failed_checks,
            "failed_judges": self.failed_judges,
            "flags": self.flags,
            "first_problem_step": self.first_problem_step,
            "first_problem_node": self.first_problem_node,
            "tool_errors": self.tool_errors,
            "divergence": self.divergence,
        }


def facts(ctx: RunContext, profile: Profile | None, scores: list[Score], trusted: dict[str, set[str]]) -> FailureFacts:
    run = ctx.run
    out = FailureFacts(failed=False, ending=run.ending)
    problem_steps: list[int] = []
    for score in scores:
        if score.kind == "checker" and score.label == "fail":
            out.failed_checks = [str(f) for f in score.details.get("failed", [])]
            out.reasons.append("checker failed")
        elif score.kind == "judge" and score.label == "fail" and score.version in trusted.get(score.name, set()):
            out.failed_judges.append(score.name)
            out.reasons.append(f"trusted judge {score.name} said fail")
            answer_step = _answer_step(ctx)
            if answer_step is not None:
                problem_steps.append(answer_step)
        elif score.kind == "metric" and score.name in FAILURE_FLAGS and score.details.get("flag"):
            out.flags.append(score.name)
            out.reasons.append(f"metric {score.name}")
            steps = score.details.get("steps") or (
                [score.details["first_step"]] if score.details.get("first_step") else []
            )
            problem_steps.extend(int(s) for s in steps if isinstance(s, int))
    if profile is not None and run.ending in profile.failure_endings:
        out.reasons.append(f"ending {run.ending}")
    errors = [s for s in ctx.steps if s.status == "error"]
    for step in errors:
        status = step.view.get("error") or "error"
        out.tool_errors.append(f"step {step.idx} {step.tool_name or step.kind}: {str(status)[:120]}")
    problem_steps.extend(s.idx for s in errors)
    for exchange in ctx.exchanges:
        if exchange.row.divergence:
            out.divergence = {k: exchange.row.divergence.get(k) for k in ("step", "node", "path")}
            break
    if problem_steps:
        out.first_problem_step = min(problem_steps)
        first = next((s for s in ctx.steps if s.idx == out.first_problem_step), None)
        out.first_problem_node = first.node if first is not None else None
    out.failed = bool(out.reasons)
    return out


def _answer_step(ctx: RunContext) -> int | None:
    llm = [s for s in ctx.steps if s.kind == "llm"]
    return llm[-1].idx if llm else None


def _short(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def step_line(step: Any) -> str:
    view = step.view
    if step.kind == "tool":
        tool = view.get("tool") or {}
        if tool:
            result = tool.get("result")
            outcome = f"error {view.get('error')}" if step.status == "error" else _short(result, 120)
            call = f"{tool.get('name')}({_short(tool.get('arguments'), 100)})"
            return f"#{step.idx} [{step.node or 'tool'}] {call} -> {outcome}"
        if "hits" in view:
            return f"#{step.idx} [{step.node or 'search'}] search -> {len(view['hits'])} hits"
    if step.kind == "llm":
        reply = (view.get("reply") or [{}])[0] if view.get("reply") else {}
        calls = [c.get("name") for c in reply.get("tool_calls") or []]
        said = f"calls {', '.join(str(c) for c in calls)}" if calls else _short(reply.get("content", ""), 120)
        return f"#{step.idx} [{step.node or 'model'}] model -> {said}"
    if step.kind == "embedding":
        return f"#{step.idx} [{step.node or 'embed'}] embedding of {len(view.get('inputs') or [])} texts"
    return f"#{step.idx} [{step.node or step.kind}] {step.kind}"


def digest(ctx: RunContext, found: FailureFacts, *, max_steps: int = 40) -> str:
    run = ctx.run
    lines = [f"Input: {_short(run.input_text or '', 400)}"]
    lines.append(f"Ending: {run.ending}")
    lines.append("Steps:")
    steps = ctx.steps
    if len(steps) > max_steps:
        steps = steps[: max_steps // 2] + steps[-max_steps // 2 :]
    lines.extend(f"  {step_line(step)}" for step in steps)
    if run.output_text:
        lines.append(f"Final output: {_short(run.output_text, 400)}")
    lines.append("Facts:")
    for reason in found.reasons:
        lines.append(f"  - {reason}")
    for check in found.failed_checks[:8]:
        lines.append(f"  - failed check: {_short(check, 200)}")
    for error in found.tool_errors[:5]:
        lines.append(f"  - {error}")
    if found.first_problem_step is not None:
        lines.append(f"  - first problem at step {found.first_problem_step} ({found.first_problem_node})")
    if found.divergence:
        lines.append(f"  - replay diverged at tape step {found.divergence.get('step')}")
    return "\n".join(lines)
