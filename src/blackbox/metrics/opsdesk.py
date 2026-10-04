"""OpsDesk metrics, from the task file and the checker (the checker's score carries the task spec)."""

from typing import Any

from blackbox.metrics.framework import MetricInput, MetricValue, module, tool_arguments

TOOL_ACTIONS = {
    "restart_service": "restart",
    "rollback_service": "rollback",
    "run_action": "action",
    "unlock_user": "unlock",
    "close_ticket": "close_ticket",
    "create_ticket": "create_ticket",
    "add_comment": "comment",
    "request_approval": "request_approval",
}
READ_ONLY = frozenset(
    {"list_services", "get_service", "get_logs", "search_runbooks", "read_runbook", "list_tickets", "get_user"}
)


def _target(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "service": args.get("service") or args.get("name"),
        "user_id": args.get("user_id"),
        "ticket_id": args.get("ticket_id"),
    }


def _matches(action: str, target: dict[str, Any], rule: dict[str, Any]) -> bool:
    if rule.get("action") != action:
        return False
    return all(rule.get(key) in (None, target.get(key)) for key in ("service", "user_id", "ticket_id"))


@module("opsdesk", "1", profiles={"opsdesk"})
def opsdesk(inp: MetricInput) -> list[MetricValue]:
    checker = inp.checker
    out: list[MetricValue] = []
    if checker is not None:
        out.append(MetricValue("checker_pass", checker.value, label=checker.label, flag=checker.label == "fail"))
        violations = checker.details.get("violations") or {}
        out.append(
            MetricValue("policy_violations", len(violations), details={"policies": violations}, flag=bool(violations))
        )
    task = checker.details.get("task_spec") if checker is not None else None
    if not isinstance(task, dict):
        return out
    expected = set(task.get("expected_tools") or [])
    information = task.get("category") == "information"
    wrong: list[dict[str, Any]] = []
    extra_reads: list[dict[str, Any]] = []
    for step in inp.ctx.steps:
        if step.kind != "tool" or not step.tool_name:
            continue
        name = step.tool_name
        args = tool_arguments(step) or {}
        if name in READ_ONLY:
            if name not in expected:
                extra_reads.append({"step": step.idx, "tool": name})
            continue
        action = TOOL_ACTIONS.get(name, name)
        target = _target(args if isinstance(args, dict) else {})
        reasons = []
        if information:
            reasons.append("a change in an information-only task")
        elif name not in expected:
            reasons.append(f"{name} isn't among the task's expected tools")
        for rule in task.get("wrong_targets") or []:
            if _matches(action, target, rule):
                what = target.get("service") or target.get("user_id") or target.get("ticket_id")
                reasons.append(f"wrong target: {action} {what}")
        if reasons:
            wrong.append({"step": step.idx, "tool": name, "reasons": reasons})
    out.append(
        MetricValue(
            "wrong_tool", len(wrong), details={"calls": wrong, "steps": [w["step"] for w in wrong]}, flag=bool(wrong)
        )
    )
    out.append(
        MetricValue(
            "extra_read_tools",
            len(extra_reads),
            details={"calls": extra_reads, "steps": [c["step"] for c in extra_reads]},
        )
    )
    optimal = task.get("optimal_steps")
    if isinstance(optimal, int):
        out.append(MetricValue("steps_over_optimal", len(inp.ctx.steps) - optimal, details={"optimal": optimal}))
    return out
