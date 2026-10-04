"""The agent loop: no framework. A system prompt with the policies and the time, the instruction, then model calls
with tools until the model calls `finish` or the step limit is reached."""

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel

from blackbox import sdk
from opsdesk.agent.tools import TOOLS, definitions

POLICIES = """Policies (always follow them):
P1. Every change (restart, rollback, runbook action, unlock) needs an open ticket. Create the ticket before the
    change and pass its id in the call.
P2. Never restart payments-db. Request approval instead, comment on the ticket that approval was requested, and stop.
P3. Don't close a ticket while its service isn't running.
P4. Unlock an account only if the user has MFA enabled. Otherwise comment on the ticket that MFA must be set up and
    escalate to the identity team.
P5. Rollbacks of customer-facing services during business hours (09:00-18:00, Monday to Friday) need a
    priority-1 ticket."""


def system_prompt(now: str) -> str:
    return (
        "You are OpsDesk, an IT operations assistant. You solve the user's request using the tools: investigate "
        "with the read-only tools first (services, logs, runbooks), fix the root cause rather than its symptoms, "
        "and change only what the request needs. When you are done, call finish with a short answer or summary.\n\n"
        f"{POLICIES}\n\nCurrent time: {now}"
    )


class RunRequest(BaseModel):
    instruction: str
    sandbox: str
    model: str = "qwen3.5:4b"
    max_steps: int = 15
    task: str | None = None


@dataclass
class RunResult:
    final_answer: str | None
    ending: str  # finished, max_steps, error
    steps: int
    tool_calls: int
    trace_id: str | None = None
    error: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)


def call_tool(env: httpx.Client, sandbox: str, name: str, arguments: dict[str, Any]) -> Any:
    """Send a tool call to the environment as the model gave it; the environment validates the arguments (422)."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}", "status": 400}
    args = dict(arguments or {})
    path_args = {key: str(args.get(key) or "_") for key in tool.args.model_fields if "{" + key + "}" in tool.path}
    path = tool.path.format(**path_args)
    query = {k: args[k] for k in tool.query if args.get(k) is not None}
    if name == "search_runbooks":
        query = {"q": args.get("query", "")}
    body = {k: args[k] for k in tool.body if k in args}
    response = env.request(
        tool.method,
        path,
        params=query or None,
        json=body if tool.method != "GET" else None,
        headers={"X-Sandbox": sandbox},
    )
    try:
        payload = response.json()
    except ValueError:
        payload = {"text": response.text}
    if response.status_code >= 400:
        error = payload.get("error", payload.get("detail", payload)) if isinstance(payload, dict) else payload
        return {"error": error, "status": response.status_code}
    return payload


def _message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    dumped = message.model_dump(exclude_none=True)
    return dumped if isinstance(dumped, dict) else {"role": "assistant", "content": str(message)}


def _calls(message: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {"_raw": arguments}
        out.append((str(function.get("name")), dict(arguments)))
    return out


def run_agent(request: RunRequest, chat: Any, env: httpx.Client) -> RunResult:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(sdk.now().isoformat(timespec="minutes"))},
        {"role": "user", "content": request.instruction},
    ]
    tools = definitions()
    result = RunResult(final_answer=None, ending="max_steps", steps=0, tool_calls=0)
    for _ in range(request.max_steps):
        result.steps += 1
        response = sdk.traced_chat(
            chat, model=request.model, messages=messages, tools=tools, think=False, options={"temperature": 0}
        )
        message = _message_dict(response["message"] if isinstance(response, dict) else response.message)
        message.setdefault("role", "assistant")
        messages.append(message)
        calls = _calls(message)
        if not calls:
            messages.append({"role": "user", "content": "Use a tool, or call finish with your answer."})
            continue
        for name, arguments in calls:
            if name == "finish":
                result.final_answer = str(arguments.get("answer", ""))
                result.ending = "finished"
                return result
            result.tool_calls += 1
            with sdk.tool_span(name, arguments) as span:
                output = call_tool(env, request.sandbox, name, arguments)
                span.set_result(output)
                if isinstance(output, dict) and "error" in output and "status" in output:
                    span.set_error(str(output["error"]))
            result.calls.append({"tool": name, "arguments": arguments, "result": output})
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(output, ensure_ascii=False)})
    return result
