"""OpsDesk, the Python test agent for a simulated IT ops desk (`src/opsdesk`). BlackBox talks to it only over HTTP,
as it would to any outside agent: the agent at 8220, the environment's admin API at 8221 for sandboxes and the
checker, and the environment's tools through the proxy at 8213."""

import json
import random
import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from blackbox.net import make_client
from blackbox.otlp.decode import SpanData
from blackbox.profiles.base import Profile, StartRequest, TrafficCase
from blackbox.runs.context import ExchangeData, RunContext, StepDraft

if TYPE_CHECKING:
    from blackbox.proxy.sessions import ReplaySession
    from blackbox.proxy.views import ExchangeView
    from blackbox.services import Services
    from blackbox.store.models import Run

ROOT = "invoke_agent opsdesk"
SANDBOX_HEADER = "x-sandbox"

# method, path pattern → tool name. Path groups become arguments.
ROUTES: list[tuple[str, re.Pattern[str], str]] = [
    ("GET", re.compile(r"^/services$"), "list_services"),
    ("GET", re.compile(r"^/services/(?P<name>[^/]+)$"), "get_service"),
    ("GET", re.compile(r"^/services/(?P<service>[^/]+)/logs$"), "get_logs"),
    ("GET", re.compile(r"^/runbooks/search$"), "search_runbooks"),
    ("GET", re.compile(r"^/runbooks/(?P<id>[^/]+)$"), "read_runbook"),
    ("POST", re.compile(r"^/services/(?P<name>[^/]+)/restart$"), "restart_service"),
    ("POST", re.compile(r"^/services/(?P<name>[^/]+)/rollback$"), "rollback_service"),
    ("POST", re.compile(r"^/services/(?P<service>[^/]+)/actions$"), "run_action"),
    ("POST", re.compile(r"^/tickets$"), "create_ticket"),
    ("POST", re.compile(r"^/tickets/(?P<ticket_id>[^/]+)/comments$"), "add_comment"),
    ("POST", re.compile(r"^/tickets/(?P<ticket_id>[^/]+)/close$"), "close_ticket"),
    ("GET", re.compile(r"^/tickets$"), "list_tickets"),
    ("GET", re.compile(r"^/users/(?P<user_id>[^/]+)$"), "get_user"),
    ("POST", re.compile(r"^/users/(?P<user_id>[^/]+)/unlock$"), "unlock_user"),
    ("POST", re.compile(r"^/approvals$"), "request_approval"),
]
READ_ONLY = frozenset(
    {"list_services", "get_service", "get_logs", "search_runbooks", "read_runbook", "list_tickets", "get_user"}
)
IDENTIFIER_PATHS = {("POST", "/tickets"): ["$.id"], ("POST", "/approvals"): ["$.id"]}


def tool_of(method: str, path: str) -> tuple[str, dict[str, str]] | None:
    for route_method, pattern, name in ROUTES:
        found = pattern.match(path)
        if found and route_method == method.upper():
            return name, dict(found.groupdict())
    return None


class OpsDeskProfile(Profile):
    name = "opsdesk"
    description = "OpsDesk (Python tool-calling agent over a simulated IT ops desk)"
    failure_endings = frozenset({"max_steps", "error"})
    stateful_upstreams = frozenset({"opsdesk-env"})
    tool_groups: ClassVar[list[set[str]]] = [{"run_action", "restart_service", "rollback_service"}]

    @property
    def agent_url(self) -> str:
        return str(self.options.get("agent_url", "http://127.0.0.1:8220")).rstrip("/")

    @property
    def env_url(self) -> str:
        """The environment's own port, for admin calls (sandboxes, checker); the agent's tools go via the proxy."""
        return str(self.options.get("env_url", "http://127.0.0.1:8221")).rstrip("/")

    def matches(self, spans: Sequence[SpanData]) -> bool:
        return any(span.name == ROOT for span in spans)

    def node_for(self, step: StepDraft, chain: Iterable[SpanData], ctx: RunContext) -> str | None:
        if step.kind == "tool":
            return step.tool_name
        if step.kind == "llm":
            return "plan"
        return None

    # Starting runs -------------------------------------------------------------------------------------------------

    def parse_input(self, text: str, **options: Any) -> dict[str, Any]:
        run_input: dict[str, Any] = {
            "model": str(options.get("model") or self.options.get("model", "qwen3.5:4b")),
            "mode": str(options.get("mode") or "seeded"),
        }
        if options.get("task"):
            run_input["task"] = str(options["task"])
        if text:
            run_input["instruction"] = text
        if options.get("seed") is not None:
            run_input["seed"] = int(options["seed"])
        if options.get("max_steps"):
            run_input["max_steps"] = int(options["max_steps"])
        return run_input

    async def traffic_case(self, rng: random.Random) -> TrafficCase | None:
        """A task from the environment, in `seeded` mode with a seed of its own."""
        async with make_client(timeout=10) as client:
            response = await client.get(f"{self.env_url}/_tasks")
        response.raise_for_status()
        tasks = sorted(task["id"] for task in response.json())
        if not tasks:
            return None
        task, seed = rng.choice(tasks), rng.randrange(1_000_000)
        return TrafficCase("", {"task": task, "mode": "seeded", "seed": seed}, {"task": task, "seed": seed})

    async def prepare_input(self, run_input: dict[str, Any], services: Services) -> dict[str, Any]:
        """Create the run's sandbox (from its task's scenario and seed) and fill in the task's instruction."""
        run_input = dict(run_input)
        async with make_client(timeout=30) as client:
            task = None
            if run_input.get("task"):
                response = await client.get(f"{self.env_url}/_tasks/{run_input['task']}")
                response.raise_for_status()
                task = response.json()
                run_input.setdefault("instruction", task["instruction"])
                run_input.setdefault("max_steps", task["max_steps"])
                run_input.setdefault("seed", task["seed"])
            if "sandbox" not in run_input:
                body = {
                    "task": run_input.get("task"),
                    "mode": run_input.get("mode", "seeded"),
                    "seed": run_input.get("seed"),
                }
                if task is None:
                    body["scenario"] = run_input.get("scenario", "healthy")
                response = await client.post(f"{self.env_url}/_sandboxes", json=body)
                response.raise_for_status()
                created = response.json()
                run_input["sandbox"] = created["id"]
                run_input["seed"] = created["seed"]
        return run_input

    def build_request(self, run_input: dict[str, Any]) -> StartRequest:
        if "sandbox" not in run_input or "instruction" not in run_input:
            raise ValueError("an OpsDesk run needs a sandbox and an instruction (give --task)")
        return StartRequest(
            method="POST",
            url=f"{self.agent_url}/run",
            body=run_input,
            headers={"content-type": "application/json"},
            timeout_seconds=float(self.options.get("timeout_seconds", 900)),
        )

    # Reading runs -------------------------------------------------------------------------------------------------

    def input_text(self, ctx: RunContext) -> str | None:
        body = ctx.entry_body
        if isinstance(body, dict) and isinstance(body.get("instruction"), str):
            return str(body["instruction"])
        return super().input_text(ctx)

    def output_text(self, output: Any) -> str | None:
        if isinstance(output, dict):
            answer = output.get("final_answer")
            return str(answer) if answer is not None else (str(output.get("error")) if output.get("error") else None)
        return super().output_text(output)

    def ending(self, ctx: RunContext, output: Any) -> str | None:
        if isinstance(output, dict) and isinstance(output.get("ending"), str):
            return str(output["ending"])
        return super().ending(ctx, output)

    def exchange_view(self, exchange: ExchangeData, ctx: RunContext) -> ExchangeView | None:
        from blackbox.proxy.views import ExchangeView, decode_body

        row = exchange.row
        if row.upstream not in self.stateful_upstreams:
            return None
        found = tool_of(row.method, row.path)
        name, arguments = found if found else (f"{row.method} {row.path}", {})
        if row.query:
            from urllib.parse import parse_qsl

            for key, value in parse_qsl(row.query):
                arguments["query" if key == "q" else key] = value
        if exchange.request_body:
            try:
                body = json.loads(exchange.request_body)
                if isinstance(body, dict):
                    arguments.update(body)
            except ValueError:
                pass
        raw = decode_body(exchange.response_body, row.response_headers)
        try:
            result: Any = json.loads(raw) if raw else None
        except ValueError:
            result = raw.decode("utf-8", errors="replace")
        error = None
        if (row.status or 0) >= 400:
            error = result.get("error") if isinstance(result, dict) else str(result)
        view = {
            "source": "exchange",
            "tool": {"name": name, "arguments": arguments, "result": result},
            "read_only": name in READ_ONLY,
        }
        return ExchangeView(kind="tool", view=view, tool_name=name, error=str(error) if error else None)

    # Replay -------------------------------------------------------------------------------------------------------

    def identifier_paths(self, method: str, path: str) -> list[str]:
        return IDENTIFIER_PATHS.get((method.upper(), path), [])

    async def prepare(self, session: ReplaySession, services: Services) -> None:
        """Every replay or fork gets a fresh sandbox built like the source run's (same task, scenario, seed, mode).
        Sync-forwarding then keeps its state in step with the tape."""
        source = await services.store.reader.run(session.spec.source_run_id)
        if source is None:
            return
        from blackbox.runs.context import load_run_context

        ctx = await load_run_context(services.store, source)
        body = ctx.entry_body if isinstance(ctx.entry_body, dict) else {}
        async with make_client(timeout=30) as client:
            state = {}
            if body.get("sandbox"):
                response = await client.get(f"{self.env_url}/_sandboxes/{body['sandbox']}/state")
                if response.status_code == 200:
                    state = response.json()
            request = {
                "task": body.get("task") or state.get("task_id"),
                "scenario": state.get("scenario") or body.get("scenario"),
                "seed": body.get("seed", state.get("seed")),
                "mode": body.get("mode") or state.get("mode") or "seeded",
            }
            if request["scenario"] is None and request["task"] is None:
                request["scenario"] = "healthy"
            response = await client.post(f"{self.env_url}/_sandboxes", json=request)
            response.raise_for_status()
        session.state["sandbox"] = response.json()["id"]
        session.state["source_sandbox"] = body.get("sandbox")

    def replay_request(self, request: StartRequest, session: ReplaySession) -> StartRequest:
        sandbox = session.state.get("sandbox")
        if sandbox is None or not isinstance(request.body, dict):
            return request
        return StartRequest(
            request.method,
            request.url,
            {**request.body, "sandbox": sandbox},
            dict(request.headers),
            request.timeout_seconds,
        )

    # After completion ---------------------------------------------------------------------------------------------

    async def after_complete(self, services: Services, run: Run, ctx: RunContext) -> None:
        """Run the checker on the run's sandbox and store it as a `checker` score (with the action log)."""
        from blackbox.runs.scores import write_score

        body = ctx.entry_body if isinstance(ctx.entry_body, dict) else {}
        sandbox, task = body.get("sandbox"), body.get("task")
        if not sandbox or not task:
            return
        output = self.read_output(ctx)
        answer = output.get("final_answer") if isinstance(output, dict) else None
        async with make_client(timeout=30) as client:
            checked = await client.post(
                f"{self.env_url}/_sandboxes/{sandbox}/check", json={"task_id": task, "final_answer": answer}
            )
            actions = await client.get(f"{self.env_url}/_sandboxes/{sandbox}/actions")
            task_response = await client.get(f"{self.env_url}/_tasks/{task}")
        checked.raise_for_status()
        result = checked.json()
        details = {
            "task": task,
            "sandbox": sandbox,
            "failed": result["failed"],
            "checks": result["checks"],
            "violations": result["violations"],
            "actions": actions.json() if actions.status_code == 200 else [],
            "task_spec": task_response.json() if task_response.status_code == 200 else None,
        }
        await write_score(
            services.store,
            run.id,
            kind="checker",
            name="checker",
            version="1",
            label="pass" if result["passed"] else "fail",
            value=1.0 if result["passed"] else 0.0,
            details=details,
        )
