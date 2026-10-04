"""The OpsDesk environment (`opsdesk env`, port 8221): any number of sandboxes, each an independent IT estate.

Agent-facing endpoints are the agent's tools; every call names its sandbox in an `X-Sandbox` header. Admin endpoints
(`/_sandboxes`, `/_tasks`) are for runners and checkers, never for the agent.
"""

import re
import uuid
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from opsdesk.env.model import Mode, Sandbox, Ticket
from opsdesk.env.scenarios import RUNBOOKS, build
from opsdesk.tasks import Task, load_tasks
from opsdesk.tasks.checker import check

RUNBOOK_ACTIONS = {
    "clear_cache": "cache_full",
    "rotate_logs": "disk_full",
    "renew_certificate": "cert_expired",
    "flush_queue": "queue_backlog",
    "scale_up": "slow",
}


class EnvError(Exception):
    def __init__(self, status: int, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status, self.message, self.extra = status, message, extra


class RestartBody(BaseModel):
    ticket_id: str | None = None


class RollbackBody(BaseModel):
    ticket_id: str | None = None
    version: str | None = None


class ActionBody(BaseModel):
    action: str
    ticket_id: str | None = None


class TicketBody(BaseModel):
    title: str
    service: str | None = None
    priority: int = Field(default=3, ge=1, le=4)
    tags: list[str] = Field(default_factory=list)


class CommentBody(BaseModel):
    text: str


class CloseBody(BaseModel):
    resolution: str = ""


class UnlockBody(BaseModel):
    ticket_id: str | None = None


class ApprovalBody(BaseModel):
    ticket_id: str | None = None
    action: str
    service: str | None = None
    reason: str = ""


class SandboxBody(BaseModel):
    task: str | None = None
    scenario: str | None = None
    seed: int | None = None
    mode: Mode = "seeded"


class CheckBody(BaseModel):
    task_id: str | None = None
    final_answer: str | None = None


def make_env_app(tasks: dict[str, Task] | None = None) -> FastAPI:
    app = FastAPI(title="OpsDesk env", docs_url="/_docs", openapi_url="/_openapi.json")
    sandboxes: dict[str, Sandbox] = {}
    known_tasks = tasks if tasks is not None else load_tasks()
    app.state.sandboxes = sandboxes

    @app.exception_handler(EnvError)
    async def env_error(_: Request, exc: EnvError) -> JSONResponse:
        return JSONResponse({"error": exc.message, **exc.extra}, status_code=exc.status)

    def box_of(sandbox: str | None, tool: str) -> Sandbox:
        if not sandbox:
            raise EnvError(400, "missing X-Sandbox header")
        box = sandboxes.get(sandbox)
        if box is None:
            raise EnvError(404, f"no sandbox {sandbox}")
        box.calls += 1
        if box.failures.get(tool, 0) > 0:
            box.failures[tool] -= 1
            raise EnvError(503, f"{tool} is temporarily unavailable, try again", retry=True)
        return box

    def service_of(box: Sandbox, name: str) -> Any:
        service = box.services.get(name)
        if service is None:
            raise EnvError(404, f"no service {name}", known=sorted(box.services))
        return service

    def ticket_context(box: Sandbox, ticket_id: str | None) -> dict[str, Any]:
        if ticket_id is None:
            return {"id": None, "exists": False}
        ticket = box.tickets.get(ticket_id)
        if ticket is None:
            raise EnvError(404, f"no ticket {ticket_id}")
        return {
            "id": ticket.id,
            "exists": True,
            "open": ticket.status == "open",
            "priority": ticket.priority,
            "service": ticket.service,
        }

    def refresh(box: Sandbox) -> None:
        """Services failing only because of a dependency follow it, transitively (until nothing changes)."""
        changed = True
        while changed:
            changed = False
            for service in box.services.values():
                if service.problem != "dependency":
                    continue
                ok = all(box.services[d].status == "running" for d in service.dependencies if d in box.services)
                status = "running" if ok else "degraded"
                if status != service.status:
                    service.status, changed = status, True
                    box.log(
                        service.name,
                        "INFO" if ok else "WARN",
                        "dependencies healthy again" if ok else "a dependency is unhealthy",
                    )

    Sandboxed = Annotated[str | None, Header(alias="X-Sandbox")]

    # Read-only tools --------------------------------------------------------------------------------------------

    @app.get("/services")
    def list_services(x_sandbox: Sandboxed = None) -> list[dict[str, Any]]:
        box = box_of(x_sandbox, "list_services")
        return [s.summary() for s in box.services.values()]

    @app.get("/services/{name}")
    def get_service(name: str, x_sandbox: Sandboxed = None) -> dict[str, Any]:
        box = box_of(x_sandbox, "get_service")
        service = service_of(box, name)
        return {**service.detail(), "deployments": box.deployments.get(name, [])}

    @app.get("/services/{name}/logs")
    def get_logs(name: str, lines: int = 20, x_sandbox: Sandboxed = None) -> dict[str, Any]:
        box = box_of(x_sandbox, "get_logs")
        service_of(box, name)
        return {"service": name, "lines": box.logs.get(name, [])[-max(1, min(lines, 200)) :]}

    @app.get("/runbooks/search")
    def search_runbooks(q: str = "", x_sandbox: Sandboxed = None) -> list[dict[str, Any]]:
        box_of(x_sandbox, "search_runbooks")
        words = [w for w in re.findall(r"[a-z0-9]+", q.lower()) if len(w) > 2]
        scored = []
        for runbook in RUNBOOKS:
            text = (runbook["title"] + " " + runbook["body"]).lower()
            score = sum(text.count(word) for word in words)
            if score:
                scored.append((score, runbook))
        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        return [{"id": r["id"], "title": r["title"]} for _, r in scored[:5]]

    @app.get("/runbooks/{runbook_id}")
    def read_runbook(runbook_id: str, x_sandbox: Sandboxed = None) -> dict[str, Any]:
        box_of(x_sandbox, "read_runbook")
        for runbook in RUNBOOKS:
            if runbook["id"].lower() == runbook_id.lower():
                return runbook
        raise EnvError(404, f"no runbook {runbook_id}")

    @app.get("/tickets")
    def list_tickets(status: str | None = None, x_sandbox: Sandboxed = None) -> list[dict[str, Any]]:
        box = box_of(x_sandbox, "list_tickets")
        return [t.view() for t in box.tickets.values() if status is None or t.status == status]

    @app.get("/users/{user_id}")
    def get_user(user_id: str, x_sandbox: Sandboxed = None) -> dict[str, Any]:
        box = box_of(x_sandbox, "get_user")
        user = box.users.get(user_id)
        if user is None:
            raise EnvError(404, f"no user {user_id}")
        return user.view()

    # State-changing tools ---------------------------------------------------------------------------------------

    @app.post("/services/{name}/restart")
    def restart_service(name: str, body: RestartBody, x_sandbox: Sandboxed = None) -> dict[str, Any]:
        box = box_of(x_sandbox, "restart_service")
        service = service_of(box, name)
        ticket = ticket_context(box, body.ticket_id)
        before = service.status
        service.restarts += 1
        if service.problem in ("crash", None, "slow", "cache_full"):
            service.status, service.problem = "running", None
            box.log(name, "INFO", f"{name} restarted")
        elif service.problem == "bad_deploy":
            service.status = "down"
            box.log(name, "ERROR", f"{name} {service.version} crashed again after restart")
        elif service.problem == "dependency":
            box.log(name, "WARN", f"{name} restarted; still failing: a dependency is unhealthy")
        else:
            box.log(name, "WARN", f"{name} restarted; problem persists ({service.problem})")
        refresh(box)
        result = {"service": name, "status": service.status, "restart_count": service.restarts}
        box.record(
            "restart",
            {"service": name, "ticket_id": body.ticket_id},
            result,
            True,
            ticket=ticket,
            status_before=before,
            protected=service.protected,
        )
        return result

    @app.post("/services/{name}/rollback")
    def rollback_service(name: str, body: RollbackBody, x_sandbox: Sandboxed = None) -> dict[str, Any]:
        box = box_of(x_sandbox, "rollback_service")
        service = service_of(box, name)
        ticket = ticket_context(box, body.ticket_id)
        history = box.deployments.get(name, [])
        healthy = [d for d in history if d["healthy"] and d["version"] != service.version]
        target = body.version or (healthy[-1]["version"] if healthy else None)
        if target is None or all(d["version"] != target for d in history):
            raise EnvError(409, f"no earlier version of {name} to roll back to")
        before = service.version
        service.version = target
        history.append({"version": target, "deployed_at": box.stamp(), "healthy": True})
        if service.problem == "bad_deploy" and any(d["version"] == target and d["healthy"] for d in history[:-1]):
            service.status, service.problem = "running", None
            box.log(name, "INFO", f"rolled back {name} {before} → {target}; healthy")
        refresh(box)
        result = {"service": name, "version": target, "previous_version": before, "status": service.status}
        box.record(
            "rollback",
            {"service": name, "ticket_id": body.ticket_id, "version": body.version},
            result,
            True,
            ticket=ticket,
            customer_facing=service.customer_facing,
            business_hours=box.business_hours(),
        )
        return result

    @app.post("/services/{name}/actions")
    def run_action(name: str, body: ActionBody, x_sandbox: Sandboxed = None) -> dict[str, Any]:
        box = box_of(x_sandbox, "run_action")
        service = service_of(box, name)
        ticket = ticket_context(box, body.ticket_id)
        args = {"service": name, "action": body.action, "ticket_id": body.ticket_id}
        if body.action not in RUNBOOK_ACTIONS:
            raise EnvError(422, f"unknown action {body.action!r}", known=sorted(RUNBOOK_ACTIONS))
        if body.action in box.broken_actions:
            box.record("action", args, {"error": box.broken_actions[body.action]}, False, ticket=ticket)
            raise EnvError(500, box.broken_actions[body.action])
        fixed = service.problem == RUNBOOK_ACTIONS[body.action]
        if fixed:
            service.status, service.problem = "running", None
            box.log(name, "INFO", f"{body.action} done; {name} healthy")
        else:
            box.log(name, "INFO", f"{body.action} done; no change")
        refresh(box)
        result = {"service": name, "action": body.action, "status": service.status, "fixed": fixed}
        box.record("action", args, result, True, ticket=ticket)
        return result

    @app.post("/tickets")
    def create_ticket(body: TicketBody, x_sandbox: Sandboxed = None) -> dict[str, Any]:
        box = box_of(x_sandbox, "create_ticket")
        ticket = Ticket(
            box.new_ticket_id(), body.title, body.service, body.priority, tags=body.tags, created_at=box.stamp()
        )
        box.tickets[ticket.id] = ticket
        result = ticket.view()
        box.record("create_ticket", body.model_dump(), {"id": ticket.id}, True)
        return result

    @app.post("/tickets/{ticket_id}/comments")
    def add_comment(ticket_id: str, body: CommentBody, x_sandbox: Sandboxed = None) -> dict[str, Any]:
        box = box_of(x_sandbox, "add_comment")
        ticket = box.tickets.get(ticket_id)
        if ticket is None:
            raise EnvError(404, f"no ticket {ticket_id}")
        ticket.comments.append({"at": box.stamp(), "text": body.text})
        box.record("comment", {"ticket_id": ticket_id, "text": body.text}, {"comments": len(ticket.comments)}, True)
        return ticket.view()

    @app.post("/tickets/{ticket_id}/close")
    def close_ticket(ticket_id: str, body: CloseBody, x_sandbox: Sandboxed = None) -> dict[str, Any]:
        box = box_of(x_sandbox, "close_ticket")
        ticket = box.tickets.get(ticket_id)
        if ticket is None:
            raise EnvError(404, f"no ticket {ticket_id}")
        service_status = box.services[ticket.service].status if ticket.service in box.services else None
        ticket.status, ticket.closed_at, ticket.resolution = "closed", box.stamp(), body.resolution
        box.record(
            "close_ticket",
            {"ticket_id": ticket_id, "resolution": body.resolution},
            {"status": "closed"},
            True,
            service=ticket.service,
            service_status=service_status,
        )
        return ticket.view()

    @app.post("/users/{user_id}/unlock")
    def unlock_user(user_id: str, body: UnlockBody, x_sandbox: Sandboxed = None) -> dict[str, Any]:
        box = box_of(x_sandbox, "unlock_user")
        user = box.users.get(user_id)
        if user is None:
            raise EnvError(404, f"no user {user_id}")
        ticket = ticket_context(box, body.ticket_id)
        user.locked = False
        result = user.view()
        box.record(
            "unlock", {"user_id": user_id, "ticket_id": body.ticket_id}, result, True, ticket=ticket, mfa=user.mfa
        )
        return result

    @app.post("/approvals")
    def request_approval(body: ApprovalBody, x_sandbox: Sandboxed = None) -> dict[str, Any]:
        box = box_of(x_sandbox, "request_approval")
        ticket = ticket_context(box, body.ticket_id)
        approval = {"id": box.new_approval_id(), "status": "pending", **body.model_dump(), "requested_at": box.stamp()}
        box.approvals.append(approval)
        box.record("request_approval", body.model_dump(), {"id": approval["id"]}, True, ticket=ticket)
        return approval

    # Admin ------------------------------------------------------------------------------------------------------

    @app.post("/_sandboxes")
    def create_sandbox(body: SandboxBody) -> dict[str, Any]:
        task = None
        if body.task is not None:
            task = known_tasks.get(body.task)
            if task is None:
                raise EnvError(404, f"no task {body.task}")
        scenario = body.scenario or (task.scenario if task else None)
        if scenario is None:
            raise EnvError(422, "give a task or a scenario")
        seed = body.seed if body.seed is not None else (task.seed if task else 0)
        sandbox_id = f"sbx-{uuid.uuid4().hex[:10]}"
        try:
            box = build(sandbox_id, scenario, seed, body.mode, task.id if task else None)
        except KeyError as exc:
            raise EnvError(404, str(exc)) from None
        sandboxes[sandbox_id] = box
        return {"id": sandbox_id, "scenario": scenario, "seed": seed, "mode": body.mode, "task_id": box.task_id}

    @app.get("/_sandboxes/{sandbox_id}/state")
    def sandbox_state(sandbox_id: str) -> dict[str, Any]:
        return admin_box(sandbox_id).state()

    @app.get("/_sandboxes/{sandbox_id}/actions")
    def sandbox_actions(sandbox_id: str) -> list[dict[str, Any]]:
        return admin_box(sandbox_id).actions

    @app.delete("/_sandboxes/{sandbox_id}")
    def delete_sandbox(sandbox_id: str) -> dict[str, Any]:
        admin_box(sandbox_id)
        del sandboxes[sandbox_id]
        return {"deleted": sandbox_id}

    @app.post("/_sandboxes/{sandbox_id}/check")
    def check_sandbox(sandbox_id: str, body: CheckBody) -> dict[str, Any]:
        box = admin_box(sandbox_id)
        task_id = body.task_id or box.task_id
        if task_id is None or task_id not in known_tasks:
            raise EnvError(404, f"no task {task_id}")
        return check(known_tasks[task_id], box.state(), box.actions, body.final_answer).as_dict()

    @app.get("/_tasks")
    def list_tasks() -> list[dict[str, Any]]:
        return [t.model_dump(mode="json") for t in known_tasks.values()]

    @app.get("/_tasks/{task_id}")
    def get_task(task_id: str) -> dict[str, Any]:
        task = known_tasks.get(task_id)
        if task is None:
            raise EnvError(404, f"no task {task_id}")
        return task.model_dump(mode="json")

    def admin_box(sandbox_id: str) -> Sandbox:
        box = sandboxes.get(sandbox_id)
        if box is None:
            raise HTTPException(404, f"no sandbox {sandbox_id}")
        return box

    @app.get("/_health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "sandboxes": len(sandboxes), "tasks": len(known_tasks)}

    return app
