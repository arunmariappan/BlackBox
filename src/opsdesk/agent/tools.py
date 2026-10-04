"""The agent's tools: one per environment endpoint, plus `finish`. JSON schemas come from Pydantic models."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class NoArgs(BaseModel):
    pass


class ServiceName(BaseModel):
    name: str = Field(description="Service name, e.g. checkout-api")


class LogsArgs(BaseModel):
    service: str = Field(description="Service name")
    lines: int = Field(default=20, description="How many recent log lines")


class SearchArgs(BaseModel):
    query: str = Field(description="Words to search runbooks for")


class RunbookArgs(BaseModel):
    id: str = Field(description="Runbook id, e.g. RB-05")


class RestartArgs(BaseModel):
    name: str = Field(description="Service to restart")
    ticket_id: str = Field(description="The open ticket this change belongs to")


class RollbackArgs(BaseModel):
    name: str = Field(description="Service to roll back to its previous healthy version")
    ticket_id: str = Field(description="The open ticket this change belongs to")


class ActionArgs(BaseModel):
    service: str = Field(description="Service to run the runbook action on")
    action: str = Field(description="One of clear_cache, rotate_logs, renew_certificate, flush_queue, scale_up")
    ticket_id: str = Field(description="The open ticket this change belongs to")


class TicketArgs(BaseModel):
    title: str = Field(description="Short summary of the problem")
    service: str = Field(description="The affected service")
    priority: int = Field(description="1 (highest) to 4", ge=1, le=4)


class CommentArgs(BaseModel):
    ticket_id: str
    text: str


class CloseArgs(BaseModel):
    ticket_id: str
    resolution: str = Field(description="What fixed it")


class ListTicketsArgs(BaseModel):
    status: str | None = Field(default=None, description="open or closed; omit for all")


class UserArgs(BaseModel):
    user_id: str = Field(description="User id, e.g. u-101")


class UnlockArgs(BaseModel):
    user_id: str
    ticket_id: str = Field(description="The open ticket this change belongs to")


class ApprovalArgs(BaseModel):
    ticket_id: str
    action: str = Field(description="What needs approval, e.g. restart")
    service: str
    reason: str


class FinishArgs(BaseModel):
    answer: str = Field(description="Your final answer or summary of what you did")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    method: str
    path: str  # with {placeholders} from the arguments
    body: tuple[str, ...] = ()  # arguments sent as the JSON body
    query: tuple[str, ...] = ()  # arguments sent as query parameters

    def definition(self) -> dict[str, Any]:
        schema = self.args.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": schema},
        }

    @property
    def read_only(self) -> bool:
        return self.method == "GET"


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("list_services", "List every service with its status and version.", NoArgs, "GET", "/services"),
        Tool(
            "get_service",
            "A service's details: status, version, dependencies, team, deployments.",
            ServiceName,
            "GET",
            "/services/{name}",
        ),
        Tool(
            "get_logs", "Recent log lines of a service.", LogsArgs, "GET", "/services/{service}/logs", query=("lines",)
        ),
        Tool(
            "search_runbooks",
            "Search the runbooks; returns ids and titles.",
            SearchArgs,
            "GET",
            "/runbooks/search",
            query=("query",),
        ),
        Tool("read_runbook", "Read a runbook by id.", RunbookArgs, "GET", "/runbooks/{id}"),
        Tool(
            "restart_service",
            "Restart a service. Needs an open ticket.",
            RestartArgs,
            "POST",
            "/services/{name}/restart",
            body=("ticket_id",),
        ),
        Tool(
            "rollback_service",
            "Roll a service back to its previous healthy version. Needs an open ticket.",
            RollbackArgs,
            "POST",
            "/services/{name}/rollback",
            body=("ticket_id",),
        ),
        Tool(
            "run_action",
            "Run a runbook action on a service. Needs an open ticket.",
            ActionArgs,
            "POST",
            "/services/{service}/actions",
            body=("action", "ticket_id"),
        ),
        Tool(
            "create_ticket",
            "Open a ticket; returns its id.",
            TicketArgs,
            "POST",
            "/tickets",
            body=("title", "service", "priority"),
        ),
        Tool(
            "add_comment",
            "Add a comment to a ticket.",
            CommentArgs,
            "POST",
            "/tickets/{ticket_id}/comments",
            body=("text",),
        ),
        Tool("close_ticket", "Close a ticket.", CloseArgs, "POST", "/tickets/{ticket_id}/close", body=("resolution",)),
        Tool(
            "list_tickets", "List tickets, optionally by status.", ListTicketsArgs, "GET", "/tickets", query=("status",)
        ),
        Tool("get_user", "A user's account: locked or not, MFA enabled or not.", UserArgs, "GET", "/users/{user_id}"),
        Tool(
            "unlock_user",
            "Unlock a user's account. Needs an open ticket.",
            UnlockArgs,
            "POST",
            "/users/{user_id}/unlock",
            body=("ticket_id",),
        ),
        Tool(
            "request_approval",
            "Ask a human to approve an action you may not do yourself.",
            ApprovalArgs,
            "POST",
            "/approvals",
            body=("ticket_id", "action", "service", "reason"),
        ),
        Tool("finish", "End the task with your final answer.", FinishArgs, "NONE", ""),
    ]
}


def definitions() -> list[dict[str, Any]]:
    return [tool.definition() for tool in TOOLS.values()]
