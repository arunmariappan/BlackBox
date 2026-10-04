"""OpsDesk's tasks: YAML files with an instruction, a scenario and structured checks (never evaluated code)."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

TASKS_DIR = Path(__file__).parent

Category = Literal["diagnose_and_fix", "information", "policy_trap", "error_recovery", "dependencies"]
READ_ONLY_TOOLS = frozenset(
    {"list_services", "get_service", "get_logs", "search_runbooks", "read_runbook", "list_tickets", "get_user"}
)
CHANGE_TOOLS = frozenset(
    {
        "restart_service",
        "rollback_service",
        "run_action",
        "create_ticket",
        "add_comment",
        "close_ticket",
        "unlock_user",
        "request_approval",
    }
)
ALL_TOOLS = READ_ONLY_TOOLS | CHANGE_TOOLS | {"finish"}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class StateCheck(Strict):
    path: str
    equals: Any = None
    in_: list[Any] | None = Field(default=None, alias="in")
    not_equals: Any = None


class TicketCheck(Strict):
    service: str | None = None
    status_in: list[str] = Field(default_factory=lambda: ["open", "closed"])
    priority: int | None = None
    min_count: int = 1
    max_count: int | None = None
    comment_contains_any: list[str] = Field(default_factory=list)
    new_only: bool = True  # only tickets created during the run, not the scenario's own


class ActionMatch(Strict):
    action: str  # restart, rollback, action, create_ticket, comment, close_ticket, unlock, request_approval
    service: str | None = None
    user_id: str | None = None
    ticket_id: str | None = None


class AnswerCheck(Strict):
    contains_any: list[str] = Field(default_factory=list)
    contains_all: list[str] = Field(default_factory=list)
    regex: str | None = None


class Checks(Strict):
    state: list[StateCheck] = Field(default_factory=list)
    tickets: list[TicketCheck] = Field(default_factory=list)
    policies: list[Literal["P1", "P2", "P3", "P4", "P5"]] = Field(default_factory=list)
    forbidden_actions: list[ActionMatch] = Field(default_factory=list)
    required_actions: list[ActionMatch] = Field(default_factory=list)
    answer: AnswerCheck | None = None
    no_changes: bool = False


class Task(Strict):
    id: str
    category: Category
    instruction: str
    scenario: str
    seed: int = 0
    max_steps: int = 15
    expected_tools: list[str]
    optimal_steps: int
    wrong_targets: list[ActionMatch] = Field(default_factory=list)
    checks: Checks


def load_task(path: Path) -> Task:
    return Task.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_tasks(directory: Path = TASKS_DIR) -> dict[str, Task]:
    tasks = {}
    for path in sorted(directory.glob("*.yaml")):
        task = load_task(path)
        if task.id != path.stem:
            raise ValueError(f"{path.name}: id {task.id!r} must match the file name")
        unknown = set(task.expected_tools) - ALL_TOOLS
        if unknown:
            raise ValueError(f"{path.name}: unknown tools {sorted(unknown)}")
        tasks[task.id] = task
    return tasks
