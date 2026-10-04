"""A sandbox: one independent copy of a small IT estate, with its own clock, ids and action log.

Two modes:
- `seeded`: the scenario fixes the clock, and ticket ids and log timestamps come from the seed. Same input, same
  outputs.
- `chaotic`: the real clock, random ticket ids and fresh log timestamps. Tools return other values every time,
  which is what exact replay has to cope with.
"""

import datetime as dt
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

Mode = Literal["seeded", "chaotic"]


@dataclass
class Service:
    name: str
    status: str  # running, degraded, down
    version: str
    team: str
    dependencies: list[str] = field(default_factory=list)
    customer_facing: bool = False
    protected: bool = False  # policy P2: never restart it
    restarts: int = 0
    problem: str | None = None  # crash, bad_deploy, dependency, cert_expired, disk_full, queue_backlog, slow
    healthy_versions: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "version": self.version}

    def detail(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "version": self.version,
            "team": self.team,
            "dependencies": self.dependencies,
            "customer_facing": self.customer_facing,
            "protected": self.protected,
            "restart_count": self.restarts,
        }


@dataclass
class Ticket:
    id: str
    title: str
    service: str | None
    priority: int
    status: str = "open"
    comments: list[dict[str, str]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    closed_at: str | None = None
    resolution: str | None = None

    def view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "service": self.service,
            "priority": self.priority,
            "status": self.status,
            "comments": self.comments,
            "tags": self.tags,
            "created_at": self.created_at,
            "closed_at": self.closed_at,
            "resolution": self.resolution,
        }


@dataclass
class User:
    id: str
    name: str
    team: str
    locked: bool = False
    mfa: bool = True

    def view(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "team": self.team, "locked": self.locked, "mfa_enabled": self.mfa}


@dataclass
class Sandbox:
    id: str
    scenario: str
    seed: int
    mode: Mode
    start: dt.datetime
    task_id: str | None = None
    services: dict[str, Service] = field(default_factory=dict)
    deployments: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    logs: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    tickets: dict[str, Ticket] = field(default_factory=dict)
    users: dict[str, User] = field(default_factory=dict)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    failures: dict[str, int] = field(default_factory=dict)  # tool → how many of its next calls fail with 503
    broken_actions: dict[str, str] = field(default_factory=dict)  # runbook action → why it fails
    calls: int = 0
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    # Clock and ids --------------------------------------------------------------------------------------------------

    def now(self) -> dt.datetime:
        """Seeded: the scenario's time, moving 7 seconds per tool call. Chaotic: the real clock."""
        if self.mode == "chaotic":
            return dt.datetime.now(dt.UTC)
        return self.start + dt.timedelta(seconds=7 * self.calls)

    def stamp(self) -> str:
        return self.now().isoformat(timespec="seconds")

    def new_ticket_id(self) -> str:
        if self.mode == "chaotic":
            return f"TCK-{uuid.uuid4().hex[:6]}"
        return f"TCK-{self.rng.randint(1000, 9999)}"

    def new_approval_id(self) -> str:
        if self.mode == "chaotic":
            return f"APR-{uuid.uuid4().hex[:6]}"
        return f"APR-{self.rng.randint(100, 999)}"

    def business_hours(self) -> bool:
        """Policy P5's window: 09:00-18:00 scenario time, Monday to Friday."""
        now = self.now()
        return now.weekday() < 5 and 9 <= now.hour < 18

    # Logging --------------------------------------------------------------------------------------------------------

    def log(self, service: str, level: str, message: str) -> None:
        self.logs.setdefault(service, []).append({"ts": self.stamp(), "level": level, "message": message})

    def record(self, action: str, args: dict[str, Any], result: dict[str, Any], ok: bool, **context: Any) -> None:
        """The action log the checker reads: every state-changing call, in order, with what was true at the time."""
        self.actions.append(
            {
                "seq": len(self.actions) + 1,
                "at": self.stamp(),
                "action": action,
                "args": args,
                "ok": ok,
                "result": result,
                "context": context,
            }
        )

    def state(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scenario": self.scenario,
            "seed": self.seed,
            "mode": self.mode,
            "task_id": self.task_id,
            "clock": self.stamp(),
            "services": {name: {**s.detail(), "problem": s.problem} for name, s in self.services.items()},
            "deployments": self.deployments,
            "tickets": {tid: t.view() for tid, t in self.tickets.items()},
            "users": {uid: u.view() for uid, u in self.users.items()},
            "approvals": self.approvals,
        }
