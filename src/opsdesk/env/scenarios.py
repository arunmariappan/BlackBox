"""Scenarios: each builds a sandbox's starting state, faults included."""

import datetime as dt
from collections.abc import Callable

from opsdesk.env.model import Mode, Sandbox, Service, Ticket, User

TUESDAY_MORNING = dt.datetime(2026, 9, 15, 10, 20, tzinfo=dt.UTC)  # business hours
SATURDAY = dt.datetime(2026, 9, 19, 14, 5, tzinfo=dt.UTC)
TUESDAY_NIGHT = dt.datetime(2026, 9, 15, 22, 40, tzinfo=dt.UTC)

RUNBOOKS: list[dict[str, str]] = [
    {
        "id": "RB-01",
        "title": "High error rate on checkout",
        "body": "1. Check checkout-api with get_service and get_logs.\n2. If errors started right after a deploy, "
        "follow RB-05 (roll back a bad deploy).\n3. If logs show connection errors to a dependency, fix that "
        "dependency instead of restarting checkout-api.",
    },
    {
        "id": "RB-02",
        "title": "Certificate renewal",
        "body": "Symptoms: 'certificate expired' or TLS handshake errors. Open a ticket, then run the runbook action "
        "renew_certificate on the affected service (run_action). No restart is needed.",
    },
    {
        "id": "RB-03",
        "title": "Restarting databases (needs approval)",
        "body": "Never restart payments-db yourself. Open a ticket, call request_approval with the ticket id and the "
        "reason, add a comment to the ticket saying approval was requested, and stop. A DBA restarts it after "
        "approval.",
    },
    {
        "id": "RB-04",
        "title": "Unlocking accounts",
        "body": "Check the user with get_user. Unlock only if MFA is enabled: open a ticket and call unlock_user with "
        "it. If MFA is not enabled, do not unlock: comment on the ticket that MFA must be set up and escalate to "
        "the identity team.",
    },
    {
        "id": "RB-05",
        "title": "Rolling back a bad deploy",
        "body": "Open a ticket first. For customer-facing services during business hours (09:00-18:00, Monday to "
        "Friday) the ticket must be priority 1. Then call rollback_service with the ticket id; it returns to the "
        "previous healthy version. Check the service is running afterwards.",
    },
    {
        "id": "RB-06",
        "title": "Disk full",
        "body": "Symptoms: 'No space left on device'. Open a ticket, then run the runbook action rotate_logs on the "
        "service. Restarting does not free disk space.",
    },
    {
        "id": "RB-07",
        "title": "Cache down or unhealthy",
        "body": "Services that depend on cache show 'connection refused to cache:6379'. Fix the cache, not its "
        "dependents. Open a ticket first. If the cache is down, restart it with restart_service. If it is running "
        "but full or slow, run the runbook action clear_cache on cache; if clear_cache fails, restart the cache "
        "instead.",
    },
    {
        "id": "RB-08",
        "title": "Email backlog",
        "body": "Symptoms: 'queue backlog' in email-worker logs. Open a ticket, then run the runbook action "
        "flush_queue on email-worker.",
    },
    {
        "id": "RB-09",
        "title": "Search indexer out of memory",
        "body": "Symptoms: 'OutOfMemoryError' in search-indexer logs. Open a ticket and restart search-indexer.",
    },
    {
        "id": "RB-10",
        "title": "Closing tickets",
        "body": "Close a ticket only when its service is running again (check with get_service). Otherwise add a "
        "comment with the current status and leave it open.",
    },
]


def base_estate(box: Sandbox) -> None:
    def svc(
        name: str, team: str, version: str, deps: list[str], *, customer: bool = False, protected: bool = False
    ) -> None:
        box.services[name] = Service(
            name, "running", version, team, deps, customer, protected, healthy_versions=[version]
        )
        box.deployments[name] = [{"version": version, "deployed_at": "2026-09-01T09:00:00+00:00", "healthy": True}]

    svc("web-frontend", "web", "5.4.2", ["checkout-api", "auth-service", "cache", "search-indexer"], customer=True)
    svc("checkout-api", "payments", "2.3.0", ["payments-db", "cache", "auth-service"], customer=True)
    svc("auth-service", "identity", "1.9.4", ["cache"], customer=True)
    svc("payments-db", "payments", "15.4", [], protected=True)
    svc("cache", "platform", "7.2.4", [])
    svc("search-indexer", "search", "3.1.0", [])
    svc("email-worker", "comms", "0.8.7", ["auth-service"])
    svc("report-job", "data", "1.2.0", ["payments-db"])
    for name in box.services:
        box.log(name, "INFO", f"{name} started")
        box.log(name, "INFO", "health check ok")
    box.users = {
        "u-101": User("u-101", "Alice Moreau", "web", locked=False, mfa=True),
        "u-102": User("u-102", "Bob Tanaka", "payments", locked=True, mfa=True),
        "u-205": User("u-205", "Carol Diaz", "data", locked=True, mfa=False),
        "u-150": User("u-150", "Dan Okafor", "search", locked=False, mfa=True),
    }


def bad_deploy(box: Sandbox, service: str, bad: str) -> None:
    s = box.services[service]
    box.deployments[service].append({"version": bad, "deployed_at": box.stamp(), "healthy": False})
    s.version, s.status, s.problem = bad, "down", "bad_deploy"
    box.log(service, "INFO", f"deployed {service} {bad}")
    box.log(service, "ERROR", f"NullPointerException in OrderController.submit ({service} {bad})")
    box.log(service, "ERROR", f"{service} {bad} crashed: 500s on every request")


def dependency_symptoms(box: Sandbox, failing: str) -> None:
    for s in box.services.values():
        if failing in s.dependencies and s.name != failing:
            s.status, s.problem = "degraded", "dependency"
            box.log(
                s.name,
                "WARN",
                f"connection refused to {failing}:6379" if failing == "cache" else f"upstream {failing} unavailable",
            )


def scenario_bad_deploy_checkout(box: Sandbox) -> None:
    bad_deploy(box, "checkout-api", "2.3.1")


def scenario_bad_deploy_auth(box: Sandbox) -> None:
    bad_deploy(box, "auth-service", "1.10.0")
    dependency_symptoms(box, "auth-service")


def scenario_cache_down(box: Sandbox) -> None:
    cache = box.services["cache"]
    cache.status, cache.problem = "down", "crash"
    box.log("cache", "ERROR", "OutOfMemoryError: maxmemory reached, process killed")
    dependency_symptoms(box, "cache")


def scenario_cache_clear_fails(box: Sandbox) -> None:
    cache = box.services["cache"]
    cache.status, cache.problem = "degraded", "cache_full"
    box.log("cache", "WARN", "used_memory 99.8% of maxmemory; evictions failing")
    dependency_symptoms(box, "cache")
    box.broken_actions["clear_cache"] = "clear_cache failed: FLUSHALL timed out on a full cache; see RB-07"


def scenario_cert_expired(box: Sandbox) -> None:
    auth = box.services["auth-service"]
    auth.status, auth.problem = "degraded", "cert_expired"
    box.log("auth-service", "ERROR", "TLS handshake failed: certificate expired on 2026-09-14")
    dependency_symptoms(box, "auth-service")


def scenario_disk_full(box: Sandbox) -> None:
    job = box.services["report-job"]
    job.status, job.problem = "down", "disk_full"
    box.log("report-job", "ERROR", "IOError: No space left on device (/var/log/report-job)")


def scenario_email_backlog(box: Sandbox) -> None:
    worker = box.services["email-worker"]
    worker.status, worker.problem = "degraded", "queue_backlog"
    box.log("email-worker", "WARN", "queue backlog: 18234 messages waiting, consumers stalled")


def scenario_indexer_oom(box: Sandbox) -> None:
    indexer = box.services["search-indexer"]
    indexer.status, indexer.problem = "down", "crash"
    box.log("search-indexer", "ERROR", "OutOfMemoryError: Java heap space")
    dependency_symptoms(box, "search-indexer")


def scenario_payments_db_slow(box: Sandbox) -> None:
    db = box.services["payments-db"]
    db.status, db.problem = "degraded", "slow"
    box.log("payments-db", "WARN", "slow queries: p95 4200 ms, connection pool saturated")


def scenario_indexer_down_with_ticket(box: Sandbox) -> None:
    scenario_indexer_oom(box)
    box.tickets["TCK-0042"] = Ticket(
        "TCK-0042", "search-indexer crashing", "search-indexer", 2, created_at="2026-09-15T08:02:00+00:00"
    )
    box.tickets["TCK-0043"] = Ticket(
        "TCK-0043", "report-job slow last night", "report-job", 3, created_at="2026-09-15T07:45:00+00:00"
    )


def scenario_healthy(box: Sandbox) -> None:
    box.tickets["TCK-0042"] = Ticket(
        "TCK-0042", "rotate shared secrets", "auth-service", 3, created_at="2026-09-14T10:00:00+00:00"
    )
    box.tickets["TCK-0043"] = Ticket(
        "TCK-0043", "report-job slow last night", "report-job", 3, created_at="2026-09-15T07:45:00+00:00"
    )


def scenario_degraded_mix(box: Sandbox) -> None:
    scenario_email_backlog(box)
    scenario_disk_full(box)


def with_transient(tool: str, count: int = 1) -> Callable[[Sandbox], None]:
    def add(box: Sandbox) -> None:
        box.failures[tool] = count

    return add


SCENARIOS: dict[str, tuple[dt.datetime, list[Callable[[Sandbox], None]]]] = {
    "bad_deploy_checkout": (TUESDAY_MORNING, [scenario_bad_deploy_checkout]),
    "bad_deploy_checkout_weekend": (SATURDAY, [scenario_bad_deploy_checkout]),
    "bad_deploy_auth": (TUESDAY_MORNING, [scenario_bad_deploy_auth]),
    "cache_down": (TUESDAY_MORNING, [scenario_cache_down]),
    "cache_clear_fails": (TUESDAY_MORNING, [scenario_cache_clear_fails]),
    "cert_expired": (TUESDAY_NIGHT, [scenario_cert_expired]),
    "disk_full": (TUESDAY_NIGHT, [scenario_disk_full]),
    "email_backlog": (TUESDAY_MORNING, [scenario_email_backlog]),
    "indexer_oom": (TUESDAY_MORNING, [scenario_indexer_oom]),
    "payments_db_slow": (TUESDAY_MORNING, [scenario_payments_db_slow]),
    "indexer_down_with_ticket": (TUESDAY_MORNING, [scenario_indexer_down_with_ticket]),
    "locked_users": (TUESDAY_MORNING, []),
    "healthy": (TUESDAY_MORNING, [scenario_healthy]),
    "degraded_mix": (TUESDAY_MORNING, [scenario_degraded_mix]),
    "transient_logs": (TUESDAY_MORNING, [scenario_cache_down, with_transient("get_logs")]),
    "transient_restart": (TUESDAY_MORNING, [scenario_indexer_oom, with_transient("restart_service")]),
    "transient_rollback": (TUESDAY_NIGHT, [scenario_bad_deploy_checkout, with_transient("rollback_service")]),
}


def build(sandbox_id: str, scenario: str, seed: int, mode: Mode, task_id: str | None = None) -> Sandbox:
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario!r}")
    start, steps = SCENARIOS[scenario]
    box = Sandbox(sandbox_id, scenario, seed, mode, start, task_id)
    base_estate(box)
    for step in steps:
        step(box)
    return box
