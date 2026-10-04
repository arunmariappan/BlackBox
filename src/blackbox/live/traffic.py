"""`blackbox traffic`: live runs on demand, paced and capped, with `source = traffic`.

Off unless started. It stops at `max_runs` or `max_minutes`, on Ctrl+C, after 3 failed runs in a row (the agent's
request failed, not a run that scored badly), or when free memory drops below 1.5 GB, because this PC has shut down
during long GPU runs.
"""

import asyncio
import random
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import psutil

from blackbox.profiles.base import TrafficCase

CONFIRM_RUNS = 30
CONFIRM_MINUTES = 30
MIN_FREE_BYTES = int(1.5 * 1024**3)
MAX_FAILURES_IN_A_ROW = 3

_RATE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(s|sec|second|m|min|minute|h|hour)\s*$")
_UNIT_SECONDS = {"s": 1, "sec": 1, "second": 1, "m": 60, "min": 60, "minute": 60, "h": 3600, "hour": 3600}


def parse_rate(text: str) -> float:
    """Runs per second from `1/min`, `30/h`, `0.5/min`, `1/s`."""
    match = _RATE.match(text)
    if match is None or float(match.group(1)) <= 0:
        raise ValueError(f"rate must look like 1/min or 30/h, not {text!r}")
    return float(match.group(1)) / _UNIT_SECONDS[match.group(2)]


def needs_confirmation(max_runs: int, max_minutes: float) -> bool:
    return max_runs > CONFIRM_RUNS or max_minutes > CONFIRM_MINUTES


def free_memory() -> int:
    return int(psutil.virtual_memory().available)


@dataclass
class TrafficSummary:
    started: int = 0
    ok: int = 0
    failed: int = 0
    stopped: str = ""
    run_ids: list[str] = field(default_factory=list)


class Traffic:
    def __init__(
        self,
        client: httpx.AsyncClient,
        profile: str,
        cases: Callable[[random.Random], Awaitable[TrafficCase | None]],
        *,
        rate_per_second: float,
        max_runs: int,
        max_minutes: float,
        seed: int | None = None,
        memory: Callable[[], int] = free_memory,
        min_free_bytes: int = MIN_FREE_BYTES,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        report: Callable[[str], None] = print,
    ) -> None:
        self.client = client
        self.profile = profile
        self.cases = cases
        self.interval = 1 / rate_per_second
        self.max_runs = max_runs
        self.max_seconds = max_minutes * 60
        self.rng = random.Random(seed)
        self.memory = memory
        self.min_free_bytes = min_free_bytes
        self.sleep = sleep
        self.clock = clock
        self.report = report
        self.summary = TrafficSummary()

    async def run(self) -> TrafficSummary:
        summary = self.summary
        start = self.clock()
        in_a_row = 0
        for i in range(self.max_runs):
            due = start + i * self.interval
            if due - start >= self.max_seconds:
                summary.stopped = f"time limit ({self.max_seconds / 60:g} min)"
                return summary
            wait = due - self.clock()
            if wait > 0:
                await self.sleep(wait)
            free = self.memory()
            if free < self.min_free_bytes:
                summary.stopped = f"free memory {free / 1024**3:.1f} GB is under {self.min_free_bytes / 1024**3:.1f} GB"
                return summary
            case = await self.cases(self.rng)
            if case is None:
                summary.stopped = f"profile {self.profile} has no traffic inputs"
                return summary
            ok, detail = await self.start_one(case)
            summary.started += 1
            if ok:
                summary.ok += 1
                in_a_row = 0
            else:
                summary.failed += 1
                in_a_row += 1
            self.report(f"[{i + 1}/{self.max_runs}] {'ok' if ok else 'FAILED'} {detail}")
            if in_a_row >= MAX_FAILURES_IN_A_ROW:
                summary.stopped = f"{MAX_FAILURES_IN_A_ROW} failed runs in a row"
                return summary
        summary.stopped = f"{self.max_runs} runs done"
        return summary

    async def start_one(self, case: TrafficCase) -> tuple[bool, str]:
        body = {
            "profile": self.profile,
            "input": case.input,
            "options": case.options,
            "source": "traffic",
            "wait": True,
            "tags": {**case.tags, "traffic": True},
        }
        try:
            response = await self.client.post("/api/runs", json=body, timeout=900)
        except httpx.HTTPError as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if response.status_code >= 400:
            return False, f"HTTP {response.status_code}: {response.text[:200]}"
        started = response.json()
        run_id = str(started.get("run_id"))
        self.summary.run_ids.append(run_id)
        status, error = started.get("status"), started.get("error")
        if error or status is None or int(status) >= 400:
            return False, f"run {run_id}: {error or f'agent answered {status}'}"
        return True, f"run {run_id} {case.tags or case.input[:60]}"
