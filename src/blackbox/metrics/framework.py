"""Run metrics: plain functions over a run's steps, spans, output and (for OpsDesk) its task and checker result.

They take milliseconds, need no model, and are stored as `metric` scores whose details name the steps involved.
Each metric module has a version; `blackbox metrics recompute` replaces stored scores after a change.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.runs.context import ExchangeData, RunContext, StepDraft, load_run_context
from blackbox.store import Store
from blackbox.store.models import Run, Score
from blackbox.util import canonical_json, new_id, now_ms

if TYPE_CHECKING:
    from blackbox.profiles import ProfileRegistry
    from blackbox.profiles.base import Profile


@dataclass
class MetricValue:
    name: str
    value: float | None
    label: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    flag: bool = False  # this value says something went wrong (badges, filters, failure detection)


@dataclass
class MetricInput:
    ctx: RunContext
    profile: Profile | None
    scores: list[Score]

    @property
    def run(self) -> Run:
        return self.ctx.run

    @property
    def checker(self) -> Score | None:
        return next((s for s in self.scores if s.kind == "checker"), None)

    @cached_property
    def exchanges(self) -> dict[str, ExchangeData]:
        return {e.id: e for e in self.ctx.exchanges}

    def exchange(self, step: StepDraft) -> ExchangeData | None:
        return self.exchanges.get(step.exchange_id) if step.exchange_id else None


@dataclass(frozen=True)
class MetricModule:
    name: str
    version: str
    compute: Callable[[MetricInput], list[MetricValue]]
    profiles: frozenset[str] | None = None  # None: every profile

    @property
    def tag(self) -> str:
        return f"{self.name}-{self.version}"

    def applies(self, profile: str | None) -> bool:
        return self.profiles is None or profile in self.profiles


MODULES: list[MetricModule] = []


def module(
    name: str, version: str, profiles: set[str] | None = None
) -> Callable[[Callable[[MetricInput], list[MetricValue]]], Callable[[MetricInput], list[MetricValue]]]:
    def register(fn: Callable[[MetricInput], list[MetricValue]]) -> Callable[[MetricInput], list[MetricValue]]:
        MODULES.append(MetricModule(name, version, fn, frozenset(profiles) if profiles else None))
        return fn

    return register


# Step signatures ----------------------------------------------------------------------------------------------------


def tool_arguments(step: StepDraft) -> Any:
    view = step.view
    if isinstance(view.get("tool"), dict):
        return view["tool"].get("arguments")
    if "query" in view:
        return view.get("query")
    if "inputs" in view:
        return view.get("inputs")
    return None


def step_signature(step: StepDraft, exchange: ExchangeData | None) -> str:
    """The tool name and its arguments with sorted keys, or a model call's request key. Arguments are the agent's
    own (the recorded request), so ids mapped by aliasing in a fork keep the tape's value and compare equal."""
    if step.kind == "llm":
        if exchange is not None:
            return f"llm:{exchange.row.request_key}"
        return "llm:" + canonical_json(step.view.get("messages"))
    name = step.tool_name or step.view.get("tool", {}).get("name") or step.kind
    return f"{name}:{canonical_json(tool_arguments(step))}"


def http_status(inp: MetricInput, step: StepDraft) -> int | None:
    exchange = inp.exchange(step)
    return exchange.row.status if exchange is not None else None


# Computing and storing ------------------------------------------------------------------------------------------------


def compute(inp: MetricInput) -> list[tuple[MetricModule, MetricValue]]:
    out: list[tuple[MetricModule, MetricValue]] = []
    profile_name = inp.run.profile
    for mod in MODULES:
        if mod.applies(profile_name):
            out.extend((mod, value) for value in mod.compute(inp))
    return out


async def compute_and_store(store: Store, profiles: ProfileRegistry, run: Run) -> list[MetricValue]:
    ctx = await load_run_context(store, run)
    async with store.read() as s:
        scores = list((await s.execute(select(Score).where(Score.run_id == run.id))).scalars())
    inp = MetricInput(ctx, profiles.find(run.profile), scores)
    results = compute(inp)
    now = now_ms()

    async def op(session: AsyncSession) -> None:
        await session.execute(delete(Score).where(Score.run_id == run.id, Score.kind == "metric"))
        for mod, value in results:
            details = {**value.details, "flag": value.flag}
            session.add(
                Score(
                    id=new_id(),
                    run_id=run.id,
                    kind="metric",
                    name=value.name,
                    version=mod.tag,
                    value=value.value,
                    label=value.label,
                    details=json.loads(json.dumps(details, default=str)),
                    created_ms=now,
                )
            )

    await store.write(op)
    return [value for _, value in results]
