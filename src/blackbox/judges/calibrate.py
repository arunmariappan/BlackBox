"""Calibration: compare each judge version with your labels (or the checker), store its agreement and trust."""

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.judges import agreement
from blackbox.judges.framework import Judge
from blackbox.judges.inputs import Inputs
from blackbox.judges.runner import JudgeRunner
from blackbox.store import Store
from blackbox.store.models import Judge as JudgeRow
from blackbox.store.models import Label, Run, Score


async def truth_labels(store: Store, judge: Judge) -> dict[str, str]:
    """run id → your latest label for the judge's question, or the checker's verdict for checker-calibrated judges."""
    async with store.read() as s:
        if judge.agreement_source == "checker":
            query = (
                select(Score.run_id, Score.label)
                .where(Score.kind == "checker", Score.name == "checker")
                .order_by(Score.created_ms)
            )
        else:
            query = (
                select(Label.run_id, Label.value)
                .where(Label.question == judge.label_question)
                .order_by(Label.created_ms)
            )
        rows = (await s.execute(query)).all()
    return {run_id: str(value) for run_id, value in rows}  # later rows win


async def judge_labels(store: Store, name: str, version: str) -> dict[str, str]:
    async with store.read() as s:
        rows = (
            await s.execute(
                select(Score.run_id, Score.label).where(
                    Score.kind == "judge", Score.name == name, Score.version == version
                )
            )
        ).all()
    return {run_id: str(label) for run_id, label in rows if label in ("pass", "fail")}


async def versions(store: Store, name: str) -> list[JudgeRow]:
    async with store.read() as s:
        query = select(JudgeRow).where(JudgeRow.name == name).order_by(JudgeRow.created_ms)
        return list((await s.execute(query)).scalars())


async def compute(store: Store, judge: Judge, version: str) -> dict[str, Any]:
    truth = await truth_labels(store, judge)
    verdicts = await judge_labels(store, judge.name, version)
    pairs = [agreement.Pair(run_id, verdicts[run_id], truth[run_id]) for run_id in sorted(truth) if run_id in verdicts]
    return agreement.json_safe(agreement.report(pairs))  # type: ignore[no-any-return]


async def store_agreement(store: Store, name: str, version: str, stats: dict[str, Any]) -> None:
    async def op(session: AsyncSession) -> None:
        await session.execute(
            update(JudgeRow)
            .where(JudgeRow.name == name, JudgeRow.version == version)
            .values(agreement=stats, trusted=bool(stats.get("trusted")))
        )

    await store.write(op)


async def calibrate(
    store: Store, runner: JudgeRunner, judge: Judge, *, judge_missing: bool = True
) -> dict[str, dict[str, Any]]:
    """Judge every labelled run with the current version (cached calls make unchanged versions free), then refresh
    the agreement and trust of every version of the judge. Returns version → statistics."""
    await runner.register(judge)
    if judge_missing:
        truth = await truth_labels(store, judge)
        async with store.read() as s:
            runs = list((await s.execute(select(Run).where(Run.id.in_(list(truth))))).scalars()) if truth else []
        for run in runs:
            await runner.judge_run(judge, run)
    out: dict[str, dict[str, Any]] = {}
    for row in await versions(store, judge.name):
        stats = await compute(store, judge, row.version)
        await store_agreement(store, judge.name, row.version, stats)
        out[row.version] = stats
    return out


async def trusted_versions(store: Store) -> dict[str, set[str]]:
    """judge name → its trusted versions."""
    async with store.read() as s:
        rows = (await s.execute(select(JudgeRow.name, JudgeRow.version).where(JudgeRow.trusted.is_(True)))).all()
    out: dict[str, set[str]] = defaultdict(set)
    for name, version in rows:
        out[name].add(version)
    return out


# Stability and sensitivity ------------------------------------------------------------------------------------------


@dataclass
class StabilityResult:
    temperature: float
    runs: int
    flipped: int

    @property
    def flip_rate(self) -> float:
        return self.flipped / self.runs if self.runs else math.nan


async def stability(
    runner: JudgeRunner,
    judge: Judge,
    inputs: list[Inputs],
    *,
    repeats: int = 3,
    temperatures: tuple[float, ...] = (0.0, 0.7),
) -> list[StabilityResult]:
    """Judge the same inputs `repeats` times per temperature without the cache; a run "flips" if its verdicts differ."""
    results = []
    for temperature in temperatures:
        flipped = 0
        for item in inputs:
            labels = set()
            for _ in range(repeats):
                verdict = await runner.decide(
                    judge, item, use_cache=False, options={"temperature": temperature}, samples=1
                )
                labels.add(verdict.samples[0] if verdict.valid and verdict.samples else "invalid")
            flipped += len(labels) > 1
        results.append(StabilityResult(temperature, len(inputs), flipped))
    return results


MADE_UP_CLAIMS = [
    "It was first proposed in 1987 by researchers at the University of Oslo.",
    "The original model was trained on 12,000 GPUs for three years.",
    "This approach won the 2009 Turing Award.",
    "It only works for images smaller than 32 pixels.",
    "Its authors later showed it fails on every language except French.",
]


def with_made_up_claim(inputs: Inputs, index: int) -> Inputs:
    claim = MADE_UP_CLAIMS[index % len(MADE_UP_CLAIMS)]
    return {**inputs, "answer": f"{inputs['answer'].rstrip()} {claim}"}


def sample_order(run_id: str) -> str:
    return hashlib.sha256(f"order:{run_id}".encode()).hexdigest()
