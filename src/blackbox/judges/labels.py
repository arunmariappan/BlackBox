"""Blind labelling: one run at a time, your verdict saved before the judge's is shown."""

from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.judges.calibrate import sample_order
from blackbox.judges.framework import Judge
from blackbox.store import Store
from blackbox.store.models import Label, Run, Score
from blackbox.util import new_id, now_ms

VALUES = ("pass", "fail", "unsure")


@dataclass(frozen=True)
class Question:
    name: str
    text: str
    judge: str
    profile: str
    applies_when: str | None


def questions(judges: dict[str, Judge]) -> dict[str, Question]:
    texts = {
        "faithful": "Is every factual claim in the answer supported by the excerpts?",
        "relevant": "Does the answer address the question that was asked?",
        "scope_correct": "Was the agent right to answer (or to refuse) this question?",
        "task_success": "Did the agent do the task correctly and within policy?",
        "metrics_correct": "Are the run's metric flags right?",
    }
    out = {}
    for judge in judges.values():
        if judge.label_question and judge.agreement_source == "labels":
            out[judge.label_question] = Question(
                judge.label_question,
                texts.get(judge.label_question, judge.label_question),
                judge.name,
                judge.profile,
                judge.applies_when,
            )
    return out


async def save_label(
    store: Store, run_id: str, question: str, value: str, note: str | None = None, labeler: str = "you"
) -> None:
    if value not in VALUES:
        raise ValueError(f"a label is one of {', '.join(VALUES)}")

    async def op(session: AsyncSession) -> None:
        await session.execute(delete(Label).where(Label.run_id == run_id, Label.question == question))
        session.add(
            Label(
                id=new_id(),
                run_id=run_id,
                question=question,
                value=value,
                note=note or None,
                labeler=labeler,
                created_ms=now_ms(),
            )
        )

    await store.write(op)


async def label_of(store: Store, run_id: str, question: str) -> Label | None:
    async with store.read() as s:
        query = (
            select(Label).where(Label.run_id == run_id, Label.question == question).order_by(Label.created_ms.desc())
        )
        return (await s.execute(query.limit(1))).scalar_one_or_none()


async def queue(store: Store, question: Question, judge: Judge, *, limit: int = 500) -> list[str]:
    """Runs still to label, in order: those where the judge's samples disagreed, then a spread across endings, then
    the rest in a stable random order."""
    async with store.read() as s:
        runs = list(
            (
                await s.execute(
                    select(Run)
                    .where(Run.profile == question.profile, Run.status == "complete")
                    .order_by(Run.started_ms)
                )
            ).scalars()
        )
        labelled = set((await s.execute(select(Label.run_id).where(Label.question == question.name))).scalars().all())
        disagreed = {
            score.run_id
            for score in (
                await s.execute(select(Score).where(Score.kind == "judge", Score.name == judge.name))
            ).scalars()
            if score.details.get("samples_disagree")
        }
    todo = [run for run in runs if run.id not in labelled and judge.applies(run)]
    first = sorted((r for r in todo if r.id in disagreed), key=lambda r: sample_order(r.id))
    rest = [r for r in todo if r.id not in disagreed]
    by_ending: dict[str, list[Run]] = defaultdict(list)
    for run in sorted(rest, key=lambda r: sample_order(r.id)):
        by_ending[run.ending or ""].append(run)
    spread: list[Run] = []
    while any(by_ending.values()):
        for ending in sorted(by_ending):
            if by_ending[ending]:
                spread.append(by_ending[ending].pop(0))
    return [run.id for run in [*first, *spread]][:limit]
