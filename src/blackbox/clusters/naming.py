"""Cluster names and likely causes. The model sees the members closest to the centre and statistics over all of
them, and returns a title, a likely cause with evidence, and a suggested fix. Every piece of evidence must name a
run of the cluster and a step that exists in it; invalid evidence is dropped, and a cause left with no valid
evidence is marked `unsupported`."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from blackbox.llm.client import OllamaJSON


class Evidence(BaseModel):
    run: str = Field(description="The run's label, e.g. R2")
    step: int | None = Field(description="The step number (#n) that shows it, or null")
    observation: str


class ClusterName(BaseModel):
    title: str = Field(description="A short name for this kind of failure, under 10 words")
    likely_cause: str = Field(description="The most likely cause, in one or two sentences")
    evidence: list[Evidence] = Field(description="Two or three observations from the runs that support the cause")
    suggested_fix: str = Field(description="One concrete change that would likely fix it")


@dataclass
class Member:
    run_id: str
    label: str  # R1, R2, ... (short labels are easier for a small model to copy than ULIDs)
    description: str
    digest: str
    steps: int


@dataclass
class Naming:
    title: str | None
    likely_cause: str | None
    suggested_fix: str | None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    status: str = "supported"  # supported, unsupported, invalid


PROMPT = """These failed runs of an AI agent were grouped together because they look alike. Name the group, say
what most likely causes it, back the cause with evidence from the runs, and suggest one fix.

Statistics over all {count} runs in the group:
{stats}

The runs closest to the group's centre:
{members}
"""


def validate_evidence(
    evidence: list[Evidence], members: list[Member]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_label = {m.label: m for m in members}
    by_id = {m.run_id: m for m in members}
    valid, dropped = [], []
    for item in evidence:
        member = by_label.get(item.run.strip()) or by_id.get(item.run.strip())
        entry = {
            "run_id": member.run_id if member else None,
            "label": item.run,
            "step": item.step,
            "observation": item.observation,
        }
        if member is None:
            dropped.append({**entry, "reason": "run not in the cluster"})
        elif item.step is not None and not 1 <= item.step <= member.steps:
            dropped.append({**entry, "reason": f"run has no step {item.step}"})
        else:
            valid.append(entry)
    return valid, dropped


async def name_cluster(llm: OllamaJSON, members: list[Member], stats_text: str, count: int) -> Naming:
    shown = "\n\n".join(f"{m.label} ({m.steps} steps): {m.description}\n{m.digest}" for m in members)
    prompt = PROMPT.replace("{count}", str(count)).replace("{stats}", stats_text).replace("{members}", shown)
    result = await llm.call([{"role": "user", "content": prompt}], ClusterName)
    if not result.valid or result.parsed is None:
        return Naming(None, None, None, status="invalid")
    parsed = result.parsed
    valid, dropped = validate_evidence(parsed.evidence, members)
    return Naming(
        title=parsed.title.strip()[:120],
        likely_cause=parsed.likely_cause.strip(),
        suggested_fix=parsed.suggested_fix.strip(),
        evidence=valid,
        dropped=dropped,
        status="supported" if valid else "unsupported",
    )
