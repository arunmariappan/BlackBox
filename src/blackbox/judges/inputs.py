"""Judge input builders. They read the run (its recorded steps and output), not the agent's own logs, so a judge sees
exactly what the agent saw. Each has a version that is part of the judge's version."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from blackbox.runs.context import RunContext

type Inputs = dict[str, str]


@dataclass(frozen=True)
class InputBuilder:
    name: str
    version: str
    build: Callable[[RunContext], Inputs | None]  # None: the run lacks what this judge needs

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"


BUILDERS: dict[str, InputBuilder] = {}


def builder(name: str, version: str) -> Callable[[Callable[[RunContext], Inputs | None]], InputBuilder]:
    def register(fn: Callable[[RunContext], Inputs | None]) -> InputBuilder:
        built = InputBuilder(name, version, fn)
        BUILDERS[name] = built
        return built

    return register


# PaperPilot ---------------------------------------------------------------------------------------------------------


def run_output(ctx: RunContext) -> Any:
    if ctx.output_json is not None:
        return ctx.output_json
    return ctx.output_body.decode("utf-8", errors="replace") if ctx.output_body else None


def question(ctx: RunContext) -> str | None:
    body = ctx.entry_body
    if isinstance(body, dict) and isinstance(body.get("query"), str):
        return str(body["query"])
    return ctx.run.input_text


def answer(ctx: RunContext) -> str | None:
    output = run_output(ctx)
    if isinstance(output, dict) and isinstance(output.get("answer"), str):
        return str(output["answer"])
    if isinstance(output, str):
        return output
    return ctx.run.output_text


def last_search_hits(ctx: RunContext) -> list[dict[str, Any]] | None:
    """The hits of the run's last search step, as recorded at the proxy."""
    for step in reversed(ctx.steps):
        if step.kind == "tool" and isinstance(step.view.get("hits"), list):
            return list(step.view["hits"])
    return None


def excerpts(hits: list[dict[str, Any]]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        paper = hit.get("paper_id") or hit.get("id") or "?"
        title = hit.get("title") or ""
        lines.append(f"[{i}] arXiv:{paper} {title}\n{hit.get('text') or ''}".strip())
    return "\n\n".join(lines)


@builder("pp_faithfulness", "1")
def pp_faithfulness(ctx: RunContext) -> Inputs | None:
    hits = last_search_hits(ctx)
    q, a = question(ctx), answer(ctx)
    if hits is None or q is None or a is None:
        return None
    return {"question": q, "context": excerpts(hits), "answer": a}


@builder("pp_relevance", "1")
def pp_relevance(ctx: RunContext) -> Inputs | None:
    q, a = question(ctx), answer(ctx)
    if q is None or a is None:
        return None
    return {"question": q, "answer": a}


@builder("pp_scope", "1")
def pp_scope(ctx: RunContext) -> Inputs | None:
    q = question(ctx)
    return {"question": q} if q is not None else None


# OpsDesk ------------------------------------------------------------------------------------------------------------


def _compact(value: Any, limit: int = 1500) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 1] + "…"


@builder("ops_task", "1")
def ops_task(ctx: RunContext) -> Inputs | None:
    body = ctx.entry_body
    instruction = body.get("instruction") if isinstance(body, dict) else ctx.run.input_text
    if not isinstance(instruction, str):
        return None
    calls: list[str] = []
    for step in ctx.steps:
        if step.kind != "tool":
            continue
        tool = step.view.get("tool") or {}
        name = tool.get("name") or step.tool_name
        calls.append(
            f"{len(calls) + 1}. {name}({_compact(tool.get('arguments'), 400)}) -> {_compact(tool.get('result'), 800)}"
        )
    output = run_output(ctx)
    final = output.get("final_answer") if isinstance(output, dict) else output
    return {
        "instruction": instruction,
        "tool_calls": "\n".join(calls) if calls else "(no tool calls)",
        "final_answer": _compact(final or "(no final answer)", 3000),
    }
