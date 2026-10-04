"""PaperPilot metrics, from its steps and output."""

import json
import re
from typing import Any

from blackbox.metrics.framework import MetricInput, MetricValue, module

CITATION = re.compile(r"\[arXiv:\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)\]", re.IGNORECASE)


def _reply_text(step: Any) -> str:
    reply = step.view.get("reply") or []
    return str(reply[0].get("content") or "") if reply and isinstance(reply[0], dict) else ""


def _reply_json(step: Any) -> Any:
    text = _reply_text(step)
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except ValueError:
                return None
    return None


def graded_relevant(step: Any) -> bool:
    parsed = _reply_json(step)
    if isinstance(parsed, dict):
        value = parsed.get("relevant", parsed.get("binary_score", parsed.get("score")))
        return str(value).strip().lower() in ("yes", "true", "relevant")
    return bool(re.search(r"\byes\b", _reply_text(step), re.IGNORECASE))


@module("paperpilot", "1", profiles={"paperpilot"})
def paperpilot(inp: MetricInput) -> list[MetricValue]:
    steps = inp.ctx.steps
    output = inp.profile.read_output(inp.ctx) if inp.profile is not None else inp.ctx.output_json
    output = output if isinstance(output, dict) else {}
    searches = [s for s in steps if s.kind == "tool" and isinstance(s.view.get("hits"), list)]
    attempts = output.get("retrieval_attempts")
    if not isinstance(attempts, int):
        attempts = len(searches) or len({s.span_id for s in inp.ctx.spans if s.name == "document_retrieval_initiation"})
    rewrites = [s for s in steps if s.kind == "llm" and s.node == "query_rewriting"]
    gradings = [s for s in steps if s.kind == "llm" and s.node == "document_grading"]
    relevant = [s for s in gradings if graded_relevant(s)]
    out = [
        MetricValue("retrieval_attempts", attempts),
        MetricValue("rewrites", len(rewrites), details={"steps": [s.idx for s in rewrites]}),
        MetricValue(
            "empty_retrievals",
            sum(1 for s in searches if not s.view["hits"]),
            details={"steps": [s.idx for s in searches if not s.view["hits"]]},
        ),
        MetricValue(
            "relevant_gradings", len(relevant), details={"steps": [s.idx for s in relevant], "gradings": len(gradings)}
        ),
    ]
    guardrail = next((s for s in steps if s.kind == "llm" and s.node == "guardrail_validation"), None)
    if guardrail is not None:
        parsed = _reply_json(guardrail)
        score = parsed.get("score") if isinstance(parsed, dict) else None
        out.append(
            MetricValue(
                "guardrail_score",
                float(score) if isinstance(score, int | float) else None,
                details={"step": guardrail.idx},
            )
        )
    # A rewrite after a grading that said "yes" (PaperPilot's B5 says this never happens).
    wasted = [
        r.idx
        for r in rewrites
        if any(
            g.idx < r.idx and graded_relevant(g) and not any(g.idx < x.idx < r.idx for x in searches) for g in gradings
        )
    ]
    out.append(MetricValue("wasted_rewrite", len(wasted), details={"steps": wasted}, flag=bool(wasted)))
    if inp.run.ending == "answered":
        answer = str(output.get("answer") or inp.run.output_text or "")
        cited = CITATION.findall(answer)
        sources = {
            str(s.get("paper_id") or s.get("arxiv_id") or s.get("id"))
            for s in output.get("sources") or []
            if isinstance(s, dict)
        }
        if not sources:
            sources = {str(h.get("paper_id")) for s in searches for h in s.view["hits"]}
        unknown = sorted({c for c in cited if c.split("v")[0] not in sources and c not in sources})
        out.append(MetricValue("citations_present", 1.0 if cited else 0.0, details={"cited": cited}, flag=not cited))
        out.append(
            MetricValue(
                "citations_valid",
                0.0 if unknown or not cited else 1.0,
                details={"not_in_sources": unknown},
                flag=bool(unknown),
            )
        )
    expected = (inp.run.tags or {}).get("expected_ending")
    if expected:
        mismatch = expected != inp.run.ending
        out.append(MetricValue("scope_mismatch", 1.0 if mismatch else 0.0, label=f"expected {expected}", flag=mismatch))
    return out
