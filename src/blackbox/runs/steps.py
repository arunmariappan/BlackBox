"""Build a run's steps. Until a run has proxy exchanges, every GenAI span (`chat`, `execute_tool`, `embeddings`) is a
step, ordered by start time, and its node is the nearest ancestor span the profile names as a node."""

from typing import TYPE_CHECKING, Any

from blackbox.otlp.decode import SpanData
from blackbox.otlp.genai import SpanView
from blackbox.runs.context import RunContext, StepDraft

if TYPE_CHECKING:
    from blackbox.profiles.base import Profile


def span_step_view(view: SpanView) -> dict[str, Any]:
    out: dict[str, Any] = {"source": "span"}
    if view.model:
        out["model"] = view.model
    if view.system_instructions:
        out["system"] = view.system_instructions
    if view.input_messages:
        out["messages"] = view.input_messages
    if view.output_messages:
        out["reply"] = view.output_messages
    if view.tool_definitions:
        out["tools"] = view.tool_definitions
    if view.finish_reasons:
        out["finish_reasons"] = view.finish_reasons
    if view.tool_name:
        out["tool"] = {
            "name": view.tool_name,
            "call_id": view.tool_call_id,
            "arguments": view.tool_arguments,
            "result": view.tool_result,
        }
    return out


def _has_llm_descendant(ctx: RunContext, span: SpanData) -> bool:
    return any(ctx.view(child).step_kind == "llm" for child in ctx.descendants(span.span_id))


def steps_from_spans(ctx: RunContext, profile: Profile) -> list[StepDraft]:
    candidates = []
    for span in ctx.spans:
        view = ctx.view(span)
        kind = view.step_kind
        if kind is None:
            continue
        if kind == "llm" and _has_llm_descendant(ctx, span):
            continue  # an outer orchestration span; its inner calls are the steps
        candidates.append((span, view, kind))
    candidates.sort(key=lambda item: (item[0].start_ns, item[0].span_id))
    steps: list[StepDraft] = []
    for idx, (span, view, kind) in enumerate(candidates, start=1):
        step = StepDraft(
            idx=idx,
            kind=kind,
            span_id=span.span_id,
            model=view.model,
            tool_name=view.tool_name,
            input_tokens=view.input_tokens,
            output_tokens=view.output_tokens,
            started_ms=span.start_ns // 1_000_000,
            latency_ms=round(span.duration_ms),
            status="error" if span.status_code == "error" else "ok",
            view=span_step_view(view),
        )
        step.node = profile.node_for(step, ctx.ancestors(span.parent_span_id), ctx)
        steps.append(step)
    return steps


def build_steps(ctx: RunContext, profile: Profile) -> list[StepDraft]:
    return steps_from_spans(ctx, profile)
