"""PaperPilot's agentic RAG (`POST /api/v1/ask-agentic`), a .NET agent over arXiv papers."""

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from blackbox.datasets import DatasetQuestion
from blackbox.otlp.decode import SpanData
from blackbox.profiles.base import Profile, StartRequest, TrafficCase
from blackbox.runs.context import RunContext, StepDraft

ROOT_SPAN = "agentic_rag_request"
NODES = frozenset(
    {
        "guardrail_validation",
        "document_retrieval_initiation",
        "document_grading",
        "query_rewriting",
        "answer_generation",
    }
)
QUESTIONS = Path("datasets/paperpilot/questions.yaml")
ANSWERABLE_WEIGHT = 3  # traffic draws answerable questions three times as often as the others

ENDING_MARKERS = (
    ("Generated answer from context", "answered"),
    ("Responded as out of scope", "out_of_scope"),
    ("retrieval attempts", "max_attempts"),  # "Stopped after {n} retrieval attempts"
    ("Search was unavailable", "search_unavailable"),
)


class PaperPilotProfile(Profile):
    name = "paperpilot"
    description = "PaperPilot /ask-agentic (agentic RAG over arXiv papers, .NET)"
    node_spans = NODES
    failure_endings = frozenset({"max_attempts", "search_unavailable", "error"})

    def read_only(self, step: StepDraft) -> bool | None:
        return True if step.kind in ("tool", "embedding") else None

    @property
    def base_url(self) -> str:
        return str(self.options.get("base_url", "http://127.0.0.1:8100")).rstrip("/")

    def matches(self, spans: Sequence[SpanData]) -> bool:
        return any(span.name == ROOT_SPAN for span in spans)

    def parse_input(self, text: str, **options: Any) -> dict[str, Any]:
        run_input: dict[str, Any] = {
            "query": text,
            "top_k": int(options.get("top_k") or self.options.get("top_k", 3)),
            "use_hybrid": bool(options.get("use_hybrid", self.options.get("use_hybrid", True))),
            "model": str(options.get("model") or self.options.get("model", "qwen3.5:4b")),
        }
        categories = options.get("categories")
        if categories:
            run_input["categories"] = list(categories)
        return run_input

    def traffic_questions(self) -> list[DatasetQuestion]:
        """The question set traffic draws from (`questions` option), drafts included: only the text matters."""
        from blackbox.datasets import load_dataset

        path = Path(str(self.options.get("questions", QUESTIONS)))
        return load_dataset(path).questions if path.exists() else []

    async def traffic_case(self, rng: random.Random) -> TrafficCase | None:
        """A question from the question set, weighted towards the answerable ones."""
        questions = self.traffic_questions()
        if not questions:
            return None
        weights = [ANSWERABLE_WEIGHT if q.expected_ending == "answered" else 1 for q in questions]
        question = rng.choices(questions, weights=weights)[0]
        return TrafficCase(question.question, tags={"case": question.id, "expected_ending": question.expected_ending})

    def build_request(self, run_input: dict[str, Any]) -> StartRequest:
        return StartRequest(
            method="POST",
            url=f"{self.base_url}/api/v1/ask-agentic",
            body=run_input,
            headers={"content-type": "application/json"},
            timeout_seconds=float(self.options.get("timeout_seconds", 600)),
        )

    def _request_span(self, ctx: RunContext) -> SpanData | None:
        spans = ctx.spans_named(ROOT_SPAN)
        return spans[0] if spans else None

    def input_text(self, ctx: RunContext) -> str | None:
        body = ctx.entry_body
        if isinstance(body, dict) and isinstance(body.get("query"), str):
            return str(body["query"])
        span = self._request_span(ctx)
        if span is not None:
            value = ctx.view(span).trace_input
            if isinstance(value, dict):
                value = value.get("query", value.get("question"))
            if isinstance(value, str):
                return value
        return None

    def read_output(self, ctx: RunContext) -> Any:
        if ctx.output_body is not None:
            return super().read_output(ctx)
        span = self._request_span(ctx)
        if span is not None:
            return ctx.view(span).trace_output
        return None

    def output_text(self, output: Any) -> str | None:
        if isinstance(output, dict) and isinstance(output.get("answer"), str):
            return str(output["answer"])
        return super().output_text(output)

    def ending(self, ctx: RunContext, output: Any) -> str | None:
        steps = output.get("reasoning_steps") if isinstance(output, dict) else None
        if isinstance(steps, list) and steps:
            last = str(steps[-1])
            for marker, ending in ENDING_MARKERS:
                if marker.lower() in last.lower():
                    return ending
        # No response to read: decide from which node spans ran.
        names = {span.name for span in ctx.spans}
        search_failed = any(
            span.name == "document_retrieval_initiation" and span.status_code == "error" for span in ctx.spans
        )
        if "answer_generation" in names:
            return "answered"
        if search_failed:
            return "search_unavailable"
        if "document_retrieval_initiation" in names:
            return "max_attempts"
        if "guardrail_validation" in names:
            return "out_of_scope"
        request = self._request_span(ctx)
        if request is not None and request.status_code == "error":
            return "error"
        return None

    # Replay ---------------------------------------------------------------------------------------------------------

    COMPARED_FIELDS = ("answer", "sources", "reasoning_steps", "retrieval_attempts", "chunks_used", "search_mode")

    def compare_outputs(self, source: Any, replay: Any) -> dict[str, Any]:
        """Compare the fields that matter; `trace_id` differs by design and is ignored."""
        a = source if isinstance(source, dict) else {}
        b = replay if isinstance(replay, dict) else {}
        fields = {
            key: {"equal": a.get(key) == b.get(key), "source": a.get(key), "replay": b.get(key)}
            for key in self.COMPARED_FIELDS
        }
        equal = all(f["equal"] for f in fields.values()) if (a or b) else source == replay
        return {"equal": equal, "fields": fields}

    def rebuild_input(self, ctx: RunContext) -> StartRequest | None:
        """A question asked in PaperPilot's own UI: rebuild the request from `langfuse.trace.input` and the trace
        metadata (`top_k`, `use_hybrid`, `model`)."""
        span = self._request_span(ctx)
        if span is None:
            return None
        view = ctx.view(span)
        query = view.trace_input
        if isinstance(query, dict):
            query = query.get("query", query.get("question"))
        if not isinstance(query, str):
            return None
        meta = view.metadata
        run_input: dict[str, Any] = {
            "query": query,
            "top_k": int(meta.get("top_k", self.options.get("top_k", 3))),
            "use_hybrid": bool(meta.get("use_hybrid", self.options.get("use_hybrid", True))),
            "model": str(meta.get("model", self.options.get("model", "qwen3.5:4b"))),
        }
        if meta.get("categories"):
            run_input["categories"] = meta["categories"]
        return self.build_request(run_input)
