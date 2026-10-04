"""A synthetic PaperPilot `/ask-agentic` run: spans shaped as the plan describes PaperPilot's telemetry, plus the
Ollama, Jina and OpenSearch exchanges its HTTP client spans made.

It is hand-written (spike S2 could not run where this was built). Replace it with the real export
(`paperpilot-agentic-aspire.json`) and a real bundle when they exist; the tests that use it should keep passing.

`python -m tests.fixtures.paperpilot` rewrites `tests/fixtures/otlp/paperpilot-agentic.json`.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from blackbox.otlp.decode import SpanData, encode_json

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
REMOTE_PARENT = "00f067aa0ba902b7"
T0_NS = 1_790_000_000_000_000_000  # 2026-09-21, fixed so the fixture never changes
MODEL = "qwen3.5:4b"
QUESTION = "What are transformer architectures?"
RESOURCE = {"service.name": "api", "telemetry.sdk.language": "dotnet", "telemetry.sdk.name": "opentelemetry"}

GUARDRAIL_SYSTEM = (
    "You decide whether a question is about AI research. Score from 0 to 100 how relevant the question is to "
    'AI research papers. Reply with JSON: {"score": <int>, "reason": <string>}.'
)
GRADING_SYSTEM = 'You grade whether excerpts answer a question. Reply with JSON: {"relevant": "yes" | "no"}.'
REWRITE_SYSTEM = "Rewrite the question so a search engine finds better papers. Reply with the new query only."
ANSWER_SYSTEM = (
    "Answer the question using only the excerpts. Cite papers as [arXiv:<id>]. If the excerpts don't answer it, say so."
)

HITS_FIRST = [
    {"paper_id": "2301.00001", "title": "Convolutional Sequence Models", "score": 3.1, "text": "CNNs for sequences."}
]
HITS_SECOND = [
    {
        "paper_id": "1706.03762",
        "title": "Attention Is All You Need",
        "score": 12.4,
        "text": "The Transformer is based solely on attention mechanisms, dispensing with recurrence entirely.",
    },
    {
        "paper_id": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "score": 10.9,
        "text": "BERT uses a bidirectional Transformer encoder pre-trained with masked language modelling.",
    },
]
ANSWER = (
    "Transformer architectures replace recurrence with self-attention, so every token attends to every other token "
    "[arXiv:1706.03762]. Encoder-only variants such as BERT are pre-trained bidirectionally [arXiv:1810.04805]."
)
OUTPUT = {
    "answer": ANSWER,
    "sources": [
        {"paper_id": "1706.03762", "title": "Attention Is All You Need"},
        {"paper_id": "1810.04805", "title": "BERT: Pre-training of Deep Bidirectional Transformers"},
    ],
    "reasoning_steps": [
        "Validated question relevance (score 92)",
        "Retrieved 1 chunks (attempt 1)",
        "Graded documents: not relevant",
        "Rewrote query: transformer neural network architecture self-attention",
        "Retrieved 2 chunks (attempt 2)",
        "Graded documents: relevant",
        "Generated answer from context",
    ],
    "retrieval_attempts": 2,
    "chunks_used": 2,
    "search_mode": "hybrid",
    "trace_id": TRACE_ID,
}


@dataclass
class Call:
    """One HTTP call PaperPilot makes: its client span, and the exchange the proxy would record."""

    upstream: str
    port: int
    path: str
    request: dict[str, Any]
    response: dict[str, Any]
    span_id: str
    start_ms: int
    end_ms: int


@dataclass
class Builder:
    spans: list[SpanData] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    counter: int = 0

    def next_id(self) -> str:
        self.counter += 1
        return f"{0xA000 + self.counter:016x}"

    def span(
        self,
        name: str,
        parent: str | None,
        start_ms: int,
        end_ms: int,
        *,
        kind: str = "internal",
        scope: str = "PaperPilot.Agentic",
        attributes: dict[str, Any] | None = None,
        status: str = "unset",
    ) -> str:
        span_id = self.next_id()
        self.spans.append(
            SpanData(
                trace_id=TRACE_ID,
                span_id=span_id,
                parent_span_id=parent,
                name=name,
                kind=kind,
                service="api",
                scope=scope,
                start_ns=T0_NS + start_ms * 1_000_000,
                end_ns=T0_NS + end_ms * 1_000_000,
                status_code=status,
                attributes=attributes or {},
                resource=dict(RESOURCE),
            )
        )
        return span_id

    def http(
        self, parent: str, start: int, end: int, upstream: str, port: int, path: str, request: Any, response: Any
    ) -> str:
        span_id = self.span(
            "POST",
            parent,
            start,
            end,
            kind="client",
            scope="System.Net.Http",
            attributes={
                "http.request.method": "POST",
                "url.full": f"http://127.0.0.1:{port}{path}",
                "server.address": "127.0.0.1",
                "server.port": port,
                "http.response.status_code": 200,
            },
        )
        self.calls.append(Call(upstream, port, path, request, response, span_id, start, end))
        return span_id

    def chat(
        self,
        parent: str,
        start: int,
        end: int,
        system: str,
        user: str,
        reply: str,
        tokens: tuple[int, int],
        fmt: Any = None,
    ) -> None:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        chat_id = self.span(
            f"chat {MODEL}",
            parent,
            start,
            end,
            kind="client",
            scope="Experimental.Microsoft.Extensions.AI",
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.system": "ollama",
                "gen_ai.request.model": MODEL,
                "gen_ai.response.model": MODEL,
                "gen_ai.request.temperature": 0.0,
                "gen_ai.usage.input_tokens": tokens[0],
                "gen_ai.usage.output_tokens": tokens[1],
                "gen_ai.response.finish_reasons": ["stop"],
                "gen_ai.input.messages": json.dumps(
                    [{"role": m["role"], "parts": [{"type": "text", "content": m["content"]}]} for m in messages]
                ),
                "gen_ai.output.messages": json.dumps(
                    [{"role": "assistant", "parts": [{"type": "text", "content": reply}], "finish_reason": "stop"}]
                ),
            },
        )
        request: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
        }
        if fmt is not None:
            request["format"] = fmt
        response = {
            "model": MODEL,
            "created_at": "2026-09-21T10:00:00.000000Z",
            "message": {"role": "assistant", "content": reply},
            "done": True,
            "done_reason": "stop",
            "total_duration": (end - start) * 1_000_000,
            "prompt_eval_count": tokens[0],
            "eval_count": tokens[1],
        }
        self.http(chat_id, start + 1, end - 1, "ollama", 8210, "/api/chat", request, response)

    def retrieval(self, parent: str, start: int, query: str, hits: list[dict[str, Any]]) -> None:
        self.http(
            parent,
            start,
            start + 120,
            "jina",
            8212,
            "/v1/embeddings",
            {"model": "jina-embeddings-v3", "task": "retrieval.query", "input": [query]},
            {
                "model": "jina-embeddings-v3",
                "data": [{"index": 0, "embedding": [0.01] * 8}],
                "usage": {"total_tokens": 9},
            },
        )
        self.http(
            parent,
            start + 130,
            start + 180,
            "opensearch",
            8211,
            "/arxiv-papers-chunks/_search",
            {"size": 3, "query": {"hybrid": {"queries": [{"match": {"chunk_text": query}}]}}},
            {
                "took": 12,
                "hits": {
                    "total": {"value": len(hits)},
                    "hits": [
                        {
                            "_id": f"{hit['paper_id']}-0",
                            "_score": hit["score"],
                            "_source": {"arxiv_id": hit["paper_id"], "title": hit["title"], "chunk_text": hit["text"]},
                        }
                        for hit in hits
                    ],
                },
            },
        )


def build() -> Builder:
    b = Builder()
    server = b.span(
        "POST /api/v1/ask-agentic",
        REMOTE_PARENT,
        0,
        9000,
        kind="server",
        scope="Microsoft.AspNetCore",
        attributes={
            "http.request.method": "POST",
            "url.path": "/api/v1/ask-agentic",
            "http.route": "/api/v1/ask-agentic",
            "http.response.status_code": 200,
        },
    )
    request = b.span(
        "agentic_rag_request",
        server,
        5,
        8990,
        attributes={
            "langfuse.trace.input": QUESTION,
            "langfuse.trace.output": json.dumps(OUTPUT),
            "langfuse.trace.metadata.top_k": 3,
            "langfuse.trace.metadata.use_hybrid": True,
            "langfuse.trace.metadata.model": MODEL,
        },
    )
    guard = b.span(
        "guardrail_validation",
        request,
        10,
        900,
        attributes={
            "langfuse.observation.input": QUESTION,
            "langfuse.observation.output": json.dumps({"score": 92, "reason": "about neural network architectures"}),
            "langfuse.observation.level": "DEFAULT",
        },
    )
    b.chat(
        guard,
        20,
        880,
        GUARDRAIL_SYSTEM,
        QUESTION,
        '{"score": 92, "reason": "about neural network architectures"}',
        (210, 18),
        fmt={"type": "object", "properties": {"score": {"type": "integer"}, "reason": {"type": "string"}}},
    )
    retrieve1 = b.span(
        "document_retrieval_initiation", request, 910, 1100, attributes={"langfuse.observation.input": QUESTION}
    )
    b.retrieval(retrieve1, 915, QUESTION, HITS_FIRST)
    grade1 = b.span("document_grading", request, 1110, 1900)
    b.chat(
        grade1,
        1120,
        1880,
        GRADING_SYSTEM,
        f"Question: {QUESTION}\nExcerpts:\n[1] CNNs for sequences.",
        '{"relevant": "no"}',
        (150, 6),
    )
    rewrite = b.span("query_rewriting", request, 1910, 2700)
    new_query = "transformer neural network architecture self-attention"
    b.chat(rewrite, 1920, 2680, REWRITE_SYSTEM, QUESTION, new_query, (90, 9))
    retrieve2 = b.span(
        "document_retrieval_initiation", request, 2710, 2900, attributes={"langfuse.observation.input": new_query}
    )
    b.retrieval(retrieve2, 2715, new_query, HITS_SECOND)
    grade2 = b.span("document_grading", request, 2910, 3800)
    excerpts = "\n".join(f"[{i}] {hit['text']}" for i, hit in enumerate(HITS_SECOND, 1))
    b.chat(
        grade2,
        2920,
        3780,
        GRADING_SYSTEM,
        f"Question: {QUESTION}\nExcerpts:\n{excerpts}",
        '{"relevant": "yes"}',
        (260, 6),
    )
    answer = b.span("answer_generation", request, 3810, 8980)
    b.chat(answer, 3820, 8960, ANSWER_SYSTEM, f"Question: {QUESTION}\nExcerpts:\n{excerpts}", ANSWER, (420, 96))
    return b


FIXTURE = Path(__file__).parent / "otlp" / "paperpilot-agentic.json"


def write_fixture() -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(encode_json(build().spans), indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    write_fixture()
