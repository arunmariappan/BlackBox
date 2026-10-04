import json

from blackbox.proxy.views import view_exchange
from blackbox.store import Store
from tests.fixtures import paperpilot


def ndjson(*items: object) -> bytes:
    return b"".join(json.dumps(item).encode() + b"\n" for item in items)


def test_ollama_chat_plain() -> None:
    request = {
        "model": "qwen3.5:4b",
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        "format": {"type": "object"},
        "options": {"temperature": 0},
        "think": False,
        "stream": False,
    }
    response = {
        "model": "qwen3.5:4b",
        "message": {"role": "assistant", "content": "Hello"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 12,
        "eval_count": 2,
        "total_duration": 1_500_000_000,
        "eval_duration": 300_000_000,
    }
    view = view_exchange("ollama", "POST", "/api/chat", json.dumps(request).encode(), json.dumps(response).encode())
    assert view.kind == "llm" and view.model == "qwen3.5:4b"
    assert (view.input_tokens, view.output_tokens) == (12, 2)
    assert view.view["system"] == "Be brief."
    assert view.view["messages"] == [{"role": "user", "content": "Hi"}]
    assert view.view["reply"] == [{"role": "assistant", "content": "Hello", "finish_reason": "stop"}]
    assert view.view["format"] == {"type": "object"} and view.view["options"] == {"temperature": 0}
    assert view.view["timings"] == {"total_ms": 1500.0, "eval_ms": 300.0}


def test_ollama_chat_streamed_with_tool_calls() -> None:
    request = {"model": "m", "messages": [{"role": "user", "content": "Restart?"}], "stream": True, "tools": [{}]}
    call = {"function": {"name": "restart_service", "arguments": {"name": "cache", "ticket": "T-1"}}}
    body = ndjson(
        {"message": {"role": "assistant", "content": "Let"}, "done": False},
        {"message": {"role": "assistant", "content": " me"}, "done": False},
        {"message": {"role": "assistant", "content": "", "tool_calls": [call]}, "done": False},
        {
            "model": "m",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "prompt_eval_count": 30,
            "eval_count": 9,
        },
    )
    view = view_exchange("ollama", "POST", "/api/chat", json.dumps(request).encode(), body)
    reply = view.view["reply"][0]
    assert reply["content"] == "Let me"
    assert reply["tool_calls"] == [
        {"id": None, "name": "restart_service", "arguments": {"name": "cache", "ticket": "T-1"}}
    ]
    assert view.view["chunks"] == 4 and (view.input_tokens, view.output_tokens) == (30, 9)


def test_ollama_generate() -> None:
    body = ndjson({"response": "4", "done": False}, {"response": "2", "done": True, "eval_count": 2})
    view = view_exchange("ollama", "POST", "/api/generate", b'{"model":"m","prompt":"6*7?","system":"math"}', body)
    assert view.view["messages"] == [{"role": "user", "content": "6*7?"}] and view.view["system"] == "math"
    assert view.view["reply"][0]["content"] == "42"


def test_openai_sse_with_tool_call_deltas() -> None:
    events = [
        {
            "model": "gpt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "id": "c1", "function": {"name": "get_", "arguments": '{"ci'}}]
                    },
                }
            ],
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"name": "weather", "arguments": 'ty": "Paris"}'}}]
                    },
                }
            ]
        },
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
        },
    ]
    body = b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events) + b"data: [DONE]\n\n"
    request = {"model": "gpt", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]}
    view = view_exchange("ollama", "POST", "/v1/chat/completions", json.dumps(request).encode(), body)
    assert view.view["reply"][0]["tool_calls"] == [{"id": "c1", "name": "get_weather", "arguments": {"city": "Paris"}}]
    assert view.view["reply"][0]["finish_reason"] == "tool_calls"
    assert (view.input_tokens, view.output_tokens) == (20, 8) and view.model == "gpt"


def test_openai_plain() -> None:
    response = {
        "model": "gpt",
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }
    view = view_exchange("ollama", "POST", "/v1/chat/completions", b'{"messages":[]}', json.dumps(response).encode())
    assert view.view["reply"] == [{"role": "assistant", "content": "ok", "finish_reason": "stop"}]


def test_embeddings() -> None:
    jina = view_exchange(
        "jina",
        "POST",
        "/v1/embeddings",
        b'{"model":"jina-embeddings-v3","input":["a","b"]}',
        b'{"data":[{"embedding":[0.1,0.2]},{"embedding":[0.3,0.4]}],"usage":{"total_tokens":4}}',
    )
    assert jina.kind == "embedding" and jina.view["inputs"] == ["a", "b"]
    assert (jina.view["count"], jina.view["dimensions"], jina.input_tokens) == (2, 2, 4)
    ollama = view_exchange("ollama", "POST", "/api/embed", b'{"model":"e","input":"x"}', b'{"embeddings":[[1,2,3]]}')
    assert ollama.view["inputs"] == ["x"] and ollama.view["dimensions"] == 3


def test_opensearch_hits() -> None:
    call = next(
        c for c in paperpilot.build().calls if c.upstream == "opensearch" and len(c.response["hits"]["hits"]) == 2
    )
    view = view_exchange(
        "opensearch", "POST", call.path, json.dumps(call.request).encode(), json.dumps(call.response).encode()
    )
    assert view.kind == "tool" and view.tool_name == "search"
    assert view.view["index"] == "arxiv-papers-chunks"
    assert [h["paper_id"] for h in view.view["hits"]] == ["1706.03762", "1810.04805"]
    assert view.view["hits"][0]["title"] == "Attention Is All You Need" and view.view["hits"][0]["score"] == 12.4


def test_unknown_path_is_generic() -> None:
    view = view_exchange("opsdesk-env", "GET", "/services", None, b'[{"name": "cache"}]')
    assert view.kind == "other" and view.view["tool"]["result"] == [{"name": "cache"}]


async def test_steps_from_paperpilot_exchanges(store: Store) -> None:
    run_id = await paperpilot.insert_run(store)
    run = await store.reader.run(run_id)
    assert run is not None and run.replayable and run.ending == "answered" and run.profile == "paperpilot"
    steps = await store.reader.steps(run_id)
    assert [(s.kind, s.node) for s in steps] == [
        ("llm", "guardrail_validation"),
        ("embedding", "document_retrieval_initiation"),
        ("tool", "document_retrieval_initiation"),
        ("llm", "document_grading"),
        ("llm", "query_rewriting"),
        ("embedding", "document_retrieval_initiation"),
        ("tool", "document_retrieval_initiation"),
        ("llm", "document_grading"),
        ("llm", "answer_generation"),
    ]
    spans = {s.span_id: s for s in await store.reader.spans(run.trace_id)}
    for step in steps:
        assert step.exchange_id is not None
        if step.kind == "llm":
            assert step.span_id is not None and spans[step.span_id].name == "chat qwen3.5:4b"
    assert steps[0].view["format"]["properties"]["score"]["type"] == "integer"
    assert steps[2].view["hits"][0]["paper_id"] == "2301.00001"
    assert run.input_tokens == 210 + 150 + 90 + 260 + 420 + 9 + 9
