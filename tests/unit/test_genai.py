from pathlib import Path
from typing import Any

from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from blackbox import sdk
from blackbox.otlp.decode import SpanData, decode_json, decode_protobuf
from blackbox.otlp.genai import view_of

FIXTURES = Path(__file__).parent.parent / "fixtures" / "otlp"


def by_name(spans: list[SpanData], name: str) -> list[SpanData]:
    return [s for s in spans if s.name == name]


def test_paperpilot_fixture() -> None:
    spans = decode_json((FIXTURES / "paperpilot-agentic.json").read_bytes())
    chat = view_of(by_name(spans, "chat qwen3.5:4b")[0])
    assert chat.step_kind == "llm"
    assert set(chat.dialects) >= {"current", "meai"}
    assert chat.provider == "ollama"
    assert chat.model == "qwen3.5:4b"
    assert (chat.input_tokens, chat.output_tokens) == (210, 18)
    assert [m["role"] for m in chat.input_messages] == ["system", "user"]
    assert chat.input_messages[1]["content"] == "What are transformer architectures?"
    assert chat.output_messages[0]["content"].startswith('{"score": 92')
    assert chat.finish_reasons == ["stop"]
    request = view_of(by_name(spans, "agentic_rag_request")[0])
    assert "langfuse" in request.dialects
    assert request.trace_input == "What are transformer architectures?"
    assert request.trace_output["reasoning_steps"][-1] == "Generated answer from context"
    assert request.metadata == {"top_k": 3, "use_hybrid": True, "model": "qwen3.5:4b"}
    guard = view_of(by_name(spans, "guardrail_validation")[0])
    assert guard.observation_output == {"score": 92, "reason": "about neural network architectures"}
    assert guard.level == "DEFAULT"
    http = view_of(by_name(spans, "POST")[0])
    assert http.is_http_client and http.step_kind is None
    assert http.http_url == "http://127.0.0.1:8210/api/chat"
    assert (http.server_port, http.http_status) == (8210, 200)


def test_older_event_dialect() -> None:
    spans = decode_json((FIXTURES / "events-dialect.json").read_bytes())
    first = view_of(spans[1])
    assert "events" in first.dialects
    assert first.provider == "openai"
    assert (first.input_tokens, first.output_tokens) == (41, 17)
    assert [m["role"] for m in first.input_messages] == ["system", "user"]
    assert first.system_instructions == "You are a weather bot."
    assert first.output_messages[0]["tool_calls"] == [
        {"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}}
    ]
    assert first.output_messages[0]["finish_reason"] == "tool_calls"
    second = view_of(spans[2])
    assert second.input_messages[1] == {"role": "tool", "content": "18C, sunny", "tool_call_id": "call_1"}
    assert second.output_messages[0]["content"] == "It is 18C and sunny in Paris."
    root = view_of(spans[0])
    assert root.operation == "invoke_agent" and root.agent_name == "legacy" and root.step_kind is None


class FakeChatClient:
    def chat(self, *, model: str, messages: list[Any], tools: Any = None, **kwargs: Any) -> dict[str, Any]:
        return {
            "model": model,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "get_logs", "arguments": {"service": "checkout-api"}}}],
            },
            "done_reason": "stop",
            "prompt_eval_count": 120,
            "eval_count": 15,
        }


def sdk_spans() -> list[SpanData]:
    exporter = InMemorySpanExporter()
    sdk.init("fake-agent", exporter=exporter, instrument_httpx=False, batch_delay_ms=10)

    @sdk.tool
    def get_logs(service: str, lines: int = 20) -> dict[str, Any]:
        return {"service": service, "lines": ["connection refused to cache:6379"][:lines]}

    tools = [{"type": "function", "function": {"name": "get_logs", "parameters": {"type": "object"}}}]
    try:
        with sdk.agent_run("fake-agent", input={"instruction": "why is checkout failing?"}) as run:
            sdk.traced_chat(
                FakeChatClient(),
                model="qwen3.5:4b",
                messages=[{"role": "system", "content": "You are OpsDesk."}, {"role": "user", "content": "why?"}],
                tools=tools,
                options={"temperature": 0},
            )
            get_logs(service="checkout-api")
            run.set_output({"final_answer": "cache is down"})
            run.set_ending("finished")
        sdk.flush()
        finished = exporter.get_finished_spans()
        return decode_protobuf(encode_spans(finished).SerializeToString())
    finally:
        sdk.shutdown()


def test_sdk_trace() -> None:
    spans = sdk_spans()
    names = sorted(s.name for s in spans)
    assert names == ["chat qwen3.5:4b", "execute_tool get_logs", "invoke_agent fake-agent"]
    chat = view_of(next(s for s in spans if s.name.startswith("chat")))
    assert chat.step_kind == "llm" and chat.provider == "ollama"
    assert chat.system_instructions == "You are OpsDesk."
    assert chat.input_messages == [{"role": "user", "content": "why?"}]
    assert chat.output_messages[0]["tool_calls"] == [
        {"id": None, "name": "get_logs", "arguments": {"service": "checkout-api"}}
    ]
    assert chat.tool_definitions[0]["function"]["name"] == "get_logs"
    assert (chat.input_tokens, chat.output_tokens) == (120, 15)
    tool = view_of(next(s for s in spans if s.name.startswith("execute_tool")))
    assert tool.step_kind == "tool" and tool.tool_name == "get_logs"
    assert tool.tool_arguments == {"service": "checkout-api"}
    assert tool.tool_result == {"service": "checkout-api", "lines": ["connection refused to cache:6379"]}
    root = next(s for s in spans if s.name.startswith("invoke_agent"))
    assert root.parent_span_id is None
    assert root.attributes["blackbox.run.ending"] == "finished"
    assert view_of(root).agent_name == "fake-agent"
