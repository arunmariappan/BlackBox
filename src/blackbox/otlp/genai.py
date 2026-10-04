"""Read GenAI meaning out of spans, whatever dialect wrote them.

Each adapter reads one dialect into a `SpanView`:

- `current`: the OpenTelemetry GenAI semantic conventions as of v1.37 (`gen_ai.input.messages`,
  `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.tool.definitions` as JSON attributes, with
  `parts`), which BlackBox's SDK writes;
- `events`: the older per-message span events (`gen_ai.user.message`, `gen_ai.choice`, ...) and the oldest
  `gen_ai.content.prompt` / `gen_ai.content.completion` events;
- `meai`: Microsoft.Extensions.AI (PaperPilot), which uses the two above plus `gen_ai.system`;
- `langfuse`: Langfuse's attributes on PaperPilot's node spans;
- `http`: HTTP client spans (new and old attribute names).

Attributes no adapter reads stay in the stored span, so a new adapter can be run over old spans later.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from blackbox.otlp.decode import SpanData

LLM_OPERATIONS = {"chat", "text_completion", "generate_content"}
TOOL_OPERATIONS = {"execute_tool"}
EMBEDDING_OPERATIONS = {"embeddings"}

type Message = dict[str, Any]


@dataclass
class SpanView:
    operation: str | None = None
    provider: str | None = None
    request_model: str | None = None
    response_model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    system_instructions: str | None = None
    input_messages: list[Message] = field(default_factory=list)
    output_messages: list[Message] = field(default_factory=list)
    tool_definitions: list[Any] = field(default_factory=list)
    finish_reasons: list[str] = field(default_factory=list)
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_arguments: Any = None
    tool_result: Any = None
    agent_name: str | None = None
    observation_input: Any = None
    observation_output: Any = None
    level: str | None = None
    level_message: str | None = None
    trace_input: Any = None
    trace_output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    http_method: str | None = None
    http_url: str | None = None
    http_status: int | None = None
    server_address: str | None = None
    server_port: int | None = None
    dialects: list[str] = field(default_factory=list)

    @property
    def step_kind(self) -> str | None:
        if self.operation in LLM_OPERATIONS:
            return "llm"
        if self.operation in TOOL_OPERATIONS:
            return "tool"
        if self.operation in EMBEDDING_OPERATIONS:
            return "embedding"
        return None

    @property
    def model(self) -> str | None:
        return self.response_model or self.request_model

    @property
    def is_http_client(self) -> bool:
        return self.http_method is not None and (self.http_url is not None or self.server_address is not None)


def maybe_json(value: Any) -> Any:
    """Parse strings that hold JSON objects or arrays; leave everything else alone."""
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in ("{", "[") and text[-1:] in ("}", "]"):
            try:
                return json.loads(text)
            except ValueError:
                return value
    return value


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            _text_of(part.get("content", part.get("text")) if isinstance(part, dict) else part) for part in content
        )
    if isinstance(content, dict):
        return _text_of(content.get("content", content.get("text")))
    return str(content)


def message_from_parts(raw: dict[str, Any]) -> Message:
    """A conventions-style message (`role` and `parts`) into BlackBox's flat form."""
    message: Message = {"role": raw.get("role", "unknown"), "content": ""}
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for part in raw.get("parts") or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("text", "reasoning") or (kind is None and "content" in part):
            texts.append(_text_of(part.get("content")))
        elif kind == "tool_call":
            tool_calls.append(
                {"id": part.get("id"), "name": part.get("name"), "arguments": maybe_json(part.get("arguments"))}
            )
        elif kind == "tool_call_response":
            message["tool_call_id"] = part.get("id")
            response = part.get("response", part.get("result"))
            texts.append(response if isinstance(response, str) else json.dumps(response, ensure_ascii=False))
    if "content" in raw and not raw.get("parts"):
        texts.append(_text_of(raw.get("content")))
    message["content"] = "".join(texts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    if raw.get("finish_reason"):
        message["finish_reason"] = raw["finish_reason"]
    if raw.get("tool_call_id"):
        message["tool_call_id"] = raw["tool_call_id"]
    return message


def flat_message(raw: dict[str, Any], role: str | None = None) -> Message:
    """An OpenAI- or Ollama-style message (`role`, `content`, `tool_calls`) into BlackBox's flat form."""
    message: Message = {"role": role or raw.get("role", "unknown"), "content": _text_of(raw.get("content"))}
    calls = []
    for call in raw.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        calls.append(
            {
                "id": call.get("id"),
                "name": function.get("name", call.get("name")),
                "arguments": maybe_json(function.get("arguments", call.get("arguments"))),
            }
        )
    if calls:
        message["tool_calls"] = calls
    if raw.get("tool_call_id") or (raw.get("id") and message["role"] == "tool"):
        message["tool_call_id"] = raw.get("tool_call_id") or raw.get("id")
    if raw.get("tool_name"):
        message["tool_name"] = raw["tool_name"]
    return message


def _messages(value: Any) -> list[Message]:
    parsed = maybe_json(value)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if isinstance(item, dict):
            out.append(message_from_parts(item) if "parts" in item else flat_message(item))
    return out


def _read_current(span: SpanData, view: SpanView) -> None:
    a = span.attributes
    if not any(key.startswith("gen_ai.") for key in a):
        return
    view.dialects.append("current")
    view.operation = a.get("gen_ai.operation.name", view.operation)
    view.provider = a.get("gen_ai.provider.name") or a.get("gen_ai.system") or view.provider
    view.request_model = a.get("gen_ai.request.model", view.request_model)
    view.response_model = a.get("gen_ai.response.model", view.response_model)
    view.input_tokens = _int(a.get("gen_ai.usage.input_tokens", a.get("gen_ai.usage.prompt_tokens")))
    view.output_tokens = _int(a.get("gen_ai.usage.output_tokens", a.get("gen_ai.usage.completion_tokens")))
    if "gen_ai.input.messages" in a:
        view.input_messages = _messages(a["gen_ai.input.messages"])
    if "gen_ai.output.messages" in a:
        view.output_messages = _messages(a["gen_ai.output.messages"])
    instructions = maybe_json(a.get("gen_ai.system_instructions"))
    if instructions is not None:
        view.system_instructions = (
            "".join(_text_of(p.get("content")) for p in instructions if isinstance(p, dict))
            if isinstance(instructions, list)
            else _text_of(instructions)
        )
    definitions = maybe_json(a.get("gen_ai.tool.definitions"))
    if isinstance(definitions, list):
        view.tool_definitions = definitions
    reasons = a.get("gen_ai.response.finish_reasons")
    if isinstance(reasons, list):
        view.finish_reasons = [str(r) for r in reasons]
    elif isinstance(reasons, str):
        view.finish_reasons = [reasons]
    view.tool_name = a.get("gen_ai.tool.name", view.tool_name)
    view.tool_call_id = a.get("gen_ai.tool.call.id", view.tool_call_id)
    if "gen_ai.tool.call.arguments" in a:
        view.tool_arguments = maybe_json(a["gen_ai.tool.call.arguments"])
    if "gen_ai.tool.call.result" in a:
        view.tool_result = maybe_json(a["gen_ai.tool.call.result"])
    view.agent_name = a.get("gen_ai.agent.name", view.agent_name)
    if view.operation is None:
        name = span.name.split(" ", 1)[0]
        if name in LLM_OPERATIONS | TOOL_OPERATIONS | EMBEDDING_OPERATIONS | {"invoke_agent", "create_agent"}:
            view.operation = name


_EVENT_ROLES = {
    "gen_ai.system.message": "system",
    "gen_ai.user.message": "user",
    "gen_ai.assistant.message": "assistant",
    "gen_ai.tool.message": "tool",
}


def _event_body(event: dict[str, Any]) -> dict[str, Any]:
    attributes = dict(event.get("attributes") or {})
    content = maybe_json(attributes.pop("gen_ai.event.content", None))
    if isinstance(content, dict):
        attributes.update(content)
    return attributes


def _read_events(span: SpanData, view: SpanView) -> None:
    inputs: list[Message] = []
    outputs: list[Message] = []
    for event in span.events:
        name = event.get("name", "")
        if name in _EVENT_ROLES:
            body = _event_body(event)
            message = flat_message(body, role=body.get("role") or _EVENT_ROLES[name])
            if name == "gen_ai.tool.message" and body.get("id") and "tool_call_id" not in message:
                message["tool_call_id"] = body["id"]
            inputs.append(message)
        elif name == "gen_ai.choice":
            body = _event_body(event)
            inner = maybe_json(body.get("message")) or {}
            message = flat_message(inner if isinstance(inner, dict) else {"content": inner}, role="assistant")
            if body.get("finish_reason"):
                message["finish_reason"] = body["finish_reason"]
            outputs.append(message)
        elif name == "gen_ai.content.prompt":
            inputs.extend(_messages((event.get("attributes") or {}).get("gen_ai.prompt")))
        elif name == "gen_ai.content.completion":
            outputs.extend(_messages((event.get("attributes") or {}).get("gen_ai.completion")))
    if inputs or outputs:
        view.dialects.append("events")
        if inputs and not view.input_messages:
            view.input_messages = inputs
        if outputs and not view.output_messages:
            view.output_messages = outputs
    if view.system_instructions is None and view.input_messages:
        systems = [m["content"] for m in view.input_messages if m.get("role") == "system"]
        if systems:
            view.system_instructions = "\n".join(systems)


def _read_meai(span: SpanData, view: SpanView) -> None:
    if span.scope and "Microsoft.Extensions.AI" in span.scope:
        view.dialects.append("meai")
        if view.provider is None:
            view.provider = span.attributes.get("gen_ai.system")


def _read_langfuse(span: SpanData, view: SpanView) -> None:
    a = span.attributes
    if not any(key.startswith("langfuse.") for key in a):
        return
    view.dialects.append("langfuse")
    view.observation_input = maybe_json(a.get("langfuse.observation.input"))
    view.observation_output = maybe_json(a.get("langfuse.observation.output"))
    view.level = a.get("langfuse.observation.level")
    view.level_message = a.get("langfuse.observation.status_message")
    view.trace_input = maybe_json(a.get("langfuse.trace.input"))
    view.trace_output = maybe_json(a.get("langfuse.trace.output"))
    for key, value in a.items():
        for prefix in ("langfuse.trace.metadata.", "langfuse.observation.metadata.", "langfuse.metadata."):
            if key.startswith(prefix):
                view.metadata[key.removeprefix(prefix)] = maybe_json(value)
    for key in ("langfuse.trace.metadata", "langfuse.observation.metadata"):
        blob = maybe_json(a.get(key))
        if isinstance(blob, dict):
            view.metadata.update(blob)


def _read_http(span: SpanData, view: SpanView) -> None:
    a = span.attributes
    method = a.get("http.request.method", a.get("http.method"))
    if method is None:
        return
    view.dialects.append("http")
    view.http_method = str(method)
    view.http_url = a.get("url.full", a.get("http.url"))
    view.http_status = _int(a.get("http.response.status_code", a.get("http.status_code")))
    view.server_address = a.get("server.address", a.get("net.peer.name"))
    view.server_port = _int(a.get("server.port", a.get("net.peer.port")))


ADAPTERS = (_read_current, _read_events, _read_meai, _read_langfuse, _read_http)


def view_of(span: SpanData) -> SpanView:
    view = SpanView()
    for adapter in ADAPTERS:
        adapter(span, view)
    return view
