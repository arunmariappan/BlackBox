"""BlackBox's Python SDK for agents: the same run structure as any instrumented agent, without hand-written
OpenTelemetry code.

```python
from blackbox import sdk

sdk.init(service_name="opsdesk-agent", endpoint="http://127.0.0.1:8200")

with sdk.agent_run("opsdesk", input=request, headers=incoming_headers) as run:
    response = sdk.traced_chat(ollama_client, model="qwen3.5:4b", messages=messages, tools=tools)
    result = get_logs(service="checkout-api")        # an @sdk.tool function
    run.set_output(final_answer)
```

This module is the only part of BlackBox an agent may import. It depends on OpenTelemetry and the standard
library, never on BlackBox's internals.
"""

import contextlib
import functools
import inspect
import json
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, overload

from opentelemetry import baggage, context, propagate, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from blackbox.sdk import determinism

__all__ = [
    "RunHandle",
    "agent_run",
    "current_session",
    "current_trace_id",
    "flush",
    "init",
    "now",
    "random",
    "shutdown",
    "tool",
    "tool_span",
    "traced_chat",
    "uuid4",
]

SESSION_BAGGAGE_KEY = "blackbox.session"
MAX_ATTRIBUTE_CHARS = 200_000

now = determinism.now
uuid4 = determinism.uuid4
random = determinism.random


@dataclass
class _State:
    provider: TracerProvider | None = None
    tracer: trace.Tracer = field(default_factory=lambda: trace.NoOpTracer())
    endpoint: str = "http://127.0.0.1:8200"
    instrumented: bool = False


_state = _State()
_lock = threading.Lock()


def init(
    service_name: str,
    endpoint: str = "http://127.0.0.1:8200",
    *,
    instrument_httpx: bool = True,
    batch_delay_ms: int = 500,
    exporter: SpanExporter | None = None,
) -> TracerProvider:
    """Export spans to BlackBox at `endpoint` (OTLP/HTTP protobuf, 500 ms batches) and propagate `traceparent` on
    every httpx request, so the proxy can join each model and tool call to the span that made it."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    with _lock:
        if _state.provider is not None:
            _state.provider.shutdown()
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        span_exporter = exporter or OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(span_exporter, schedule_delay_millis=batch_delay_ms))
        _state.provider = provider
        _state.tracer = provider.get_tracer("blackbox.sdk")
        _state.endpoint = endpoint.rstrip("/")
        determinism.configure(endpoint=_state.endpoint)
        if instrument_httpx:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            instrumentor = HTTPXClientInstrumentor()
            if _state.instrumented:
                instrumentor.uninstrument()
            instrumentor.instrument(tracer_provider=provider)
            _state.instrumented = True
    return provider


def flush(timeout_ms: int = 5000) -> bool:
    """Export every finished span now (blocking)."""
    return _state.provider.force_flush(timeout_ms) if _state.provider is not None else True


def shutdown() -> None:
    with _lock:
        if _state.provider is not None:
            _state.provider.shutdown()
            _state.provider = None
            _state.tracer = trace.NoOpTracer()
        if _state.instrumented:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            HTTPXClientInstrumentor().uninstrument()
            _state.instrumented = False


def _json(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=_default)
    return text if len(text) <= MAX_ATTRIBUTE_CHARS else text[:MAX_ATTRIBUTE_CHARS] + "…[truncated]"


def _default(value: Any) -> Any:
    for attr in ("model_dump", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            return method()
    return str(value)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict or an object (the ollama client returns pydantic models that allow both)."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def current_trace_id() -> str | None:
    span_context = trace.get_current_span().get_span_context()
    return format(span_context.trace_id, "032x") if span_context.is_valid else None


def current_session() -> str | None:
    """The replay session this run belongs to (W3C baggage `blackbox.session`), if any."""
    value = baggage.get_baggage(SESSION_BAGGAGE_KEY)
    return str(value) if value else None


# Runs --------------------------------------------------------------------------------------------------------------


class RunHandle:
    def __init__(self, span: Span) -> None:
        self.span = span
        self.output: Any = None
        self.ending: str | None = None

    @property
    def trace_id(self) -> str:
        return format(self.span.get_span_context().trace_id, "032x")

    def set_output(self, output: Any) -> None:
        self.output = output
        self.span.set_attribute("blackbox.run.output", _json(output))

    def set_ending(self, ending: str) -> None:
        self.ending = ending
        self.span.set_attribute("blackbox.run.ending", ending)

    def set_attribute(self, key: str, value: Any) -> None:
        self.span.set_attribute(key, value if isinstance(value, str | bool | int | float) else _json(value))


@contextlib.contextmanager
def agent_run(
    name: str,
    *,
    input: Any = None,
    headers: Mapping[str, str] | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[RunHandle]:
    """The root span of one agent run, `invoke_agent {name}`. It continues an incoming `traceparent` (and picks up
    a replay session from `baggage`) when `headers` carry one."""
    token = None
    if headers:
        token = context.attach(propagate.extract(dict(headers)))
    try:
        with _state.tracer.start_as_current_span(f"invoke_agent {name}", kind=SpanKind.INTERNAL) as span:
            span.set_attribute("gen_ai.operation.name", "invoke_agent")
            span.set_attribute("gen_ai.agent.name", name)
            if input is not None:
                span.set_attribute("blackbox.run.input", _json(input))
            for key, value in (attributes or {}).items():
                span.set_attribute(key, value if isinstance(value, str | bool | int | float) else _json(value))
            handle = RunHandle(span)
            with determinism.run_scope(trace_id=handle.trace_id, session=current_session(), span=span):
                try:
                    yield handle
                except Exception as exc:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)
                    if handle.ending is None:
                        handle.set_ending("error")
                    raise
    finally:
        if token is not None:
            context.detach(token)


# Model calls --------------------------------------------------------------------------------------------------------


def _parts_message(message: Any) -> dict[str, Any]:
    """An Ollama/OpenAI-style message as a GenAI-conventions message with parts."""
    role = _get(message, "role", "user")
    parts: list[dict[str, Any]] = []
    content = _get(message, "content")
    if role == "tool":
        parts.append(
            {
                "type": "tool_call_response",
                "id": _get(message, "tool_call_id") or _get(message, "tool_name"),
                "response": content,
            }
        )
    elif content:
        parts.append({"type": "text", "content": content})
    for call in _get(message, "tool_calls") or []:
        function = _get(call, "function") or {}
        parts.append(
            {
                "type": "tool_call",
                "id": _get(call, "id"),
                "name": _get(function, "name"),
                "arguments": _get(function, "arguments"),
            }
        )
    return {"role": role, "parts": parts}


def _tool_definitions(tools: Any) -> list[Any]:
    out = []
    for tool_def in tools or []:
        if callable(tool_def) and not isinstance(tool_def, Mapping):
            out.append({"type": "function", "name": getattr(tool_def, "__name__", str(tool_def))})
        else:
            out.append(_default(tool_def) if not isinstance(tool_def, Mapping) else dict(tool_def))
    return out


def traced_chat(
    client: Any,
    *,
    model: str,
    messages: list[Any],
    tools: list[Any] | None = None,
    provider: str = "ollama",
    **kwargs: Any,
) -> Any:
    """Call `client.chat(model=..., messages=..., tools=..., **kwargs)` inside a `chat {model}` span that records the
    request model, input and output messages, tool definitions and token usage (GenAI conventions)."""
    with _state.tracer.start_as_current_span(f"chat {model}", kind=SpanKind.CLIENT) as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", provider)
        span.set_attribute("gen_ai.request.model", model)
        system = [m for m in messages if _get(m, "role") == "system"]
        if system:
            span.set_attribute(
                "gen_ai.system_instructions", _json([{"type": "text", "content": _get(m, "content")} for m in system])
            )
        span.set_attribute(
            "gen_ai.input.messages", _json([_parts_message(m) for m in messages if _get(m, "role") != "system"])
        )
        if tools:
            span.set_attribute("gen_ai.tool.definitions", _json(_tool_definitions(tools)))
        options = kwargs.get("options")
        if isinstance(options, Mapping) and "temperature" in options:
            span.set_attribute("gen_ai.request.temperature", float(options["temperature"]))
        call_kwargs = dict(kwargs)
        if tools is not None:
            call_kwargs["tools"] = tools
        try:
            response = client.chat(model=model, messages=messages, **call_kwargs)
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise
        _record_chat_response(span, response)
        return response


def _record_chat_response(span: Span, response: Any) -> None:
    message = _get(response, "message")
    reason = _get(response, "done_reason") or ("tool_calls" if _get(message, "tool_calls") else "stop")
    if message is not None:
        output = _parts_message(message)
        output["finish_reason"] = reason
        span.set_attribute("gen_ai.output.messages", _json([output]))
    span.set_attribute("gen_ai.response.finish_reasons", [str(reason)])
    if _get(response, "model"):
        span.set_attribute("gen_ai.response.model", str(_get(response, "model")))
    if _get(response, "prompt_eval_count") is not None:
        span.set_attribute("gen_ai.usage.input_tokens", int(_get(response, "prompt_eval_count")))
    if _get(response, "eval_count") is not None:
        span.set_attribute("gen_ai.usage.output_tokens", int(_get(response, "eval_count")))


# Tools --------------------------------------------------------------------------------------------------------------


class ToolSpan:
    def __init__(self, span: Span) -> None:
        self.span = span

    def set_result(self, result: Any) -> None:
        self.span.set_attribute("gen_ai.tool.call.result", _json(result))

    def set_error(self, message: str) -> None:
        self.span.set_status(Status(StatusCode.ERROR, message))
        self.span.set_attribute("error.type", "tool_error")


@contextlib.contextmanager
def tool_span(name: str, arguments: Any = None, *, call_id: str | None = None) -> Iterator[ToolSpan]:
    """An `execute_tool {name}` span; set the result with `.set_result(...)`. Exceptions mark it as an error."""
    with _state.tracer.start_as_current_span(f"execute_tool {name}", kind=SpanKind.INTERNAL) as span:
        span.set_attribute("gen_ai.operation.name", "execute_tool")
        span.set_attribute("gen_ai.tool.name", name)
        span.set_attribute("gen_ai.tool.type", "function")
        if call_id:
            span.set_attribute("gen_ai.tool.call.id", call_id)
        if arguments is not None:
            span.set_attribute("gen_ai.tool.call.arguments", _json(arguments))
        handle = ToolSpan(span)
        try:
            yield handle
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise


@overload
def tool[F: Callable[..., Any]](fn: F, *, name: str | None = None) -> F: ...
@overload
def tool[F: Callable[..., Any]](fn: None = None, *, name: str | None = None) -> Callable[[F], F]: ...


def tool(fn: Any = None, *, name: str | None = None) -> Any:
    """Decorate a tool function so each call is an `execute_tool {name}` span with its arguments and result."""

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        signature = inspect.signature(func)

        def arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
            try:
                bound = signature.bind_partial(*args, **kwargs)
            except TypeError:
                return {"args": list(args), **kwargs}
            return {k: v for k, v in bound.arguments.items() if k not in ("self", "cls")}

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tool_span(tool_name, arguments(args, kwargs)) as span:
                    result = await func(*args, **kwargs)
                    span.set_result(result)
                    return result

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tool_span(tool_name, arguments(args, kwargs)) as span:
                result = func(*args, **kwargs)
                span.set_result(result)
                return result

        return wrapper

    return decorate(fn) if fn is not None else decorate
