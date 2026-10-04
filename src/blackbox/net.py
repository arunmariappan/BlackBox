"""BlackBox's own outgoing HTTP clients.

They use a transport that switches OpenTelemetry's httpx instrumentation off for each request. In a process where
an agent's SDK instrumented httpx globally (tests, an agent embedding BlackBox), the instrumentation would otherwise
create spans for BlackBox's calls and overwrite the `traceparent` header the proxy and the runner send on purpose.
"""

import secrets
from typing import Any

import httpx
from opentelemetry import context as otel_context
from opentelemetry.context import _SUPPRESS_HTTP_INSTRUMENTATION_KEY


class UntracedTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport | None = None, **kwargs: Any) -> None:
        self._inner = inner or httpx.AsyncHTTPTransport(**kwargs)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        token = otel_context.attach(otel_context.set_value(_SUPPRESS_HTTP_INSTRUMENTATION_KEY, True))
        try:
            return await self._inner.handle_async_request(request)
        finally:
            otel_context.detach(token)

    async def aclose(self) -> None:
        await self._inner.aclose()


def make_client(
    *,
    base_url: str = "",
    timeout: float = 60,
    max_connections: int = 100,
    verify: bool = True,
) -> httpx.AsyncClient:
    """An untraced async client that ignores proxy environment variables and adds no default headers."""
    transport = UntracedTransport(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        verify=verify,
    )
    client = httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport, trust_env=False)
    client.headers.clear()
    return client


def new_trace_id() -> str:
    return secrets.token_hex(16)


def new_span_id() -> str:
    return secrets.token_hex(8)


def traceparent(trace_id: str, span_id: str) -> str:
    return f"00-{trace_id}-{span_id}-01"


def parse_traceparent(value: str | None) -> tuple[str, str] | None:
    """`(trace_id, parent_span_id)` from a W3C `traceparent` header, or None if it isn't valid."""
    if not value:
        return None
    parts = value.strip().split("-")
    if len(parts) < 4 or len(parts[1]) != 32 or len(parts[2]) != 16:
        return None
    trace_id, span_id = parts[1].lower(), parts[2].lower()
    try:
        int(trace_id, 16)
        int(span_id, 16)
    except ValueError:
        return None
    if trace_id == "0" * 32 or span_id == "0" * 16:
        return None
    return trace_id, span_id
