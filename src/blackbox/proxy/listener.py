"""One listener per upstream: a small raw ASGI app that reads the request, asks the proxy for an outcome, and streams
it back chunk by chunk while watching for the client going away."""

import asyncio
import contextlib
import json
import logging
from typing import Any

from blackbox.config import UpstreamConfig
from blackbox.proxy.core import IncomingRequest, Outcome, Proxy

log = logging.getLogger(__name__)

type Scope = dict[str, Any]
type Message = dict[str, Any]


class ListenerApp:
    def __init__(self, proxy: Proxy, upstream: UpstreamConfig) -> None:
        self.proxy = proxy
        self.upstream = upstream

    async def __call__(self, scope: Scope, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        raw_path = scope.get("raw_path") or scope["path"].encode()
        request = IncomingRequest(
            upstream=self.upstream,
            method=scope["method"],
            path=scope["path"],
            raw_path=raw_path.decode("latin-1"),
            query=scope.get("query_string", b"").decode("latin-1"),
            headers=[(k.decode("latin-1"), v.decode("latin-1")) for k, v in scope.get("headers", [])],
            body=bytes(body),
        )
        try:
            outcome = await self.proxy.handle(request)
        except Exception as exc:
            log.exception("proxy %s failed on %s %s", self.upstream.name, request.method, request.path)
            payload = json.dumps({"error": "blackbox_proxy_error", "upstream": self.upstream.name, "detail": str(exc)})
            await send(
                {"type": "http.response.start", "status": 500, "headers": [(b"content-type", b"application/json")]}
            )
            await send({"type": "http.response.body", "body": payload.encode()})
            return
        await self._stream(outcome, receive, send)

    async def _stream(self, outcome: Outcome, receive: Any, send: Any) -> None:
        async def watch() -> None:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return

        watcher = asyncio.ensure_future(watch())
        iterator = outcome.body.__aiter__()
        error: str | None = None
        try:
            headers = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in outcome.headers]
            await send({"type": "http.response.start", "status": outcome.status, "headers": headers})
            while True:
                next_chunk = asyncio.ensure_future(anext(iterator))
                done, _ = await asyncio.wait({next_chunk, watcher}, return_when=asyncio.FIRST_COMPLETED)
                if next_chunk not in done:
                    next_chunk.cancel()
                    with contextlib.suppress(BaseException):
                        await next_chunk
                    error = "client_aborted"
                    break
                try:
                    chunk = next_chunk.result()
                except StopAsyncIteration:
                    break
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            if error is None:
                await send({"type": "http.response.body", "body": b"", "more_body": False})
        except asyncio.CancelledError:
            error = error or "client_aborted"
            raise
        except Exception as exc:
            error = error or f"upstream_aborted: {type(exc).__name__}: {exc}"
            log.warning("stream on %s ended early: %s", self.upstream.name, error)
        finally:
            watcher.cancel()
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(BaseException):
                    await aclose()
            # Storing the exchange runs as its own task: it survives the request being cancelled, and a kept-alive
            # connection can take the agent's next call without waiting for the write.
            self.proxy.spawn(outcome.finish(error))
