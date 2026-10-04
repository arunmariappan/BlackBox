"""Fake upstreams for tests: a model server that counts its calls, streams on request, and can be scripted."""

import asyncio
import gzip
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from blackbox.server import QuietServer, start_server


@dataclass
class FakeUpstream:
    """A fake Ollama/OpenAI-style server. `reply` decides each chat answer; every call is kept in `calls`."""

    reply: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    chunks: int = 50
    chunk_delay: float = 0.0
    server: QuietServer | None = None
    task: asyncio.Task[None] | None = None
    port: int = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def chat_calls(self) -> list[dict[str, Any]]:
        return [c["body"] for c in self.calls if c["path"] in ("/api/chat", "/v1/chat/completions")]

    def app(self) -> Starlette:
        async def record(request: Request) -> dict[str, Any]:
            raw = await request.body()
            try:
                body = json.loads(raw) if raw else None
            except ValueError:
                body = raw.decode()
            entry = {"path": request.url.path, "headers": dict(request.headers), "body": body}
            self.calls.append(entry)
            return entry

        async def chat(request: Request) -> Response:
            entry = await record(request)
            body = entry["body"] or {}
            if self.reply is not None:
                message = self.reply(body)
            else:
                message = {"role": "assistant", "content": f"echo: {body['messages'][-1]['content']}"}
            final = {
                "model": body.get("model"),
                "created_at": "2026-10-04T00:00:00Z",
                "message": message,
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 10 * len(body.get("messages", [])),
                "eval_count": 5,
            }
            if body.get("stream"):

                async def stream() -> AsyncIterator[bytes]:
                    content = str(message.get("content", ""))
                    for i in range(self.chunks - 1):
                        piece = content[i :: self.chunks - 1] if content else f"t{i}"
                        yield (
                            json.dumps(
                                {
                                    "model": body.get("model"),
                                    "message": {"role": "assistant", "content": piece},
                                    "done": False,
                                }
                            )
                            + "\n"
                        ).encode()
                        if i == 0:
                            await self.release.wait()
                        await asyncio.sleep(self.chunk_delay)
                    yield (json.dumps({**final, "message": {"role": "assistant", "content": ""}}) + "\n").encode()

                return StreamingResponse(stream(), media_type="application/x-ndjson")
            return JSONResponse(final)

        async def openai(request: Request) -> Response:
            entry = await record(request)
            body = entry["body"] or {}

            async def stream() -> AsyncIterator[bytes]:
                def event(data: dict[str, Any]) -> bytes:
                    return f"data: {json.dumps(data)}\n\n".encode()

                for word in ["Hello", " there", "!"]:
                    yield event({"model": body.get("model"), "choices": [{"index": 0, "delta": {"content": word}}]})
                usage = {"prompt_tokens": 7, "completion_tokens": 3}
                yield event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage})
                yield b"data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

        async def tags(request: Request) -> Response:
            await record(request)
            return JSONResponse({"models": [{"name": "qwen3.5:4b"}, {"name": "qwen3.5:9b"}]})

        async def compressed(request: Request) -> Response:
            await record(request)
            data = gzip.compress(json.dumps({"embeddings": [[0.1, 0.2, 0.3]]}).encode())
            return Response(data, headers={"content-encoding": "gzip", "content-type": "application/json"})

        async def slow(request: Request) -> Response:
            await record(request)

            async def stream() -> AsyncIterator[bytes]:
                yield b'{"part": 1}\n'
                await asyncio.sleep(30)
                yield b'{"part": 2}\n'

            return StreamingResponse(stream(), media_type="application/x-ndjson")

        async def headers(request: Request) -> Response:
            entry = await record(request)
            return JSONResponse({"seen": entry["headers"]}, headers={"set-cookie": "session=abc"})

        return Starlette(
            routes=[
                Route("/api/chat", chat, methods=["POST"]),
                Route("/v1/chat/completions", openai, methods=["POST"]),
                Route("/api/tags", tags),
                Route("/api/embed", compressed, methods=["POST"]),
                Route("/slow", slow, methods=["POST"]),
                Route("/headers", headers, methods=["POST"]),
            ]
        )

    async def start(self) -> FakeUpstream:
        self.server, self.task, self.port = await start_server(self.app(), "127.0.0.1", 0, "fake-upstream")
        return self

    async def stop(self) -> None:
        if self.server is not None and self.task is not None:
            self.server.should_exit = True
            await self.task
