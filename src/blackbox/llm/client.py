"""BlackBox's own model calls (judges, failure descriptions, cluster names): `qwen3.5:4b` in the host Ollama with
`think` off, output constrained by a JSON schema in Ollama's `format`, validated with Pydantic, retried once, then
an explicit `invalid` result. Nothing silently falls back to a default answer."""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from blackbox.config import OllamaConfig
from blackbox.net import make_client

log = logging.getLogger(__name__)


@dataclass
class LLMResult[T: BaseModel]:
    valid: bool
    parsed: T | None
    raw: str  # the last reply's content
    request: dict[str, Any]  # the last request body sent
    attempts: int
    latency_ms: int
    error: str | None = None
    replies: list[str] = field(default_factory=list)


class LLMUnavailable(Exception):
    """Ollama couldn't be reached or answered with an HTTP error; nothing about the model's output is known."""


class OllamaJSON:
    def __init__(self, config: OllamaConfig, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self._client = client or make_client(base_url=config.base_url, timeout=config.timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def call[T: BaseModel](
        self,
        messages: list[dict[str, Any]],
        schema: type[T],
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        retries: int = 1,
    ) -> LLMResult[T]:
        body: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "format": schema.model_json_schema(),
            "stream": False,
            "think": False,
            "options": {"temperature": 0, **(options or {})},
        }
        started = time.perf_counter()
        replies: list[str] = []
        error: str | None = None
        for attempt in range(1, retries + 2):
            try:
                response = await self._client.post("/api/chat", json=body)
            except httpx.HTTPError as exc:
                raise LLMUnavailable(f"Ollama at {self.config.base_url}: {type(exc).__name__}: {exc}") from exc
            if response.status_code >= 400:
                raise LLMUnavailable(f"Ollama answered {response.status_code}: {response.text[:300]}")
            try:
                content = str(response.json().get("message", {}).get("content", ""))
            except ValueError:
                content = response.text
            replies.append(content)
            try:
                parsed = schema.model_validate_json(content)
            except (ValidationError, ValueError) as exc:
                error = _short(exc)
                log.info("model reply failed %s validation (attempt %d): %s", schema.__name__, attempt, error)
                continue
            return LLMResult(True, parsed, content, body, attempt, _ms(started), None, replies)
        return LLMResult(False, None, replies[-1] if replies else "", body, len(replies), _ms(started), error, replies)


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _short(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5])
    return str(exc)[:300]


def schema_text(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema(), sort_keys=True)
