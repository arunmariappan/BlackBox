"""Parse recorded exchanges into the step views the UI, metrics and judges read.

- Ollama `/api/chat` and `/api/generate`, OpenAI `/v1/chat/completions` → `llm`: model, messages, tools, format,
  options; the reply message, tool calls, token counts and Ollama's timings; streamed chunks joined into one reply.
- Ollama `/api/embed`, Jina `/v1/embeddings` → `embedding`: model, input texts, dimensions (vectors not shown).
- OpenSearch `/*/_search` → `tool`: the query, then hits with paper id, title, score and chunk text.
- Anything else → `other`: method, path, request and response JSON. Profiles can supply better views.
"""

import gzip
import json
import re
import zlib
from dataclasses import dataclass, field
from typing import Any

from blackbox.otlp.genai import flat_message, maybe_json


@dataclass
class ExchangeView:
    kind: str  # llm, tool, embedding, other
    view: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    tool_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


def decode_body(body: bytes | None, headers: dict[str, Any] | None = None) -> bytes:
    if not body:
        return b""
    encoding = str((headers or {}).get("content-encoding", "")).lower()
    try:
        if encoding == "gzip":
            return gzip.decompress(body)
        if encoding == "deflate":
            return zlib.decompress(body)
        if encoding == "zstd":
            from compression import zstd

            return zstd.decompress(body)
        if encoding == "br":
            return body  # brotli isn't in the standard library; shown raw
    except Exception:
        return body
    return body


def _json(body: bytes) -> Any:
    try:
        return json.loads(body) if body else None
    except ValueError:
        return None


def _lines(body: bytes) -> list[Any]:
    out = []
    for line in body.splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _sse_events(body: bytes) -> list[Any]:
    out = []
    for line in body.decode("utf-8", errors="replace").splitlines():
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                try:
                    out.append(json.loads(data))
                except ValueError:
                    continue
    return out


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) else None


def _ollama_timings(final: dict[str, Any]) -> dict[str, Any]:
    keys = ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")
    return {
        k.removesuffix("_duration") + "_ms": round(final[k] / 1e6, 1) for k in keys if isinstance(final.get(k), int)
    }


def ollama_chat(request: Any, response: bytes, generate: bool = False) -> ExchangeView:
    req = request if isinstance(request, dict) else {}
    view: dict[str, Any] = {"source": "exchange", "model": req.get("model")}
    if generate:
        messages = []
        if req.get("system"):
            view["system"] = req["system"]
        if req.get("prompt") is not None:
            messages.append({"role": "user", "content": str(req.get("prompt"))})
        view["messages"] = messages
    else:
        raw_messages = [m for m in req.get("messages") or [] if isinstance(m, dict)]
        systems = [flat_message(m)["content"] for m in raw_messages if m.get("role") == "system"]
        if systems:
            view["system"] = "\n".join(systems)
        view["messages"] = [flat_message(m) for m in raw_messages if m.get("role") != "system"]
    for key in ("tools", "format", "options", "think", "stream", "keep_alive"):
        if key in req:
            view[key] = req[key]
    chunks = _lines(response)
    if not chunks:
        parsed = _json(response)
        chunks = [parsed] if isinstance(parsed, dict) else []
    content: list[str] = []
    thinking: list[str] = []
    tool_calls: list[Any] = []
    final: dict[str, Any] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if generate:
            content.append(str(chunk.get("response") or ""))
        else:
            message = chunk.get("message") or {}
            content.append(str(message.get("content") or ""))
            thinking.append(str(message.get("thinking") or ""))
            tool_calls.extend(message.get("tool_calls") or [])
        if chunk.get("done"):
            final = chunk
        if chunk.get("error"):
            final = chunk
    reply = flat_message({"role": "assistant", "content": "".join(content), "tool_calls": tool_calls})
    if any(thinking):
        reply["thinking"] = "".join(thinking)
    if final.get("done_reason"):
        reply["finish_reason"] = final["done_reason"]
    view["reply"] = [reply]
    view["timings"] = _ollama_timings(final)
    if len(chunks) > 1:
        view["chunks"] = len(chunks)
    return ExchangeView(
        kind="llm",
        view=view,
        model=str(final.get("model") or req.get("model") or "") or None,
        input_tokens=_int(final.get("prompt_eval_count")),
        output_tokens=_int(final.get("eval_count")),
        error=str(final["error"]) if final.get("error") else None,
    )


def openai_chat(request: Any, response: bytes) -> ExchangeView:
    req = request if isinstance(request, dict) else {}
    raw_messages = [m for m in req.get("messages") or [] if isinstance(m, dict)]
    view: dict[str, Any] = {"source": "exchange", "model": req.get("model")}
    systems = [flat_message(m)["content"] for m in raw_messages if m.get("role") in ("system", "developer")]
    if systems:
        view["system"] = "\n".join(systems)
    view["messages"] = [flat_message(m) for m in raw_messages if m.get("role") not in ("system", "developer")]
    for key in ("tools", "response_format", "temperature", "stream"):
        if key in req:
            view[key] = req[key]
    usage: dict[str, Any] = {}
    model = req.get("model")
    events = _sse_events(response)
    if events:
        content: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish = None
        for event in events:
            model = event.get("model", model)
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                content.append(str(delta.get("content") or ""))
                for call in delta.get("tool_calls") or []:
                    slot = calls.setdefault(int(call.get("index", 0)), {"id": None, "name": "", "arguments": ""})
                    slot["id"] = call.get("id") or slot["id"]
                    function = call.get("function") or {}
                    slot["name"] += function.get("name") or ""
                    slot["arguments"] += function.get("arguments") or ""
                finish = choice.get("finish_reason") or finish
        reply: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
        if calls:
            reply["tool_calls"] = [
                {"id": c["id"], "name": c["name"], "arguments": maybe_json(c["arguments"])}
                for _, c in sorted(calls.items())
            ]
        if finish:
            reply["finish_reason"] = finish
        view["reply"] = [reply]
        view["chunks"] = len(events)
    else:
        parsed = _json(response) or {}
        model = parsed.get("model", model)
        usage = parsed.get("usage") or {}
        replies = []
        for choice in parsed.get("choices") or []:
            message = flat_message(choice.get("message") or {}, role="assistant")
            if choice.get("finish_reason"):
                message["finish_reason"] = choice["finish_reason"]
            replies.append(message)
        view["reply"] = replies
    return ExchangeView(
        kind="llm",
        view=view,
        model=str(model) if model else None,
        input_tokens=_int(usage.get("prompt_tokens")),
        output_tokens=_int(usage.get("completion_tokens")),
    )


def embeddings(request: Any, response: bytes) -> ExchangeView:
    req = request if isinstance(request, dict) else {}
    raw_input = req.get("input", req.get("prompt"))
    inputs = [raw_input] if isinstance(raw_input, str) else [str(x) for x in raw_input or []]
    parsed = _json(response) or {}
    vectors = parsed.get("embeddings") or [d.get("embedding") for d in parsed.get("data") or [] if isinstance(d, dict)]
    if not vectors and isinstance(parsed.get("embedding"), list):
        vectors = [parsed["embedding"]]
    dims = len(vectors[0]) if vectors and isinstance(vectors[0], list) else None
    usage = parsed.get("usage") or {}
    view = {
        "source": "exchange",
        "model": req.get("model"),
        "inputs": inputs,
        "count": len(vectors),
        "dimensions": dims,
    }
    for key in ("task", "dimensions"):
        if key in req and key != "dimensions":
            view[key] = req[key]
    return ExchangeView(
        kind="embedding",
        view=view,
        model=str(req.get("model")) if req.get("model") else None,
        input_tokens=_int(usage.get("total_tokens", usage.get("prompt_tokens", parsed.get("prompt_eval_count")))),
    )


_ID_KEYS = ("arxiv_id", "paper_id", "id", "doc_id")
_TITLE_KEYS = ("title", "paper_title")
_TEXT_KEYS = ("chunk_text", "text", "content", "abstract")


def opensearch_search(path: str, request: Any, response: bytes) -> ExchangeView:
    parsed = _json(response) or {}
    hits_out = []
    for hit in (parsed.get("hits") or {}).get("hits") or []:
        source = hit.get("_source") or {}
        hits_out.append(
            {
                "paper_id": next((source[k] for k in _ID_KEYS if source.get(k)), hit.get("_id")),
                "title": next((source[k] for k in _TITLE_KEYS if source.get(k)), None),
                "score": hit.get("_score"),
                "text": next((source[k] for k in _TEXT_KEYS if source.get(k)), None),
                "id": hit.get("_id"),
            }
        )
    index = path.strip("/").split("/")[0]
    view = {"source": "exchange", "index": index, "query": request, "hits": hits_out, "took_ms": parsed.get("took")}
    error = None
    if parsed.get("error"):
        error = json.dumps(parsed["error"])[:500]
    return ExchangeView(kind="tool", view=view, tool_name="search", error=error)


def generic(method: str, path: str, request: Any, response: bytes) -> ExchangeView:
    parsed = _json(response)
    view = {
        "source": "exchange",
        "tool": {
            "name": f"{method} {path}",
            "arguments": request,
            "result": parsed if parsed is not None else response.decode("utf-8", errors="replace")[:20000],
        },
    }
    return ExchangeView(kind="other", view=view)


_SEARCH = re.compile(r"^/[^/]+/_search$")


def view_exchange(
    upstream: str,
    method: str,
    path: str,
    request_body: bytes | None,
    response_body: bytes | None,
    response_headers: dict[str, Any] | None = None,
) -> ExchangeView:
    request = maybe_json((request_body or b"").decode("utf-8", errors="replace")) if request_body else None
    response = decode_body(response_body, response_headers)
    if path in ("/api/chat",):
        return ollama_chat(request, response)
    if path == "/api/generate":
        return ollama_chat(request, response, generate=True)
    if path.endswith("/chat/completions"):
        return openai_chat(request, response)
    if path in ("/api/embed", "/api/embeddings") or path.endswith("/embeddings"):
        return embeddings(request, response)
    if _SEARCH.match(path):
        return opensearch_search(path, request, response)
    return generic(method, path, request, response)
