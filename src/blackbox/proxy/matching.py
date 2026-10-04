"""Matching replayed requests to the tape.

A request's **match key** is the SHA-256 of its canonical form (method, path, sorted query, JSON body with sorted
keys) after the profile's normalisers for that upstream. Normalisers only affect matching, never what is sent or
served. Without normalisers, exact means exact.
"""

import copy
import difflib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from jsonpath_ng.ext import parse as parse_jsonpath

from blackbox.proxy.http import canonical_body, canonical_query
from blackbox.util import canonical_json, sha256_hex

LLM_PATHS = ("/api/chat", "/api/generate", "/v1/chat/completions")


@dataclass(frozen=True)
class MaskRegex:
    """Replace every match of `pattern` in the canonical request text, e.g. ISO timestamps → `<ts>`."""

    pattern: str
    replacement: str = "<masked>"

    def apply_text(self, text: str) -> str:
        return re.sub(self.pattern, self.replacement, text)


@dataclass(frozen=True)
class DropJsonPath:
    """Remove the values at a JSONPath from the body before matching, e.g. a per-request id field."""

    path: str

    def apply_body(self, body: Any) -> Any:
        if not isinstance(body, dict | list):
            return body
        expression = parse_jsonpath(self.path)
        result = copy.deepcopy(body)
        for match in expression.find(result):
            context = match.context.value if match.context is not None else None
            key = getattr(match.path, "fields", None)
            index = getattr(match.path, "index", None)
            if isinstance(context, dict) and key:
                for field in key:
                    context.pop(field, None)
            elif isinstance(context, list) and index is not None and 0 <= index < len(context):
                context[index] = None
        return result


type Normaliser = MaskRegex | DropJsonPath

ISO_TIMESTAMP = MaskRegex(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?", "<ts>")
UUID = MaskRegex(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", "<uuid>")


def match_key(method: str, path: str, query: str, body: bytes | None, normalisers: Sequence[Normaliser] = ()) -> str:
    parsed = canonical_body(body)
    for normaliser in normalisers:
        if isinstance(normaliser, DropJsonPath):
            parsed = normaliser.apply_body(parsed)
    text = canonical_json([method.upper(), path, canonical_query(query), parsed])
    for normaliser in normalisers:
        if isinstance(normaliser, MaskRegex):
            text = normaliser.apply_text(text)
    return sha256_hex(text.encode())


def is_llm_path(path: str) -> bool:
    return path in LLM_PATHS or path.endswith("/chat/completions")


# Diffs ---------------------------------------------------------------------------------------------------------------


def _messages(body: Any) -> list[dict[str, Any]] | None:
    if isinstance(body, dict) and isinstance(body.get("messages"), list):
        return [m if isinstance(m, dict) else {"content": m} for m in body["messages"]]
    if isinstance(body, dict) and isinstance(body.get("prompt"), str):
        return [{"role": "prompt", "content": body["prompt"]}]
    return None


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def message_diff(tape: Any, actual: Any) -> list[dict[str, Any]] | None:
    """A text diff per chat message that differs (None when the bodies aren't chat requests)."""
    tape_messages, actual_messages = _messages(tape), _messages(actual)
    if tape_messages is None or actual_messages is None:
        return None
    out: list[dict[str, Any]] = []
    for index in range(max(len(tape_messages), len(actual_messages))):
        old = tape_messages[index] if index < len(tape_messages) else None
        new = actual_messages[index] if index < len(actual_messages) else None
        if old == new:
            continue
        old_text = _text((old or {}).get("content", "")) if old is not None else ""
        new_text = _text((new or {}).get("content", "")) if new is not None else ""
        lines = list(
            difflib.unified_diff(old_text.splitlines(), new_text.splitlines(), "tape", "request", lineterm="", n=2)
        )
        out.append(
            {
                "index": index,
                "role": (new or old or {}).get("role"),
                "change": "added" if old is None else "removed" if new is None else "changed",
                "diff": "\n".join(lines) if lines else None,
                "other": None if lines else {"tape": old, "request": new},
            }
        )
    rest_tape = {k: v for k, v in tape.items() if k != "messages"} if isinstance(tape, dict) else tape
    rest_actual = {k: v for k, v in actual.items() if k != "messages"} if isinstance(actual, dict) else actual
    if rest_tape != rest_actual:
        out.append(
            {"index": None, "role": None, "change": "fields", "diff": None, "other": json_diff(rest_tape, rest_actual)}
        )
    return out


def json_diff(old: Any, new: Any, path: str = "$", limit: int = 50) -> list[dict[str, Any]]:
    """The paths where two JSON values differ, with both values."""
    out: list[dict[str, Any]] = []

    def walk(a: Any, b: Any, where: str) -> None:
        if len(out) >= limit or a == b:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key, _MISSING), b.get(key, _MISSING), f"{where}.{key}")
            return
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                walk(x, y, f"{where}[{i}]")
            return
        out.append(
            {
                "path": where,
                "tape": None if a is _MISSING else a,
                "request": None if b is _MISSING else b,
                "missing": "tape" if a is _MISSING else "request" if b is _MISSING else None,
            }
        )

    walk(old, new, path)
    return out


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


def request_diff(tape_body: bytes | None, actual_body: bytes | None) -> dict[str, Any]:
    tape, actual = canonical_body(tape_body), canonical_body(actual_body)
    messages = message_diff(tape, actual)
    if messages is not None:
        return {"kind": "messages", "messages": messages}
    return {"kind": "json", "paths": json_diff(tape, actual)}
