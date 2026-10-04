"""Header and path helpers for the proxy."""

import json
import re
from collections.abc import Iterable, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode

from blackbox.util import canonical_json, sha256_hex

HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "trailers",
        "upgrade",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
    }
)
SECRET_NAME = re.compile(r"(api[-_]?key|token|secret|password|auth)", re.IGNORECASE)
REDACTED = "[redacted]"

type HeaderList = list[tuple[str, str]]


def is_hop_by_hop(name: str) -> bool:
    lower = name.lower()
    return lower in HOP_BY_HOP or lower.startswith("proxy-")


def forward_request_headers(headers: Iterable[tuple[str, str]], target_host: str) -> HeaderList:
    """Headers to send upstream: hop-by-hop headers and the body length dropped, `host` set to the target's.
    Everything else, including `traceparent` and `authorization`, passes unchanged."""
    out: HeaderList = []
    for name, value in headers:
        lower = name.lower()
        if is_hop_by_hop(lower) or lower in ("host", "content-length"):
            continue
        out.append((name, value))
    out.append(("host", target_host))
    return out


def response_headers(headers: Iterable[tuple[str, str]]) -> HeaderList:
    return [(name, value) for name, value in headers if not is_hop_by_hop(name)]


def redact(headers: Iterable[tuple[str, str]], names: Sequence[str]) -> dict[str, Any]:
    """Headers as stored: secrets dropped (configured names, and anything that looks like a key or token).
    Repeated headers become lists."""
    blocked = {name.lower() for name in names}
    out: dict[str, Any] = {}
    for name, value in headers:
        lower = name.lower()
        if lower in blocked or SECRET_NAME.search(lower):
            continue
        if lower in out:
            existing = out[lower]
            out[lower] = [*existing, value] if isinstance(existing, list) else [existing, value]
        else:
            out[lower] = value
    return out


def path_matches(path: str, patterns: Sequence[str]) -> bool:
    """Glob patterns where `*` matches one path segment (`/*/_search`) and `/*` alone matches every path."""
    for pattern in patterns:
        if pattern in ("/*", "*", "/**"):
            return True
        regex = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
        if re.fullmatch(regex, path):
            return True
    return False


def canonical_query(query: str) -> str:
    return urlencode(sorted(parse_qsl(query, keep_blank_values=True)))


def canonical_body(body: bytes | None) -> Any:
    """JSON bodies as parsed values (so key order doesn't matter); anything else as text or a hash."""
    if not body:
        return None
    try:
        return json.loads(body)
    except ValueError:
        try:
            return {"$text": body.decode("utf-8")}
        except UnicodeDecodeError:
            return {"$sha256": sha256_hex(body)}


def request_key(method: str, path: str, query: str, body: bytes | None) -> str:
    """SHA-256 of the canonical request: method, path, sorted query and the JSON body with sorted keys."""
    return sha256_hex(canonical_json([method.upper(), path, canonical_query(query), canonical_body(body)]).encode())


def header_value(headers: Iterable[tuple[str, str]], name: str) -> str | None:
    lower = name.lower()
    for key, value in headers:
        if key.lower() == lower:
            return value
    return None
