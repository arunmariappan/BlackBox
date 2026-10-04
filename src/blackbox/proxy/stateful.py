"""Replay of stateful upstreams (the OpsDesk environment).

Answering a stateful tool from the tape isn't enough: in a fork, the live steps after step N need an environment
whose state matches steps before N, and the ids the agent saw on the tape must work against it.

1. **Sync-forwarding.** While a call is served from the tape, a state-changing call (POST, PUT, PATCH, DELETE) is also
   forwarded to the session's own sandbox, so that sandbox follows the run step by step.
2. **Identifier aliasing.** When a sync-forwarded call's live response differs from the tape's at an identifier path
   (`$.id` of a new ticket), the session records an alias, tape id → live id. Once the session is live, tape ids in
   request paths and bodies are rewritten to live ids before forwarding, and live ids in responses back to tape ids,
   so the agent's view stays consistent with what it saw before the fork.
"""

import json
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import httpx
from jsonpath_ng.ext import parse as parse_jsonpath

from blackbox.proxy.core import IncomingRequest, Outcome, Proxy, Recording, json_outcome
from blackbox.proxy.http import response_headers
from blackbox.proxy.views import decode_body

if TYPE_CHECKING:
    from blackbox.proxy.sessions import ReplaySession, TapeEntry

log = logging.getLogger(__name__)

SANDBOX_HEADER = "x-sandbox"
STATE_CHANGING = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def rewrite(data: str, mapping: dict[str, str]) -> str:
    for old in sorted(mapping, key=len, reverse=True):
        data = data.replace(old, mapping[old])
    return data


def rewrite_bytes(data: bytes, mapping: dict[str, str]) -> bytes:
    if not mapping or not data:
        return data
    return rewrite(data.decode("utf-8", errors="surrogateescape"), mapping).encode("utf-8", errors="surrogateescape")


def with_sandbox(req: IncomingRequest, sandbox: str | None) -> IncomingRequest:
    if sandbox is None:
        return req
    headers = [(k, v) for k, v in req.headers if k.lower() != SANDBOX_HEADER]
    headers.append(("X-Sandbox", sandbox))
    return replace(req, headers=headers)


def _json(data: bytes) -> Any:
    try:
        return json.loads(data) if data else None
    except ValueError:
        return None


def capture_aliases(session: ReplaySession, req: IncomingRequest, tape_body: bytes, live_body: bytes) -> None:
    paths = session.profile.identifier_paths(req.method, req.path)
    if not paths:
        return
    tape, live = _json(tape_body), _json(live_body)
    if tape is None or live is None:
        return
    aliases: dict[str, str] = session.state.setdefault("aliases", {})
    for path in paths:
        expression = parse_jsonpath(path)
        tape_values = [m.value for m in expression.find(tape)]
        live_values = [m.value for m in expression.find(live)]
        for tape_value, live_value in zip(tape_values, live_values, strict=False):
            if isinstance(tape_value, str) and isinstance(live_value, str) and tape_value != live_value:
                aliases[tape_value] = live_value


async def _read_all(response: httpx.Response) -> bytes:
    try:
        return await response.aread()
    finally:
        await response.aclose()


async def sync_forward(session: ReplaySession, proxy: Proxy, req: IncomingRequest, entry: TapeEntry) -> None:
    """Replay a state-changing tape call into the session's sandbox (the agent still gets the tape's response)."""
    if req.method.upper() not in STATE_CHANGING:
        return
    sandbox = session.state.get("sandbox")
    if sandbox is None:
        return
    aliases: dict[str, str] = session.state.get("aliases", {}) if session.spec.aliasing else {}
    target = with_sandbox(req, sandbox)
    body = rewrite_bytes(req.body, aliases)
    path = rewrite(req.raw_path, aliases)
    synced: list[dict[str, Any]] = session.state.setdefault("synced", [])
    try:
        response = await proxy.send(target, body, path=path)
        live_body = decode_body(await _read_all(response), dict(response.headers))
    except httpx.HTTPError as exc:
        synced.append({"step": entry.step, "path": req.path, "error": str(exc)})
        log.warning("sync-forward of %s %s failed: %s", req.method, req.path, exc)
        return
    synced.append({"step": entry.step, "path": req.path, "status": response.status_code})
    if session.spec.aliasing:
        tape_body = decode_body(entry.exchange.response_body, entry.exchange.row.response_headers)
        capture_aliases(session, req, tape_body, live_body)


async def forward_live_stateful(
    session: ReplaySession, proxy: Proxy, req: IncomingRequest, rec: Recording, body: bytes
) -> Outcome:
    """A live call on a stateful upstream: to the session's sandbox, with tape ids mapped to live ids on the way
    out and back on the way in."""
    aliases: dict[str, str] = session.state.get("aliases", {}) if session.spec.aliasing else {}
    reverse = {live: tape for tape, live in aliases.items()}
    target = with_sandbox(req, session.state.get("sandbox"))
    sent = rewrite_bytes(body, aliases)
    path = rewrite(req.raw_path, aliases)
    if sent != req.body:
        rec.sent_request_body = sent
    try:
        response = await proxy.send(target, sent, path=path)
        raw = await _read_all(response)
    except httpx.HTTPError as exc:
        rec.status = 502
        rec.finish(f"upstream_error: {exc}")
        await proxy.store_recording(rec)
        return json_outcome(
            502, {"error": "blackbox_upstream_error", "upstream": req.upstream.name, "detail": str(exc)}
        )
    decoded = decode_body(raw, dict(response.headers))
    shown = rewrite_bytes(decoded, reverse)
    headers = [
        (k, v)
        for k, v in response_headers(response.headers.multi_items())
        if k.lower() not in ("content-length", "content-encoding")
    ]
    headers.append(("content-length", str(len(shown))))
    rec.status = response.status_code
    rec.content_type = response.headers.get("content-type")
    rec.response_headers = {k.lower(): v for k, v in headers}
    rec.add_chunk(shown)

    async def one() -> Any:
        yield shown

    async def finish(error: str | None) -> None:
        rec.finish(error)
        await proxy.store_recording(rec)

    return Outcome(response.status_code, headers, one(), finish)
