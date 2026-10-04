"""Small builders for rows used across tests."""

import secrets
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.store import PreparedBlob, Store, insert_blob
from blackbox.store.models import Exchange, Run, Score, Span, Step
from blackbox.util import new_id, now_ms


def trace_id() -> str:
    return secrets.token_hex(16)


def span_id() -> str:
    return secrets.token_hex(8)


async def add_run(store: Store, **fields: Any) -> Run:
    values: dict[str, Any] = {
        "id": new_id(),
        "trace_id": trace_id(),
        "profile": "test",
        "status": "complete",
        "started_ms": now_ms(),
        "updated_ms": now_ms(),
    }
    values.update(fields)
    run = Run(**values)

    async def op(session: AsyncSession) -> Run:
        session.add(run)
        return run

    return await store.write(op)


async def add_full_run(store: Store, *, started_ms: int | None = None) -> Run:
    """A run with spans, two exchanges with blobs, steps and a score."""
    started = started_ms if started_ms is not None else now_ms()
    run = await add_run(store, started_ms=started, updated_ms=started)
    request = PreparedBlob.of(b'{"model":"m","messages":[{"role":"user","content":"hi ' + run.id.encode() + b'"}]}')
    response = PreparedBlob.of(b'{"message":{"role":"assistant","content":"hello"}}', "application/json")
    shared = PreparedBlob.of(b"a system prompt every run shares", "text/plain")
    root = span_id()

    async def op(session: AsyncSession) -> None:
        for blob in (request, response, shared):
            await insert_blob(session, blob)
        session.add(
            Span(
                trace_id=run.trace_id,
                span_id=root,
                parent_span_id=None,
                name="invoke_agent test",
                kind="internal",
                service="svc",
                start_ns=started * 1_000_000,
                end_ns=(started + 50) * 1_000_000,
                attributes={"gen_ai.operation.name": "invoke_agent", "big": {"$blob": shared.sha256}},
                events=[{"name": "e", "time_ns": started * 1_000_000, "attributes": {"k": 1}}],
                resource={"service.name": "svc"},
                blob_refs=[shared.sha256],
                received_ms=started,
            )
        )
        for seq in (1, 2):
            exchange_id = new_id()
            session.add(
                Exchange(
                    id=exchange_id,
                    trace_id=run.trace_id,
                    parent_span_id=root,
                    upstream="ollama",
                    seq=seq,
                    method="POST",
                    path="/api/chat",
                    request_headers={"content-type": "application/json"},
                    request_blob=request.sha256,
                    request_key="k" * 64,
                    status=200,
                    response_headers={"content-type": "application/json"},
                    response_blob=response.sha256,
                    started_ms=started + seq,
                    first_byte_ms=started + seq + 1,
                    ended_ms=started + seq + 2,
                )
            )
            session.add(
                Step(run_id=run.id, idx=seq, kind="llm", node="plan", exchange_id=exchange_id, view={"seq": seq})
            )
        session.add(
            Score(
                id=new_id(),
                run_id=run.id,
                kind="metric",
                name="steps",
                version="1",
                value=2.0,
                details={"a": [1, 2]},
                created_ms=started,
            )
        )

    await store.write(op)
    return run
