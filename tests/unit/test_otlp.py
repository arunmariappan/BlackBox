import gzip
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from blackbox.otlp.decode import SpanData, decode_json, decode_protobuf, encode_request
from blackbox.otlp.receiver import otlp_router

FIXTURES = Path(__file__).parent.parent / "fixtures" / "otlp"


def load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_protobuf_and_json_give_identical_spans() -> None:
    from_json = decode_json(load("paperpilot-agentic.json"))
    from_protobuf = decode_protobuf(encode_request(from_json).SerializeToString())
    assert from_json == from_protobuf
    assert len(from_json) == 23
    first = from_json[0]
    assert first.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert first.parent_span_id == "00f067aa0ba902b7"
    assert first.kind == "server"
    assert first.service == "api"
    assert first.attributes["http.response.status_code"] == 200


def test_events_dialect_decodes_events_and_int_values() -> None:
    spans = decode_json(load("events-dialect.json"))
    chat = spans[1]
    assert [e["name"] for e in chat.events] == ["gen_ai.system.message", "gen_ai.user.message", "gen_ai.choice"]
    assert chat.attributes["gen_ai.usage.prompt_tokens"] == 41
    assert chat.events[2]["attributes"]["index"] == 0


@pytest.fixture
async def receiver() -> AsyncIterator[tuple[httpx.AsyncClient, list[SpanData]]]:
    received: list[SpanData] = []

    async def sink(spans: list[SpanData]) -> None:
        received.extend(spans)

    app = FastAPI()
    app.include_router(otlp_router(sink))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client, received


async def test_receiver_accepts_json_protobuf_and_gzip(receiver: tuple[httpx.AsyncClient, list[SpanData]]) -> None:
    client, received = receiver
    body = load("paperpilot-agentic.json")
    response = await client.post("/v1/traces", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 200 and response.json() == {}
    proto = encode_request(decode_json(body)).SerializeToString()
    response = await client.post(
        "/v1/traces",
        content=gzip.compress(proto),
        headers={"content-type": "application/x-protobuf", "content-encoding": "gzip"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-protobuf"
    assert received[:23] == received[23:]


async def test_receiver_rejects_bad_bodies(receiver: tuple[httpx.AsyncClient, list[SpanData]]) -> None:
    client, received = receiver
    too_big = b"x" * (16 * 1024 * 1024 + 1)
    assert (await client.post("/v1/traces", content=too_big)).status_code == 413
    assert (
        await client.post("/v1/traces", content=b"{", headers={"content-type": "application/json"})
    ).status_code == 400
    assert (
        await client.post("/v1/traces", content=b"\xff\xfe", headers={"content-type": "application/x-protobuf"})
    ).status_code == 400
    assert (await client.post("/v1/traces", content=b"x", headers={"content-type": "text/plain"})).status_code == 415
    bad_id = json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [{"traceId": "zz", "spanId": "00"}]}]}]})
    assert (
        await client.post("/v1/traces", content=bad_id, headers={"content-type": "application/json"})
    ).status_code == 400
    assert received == []
