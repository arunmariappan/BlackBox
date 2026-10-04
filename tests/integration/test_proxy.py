"""The recording proxy against a fake upstream, inside a whole in-process BlackBox."""

import asyncio
import gzip
import json
import statistics
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from blackbox.config import UpstreamConfig
from blackbox.net import new_span_id, new_trace_id, traceparent
from blackbox.server import Running
from blackbox.store.models import Exchange
from tests.fakes import FakeUpstream
from tests.harness import running_blackbox


@pytest.fixture
async def upstream() -> AsyncIterator[FakeUpstream]:
    fake = await FakeUpstream().start()
    fake.release.set()
    try:
        yield fake
    finally:
        await fake.stop()


@pytest.fixture
async def bb(tmp_path: Path, migrated_db: Path, upstream: FakeUpstream) -> AsyncIterator[Running]:
    config = UpstreamConfig(
        name="ollama",
        listen_port=0,
        target=upstream.url,
        record_paths=["/api/chat", "/api/embed", "/v1/chat/completions", "/slow", "/headers"],
    )
    async with running_blackbox(tmp_path, migrated_db, upstreams=[config], keep_unmatched=True) as running:
        yield running


def proxy_url(bb: Running) -> str:
    assert bb.services.proxy is not None
    return f"http://127.0.0.1:{bb.services.proxy.ports['ollama']}"


async def exchanges(bb: Running, trace_id: str | None = None, within: float = 5) -> list[Exchange]:
    """Exchanges are written just after the response ends; wait for them."""
    async with asyncio.timeout(within):
        while True:
            if trace_id is None:
                rows = list(await bb.services.store.reader.unattributed_exchanges())
            else:
                rows = list(await bb.services.store.reader.exchanges(trace_id))
            if rows:
                return rows
            await asyncio.sleep(0.02)


def headers_for(trace_id: str, parent: str | None = None) -> dict[str, str]:
    return {"traceparent": traceparent(trace_id, parent or new_span_id())}


async def test_json_call_is_byte_identical_and_recorded(bb: Running, upstream: FakeUpstream) -> None:
    trace_id, parent = new_trace_id(), new_span_id()
    body = {"model": "qwen3.5:4b", "messages": [{"role": "user", "content": "hi"}], "stream": False}
    async with httpx.AsyncClient() as client:
        direct = await client.post(f"{upstream.url}/api/chat", json=body)
        proxied = await client.post(f"{proxy_url(bb)}/api/chat", json=body, headers=headers_for(trace_id, parent))
    assert proxied.status_code == 200
    assert proxied.content == direct.content
    [row] = await exchanges(bb, trace_id)
    assert (row.trace_id, row.parent_span_id, row.seq, row.upstream, row.method, row.path) == (
        trace_id,
        parent,
        1,
        "ollama",
        "POST",
        "/api/chat",
    )
    assert await bb.services.store.blobs.get(row.response_blob or "") == proxied.content
    assert json.loads(await bb.services.store.blobs.get(row.request_blob or "")) == body
    assert row.served_from == "live" and row.stream is False and row.error is None
    assert row.first_byte_ms is not None and row.ended_ms is not None and row.ended_ms >= row.started_ms


async def test_ndjson_stream_arrives_chunk_by_chunk_and_identical(bb: Running, upstream: FakeUpstream) -> None:
    upstream.release.clear()  # the upstream holds chunk 2 until the client has chunk 1
    trace_id = new_trace_id()
    body = {"model": "qwen3.5:4b", "messages": [{"role": "user", "content": "stream please"}], "stream": True}
    received = bytearray()
    async with (
        httpx.AsyncClient(timeout=5) as client,
        client.stream("POST", f"{proxy_url(bb)}/api/chat", json=body, headers=headers_for(trace_id)) as response,
    ):
        async for chunk in response.aiter_raw():
            if not upstream.release.is_set():
                upstream.release.set()  # proves the first chunk got here before the upstream finished
            received += chunk
    lines = bytes(received).splitlines()
    assert len(lines) == 50 and json.loads(lines[-1])["done"] is True
    [row] = await exchanges(bb, trace_id)
    assert await bb.services.store.blobs.get(row.response_blob or "") == bytes(received)
    assert row.stream is True
    assert len(row.chunk_times) > 1
    offsets = [offset for offset, _ in row.chunk_times]
    assert offsets == sorted(offsets) and offsets[0] == 0


async def test_sse_stream_is_identical(bb: Running, upstream: FakeUpstream) -> None:
    trace_id = new_trace_id()
    body = {"model": "gpt", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    async with httpx.AsyncClient() as client:
        direct = await client.post(f"{upstream.url}/v1/chat/completions", json=body)
        proxied = await client.post(f"{proxy_url(bb)}/v1/chat/completions", json=body, headers=headers_for(trace_id))
    assert proxied.content == direct.content
    assert proxied.headers["content-type"].startswith("text/event-stream")
    [row] = await exchanges(bb, trace_id)
    assert row.stream is True
    assert await bb.services.store.blobs.get(row.response_blob or "") == direct.content


async def test_compressed_response_passes_through_raw(bb: Running) -> None:
    trace_id = new_trace_id()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{proxy_url(bb)}/api/embed", json={"model": "m", "input": "x"}, headers=headers_for(trace_id)
        )
    assert response.headers["content-encoding"] == "gzip"
    assert response.json() == {"embeddings": [[0.1, 0.2, 0.3]]}  # httpx decoded it; the proxy didn't
    [row] = await exchanges(bb, trace_id)
    stored = await bb.services.store.blobs.get(row.response_blob or "")
    assert json.loads(gzip.decompress(stored)) == {"embeddings": [[0.1, 0.2, 0.3]]}


async def test_secret_headers_are_forwarded_but_not_stored(bb: Running, upstream: FakeUpstream) -> None:
    trace_id = new_trace_id()
    secret_headers = {
        **headers_for(trace_id),
        "authorization": "Bearer jina_sk-very-secret",
        "x-api-key": "key-123",
        "cookie": "a=b",
        "x-custom-token": "t0k3n",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{proxy_url(bb)}/headers", json={}, headers=secret_headers)
    seen = response.json()["seen"]
    assert seen["authorization"] == "Bearer jina_sk-very-secret" and seen["x-api-key"] == "key-123"
    assert "traceparent" in seen
    [row] = await exchanges(bb, trace_id)
    stored = json.dumps([row.request_headers, row.response_headers])
    for secret in ("jina_sk-very-secret", "key-123", "a=b", "t0k3n", "session=abc"):
        assert secret not in stored
    assert "traceparent" in row.request_headers


async def test_refused_upstream_gives_502_and_is_stored(tmp_path: Path, migrated_db: Path) -> None:
    config = UpstreamConfig(name="jina", listen_port=0, target="http://127.0.0.1:9", record_paths=["/*"])
    async with running_blackbox(tmp_path, migrated_db, upstreams=[config], keep_unmatched=True) as running:
        assert running.services.proxy is not None
        trace_id = new_trace_id()
        url = f"http://127.0.0.1:{running.services.proxy.ports['jina']}/v1/embeddings"
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={"input": ["x"]}, headers=headers_for(trace_id))
        assert response.status_code == 502
        assert response.json()["upstream"] == "jina"
        [row] = await exchanges(running, trace_id)
        assert row.status == 502 and row.error is not None and row.error.startswith("upstream_error")


async def test_client_abort_is_stored(bb: Running) -> None:
    trace_id = new_trace_id()
    async with (
        httpx.AsyncClient(timeout=5) as client,
        client.stream("POST", f"{proxy_url(bb)}/slow", json={}, headers=headers_for(trace_id)) as response,
    ):
        async for _ in response.aiter_raw():
            break  # got part 1; hang up while the upstream is still sleeping
    [row] = await exchanges(bb, trace_id)
    assert row.error == "client_aborted"
    assert await bb.services.store.blobs.get(row.response_blob or "") == b'{"part": 1}\n'


async def test_unrecorded_path_passes_through(bb: Running, upstream: FakeUpstream) -> None:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{proxy_url(bb)}/api/tags", headers=headers_for(new_trace_id()))
    assert response.json()["models"][0]["name"] == "qwen3.5:4b"
    await asyncio.sleep(0.2)
    async with bb.services.store.read() as s:
        from sqlalchemy import func, select

        assert (await s.execute(select(func.count()).select_from(Exchange))).scalar_one() == 0


async def test_call_without_traceparent_is_unattributed(bb: Running) -> None:
    body = {"model": "m", "messages": [{"role": "user", "content": "who am i"}]}
    async with httpx.AsyncClient(base_url=bb.base_url) as web:
        async with httpx.AsyncClient() as client:
            await client.post(f"{proxy_url(bb)}/api/chat", json=body)
        [row] = await exchanges(bb, None)
        assert row.trace_id is None
        page = await web.get("/unattributed")
        assert row.id in page.text
        detail = await web.get(f"/exchanges/{row.id}")
        assert "who am i" in detail.text


async def test_run_waits_for_in_flight_call(bb: Running) -> None:
    """The root span has ended and the quiet period passed, but a proxy call is still streaming."""
    from tests.unit.test_assembler import fake_trace

    spans = fake_trace()
    trace_id = spans[0].trace_id
    await bb.services.assembler.ingest(spans)
    async with (
        httpx.AsyncClient(timeout=5) as client,
        client.stream("POST", f"{proxy_url(bb)}/slow", json={}, headers=headers_for(trace_id)) as response,
    ):
        async for _ in response.aiter_raw():
            await asyncio.sleep(1.0)  # well past quiet_seconds (0.3)
            run = await bb.services.store.reader.run_by_trace(trace_id)
            assert run is not None and run.status == "open"
            break
    async with asyncio.timeout(5):
        while True:
            run = await bb.services.store.reader.run_by_trace(trace_id)
            if run is not None and run.status != "open":
                break
            await asyncio.sleep(0.05)
    assert run.status == "complete", run.tags
    assert run.replayable is True
    steps = await bb.services.store.reader.steps(run.id)
    assert [s.kind for s in steps] == ["other"] and steps[0].status == "error"


async def test_added_latency_is_small(bb: Running, upstream: FakeUpstream) -> None:
    body = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    async with httpx.AsyncClient() as client:
        for url in (upstream.url, proxy_url(bb)):  # warm up connections
            await client.post(f"{url}/api/chat", json=body)
        direct, proxied = [], []
        for _ in range(40):
            start = time.perf_counter()
            await client.post(f"{upstream.url}/api/chat", json=body)
            direct.append(time.perf_counter() - start)
            start = time.perf_counter()
            await client.post(f"{proxy_url(bb)}/api/chat", json=body, headers=headers_for(new_trace_id()))
            proxied.append(time.perf_counter() - start)
    added_ms = (statistics.median(proxied) - statistics.median(direct)) * 1000
    print(f"\nproxy added latency: median {added_ms:.2f} ms per call")
    assert added_ms < 5


async def test_run_page_shows_exact_bytes(bb: Running) -> None:
    from tests.fixtures import paperpilot

    run_id = await paperpilot.insert_run(bb.services.store)
    async with httpx.AsyncClient(base_url=bb.base_url) as client:
        page = await client.get(f"/runs/{run_id}")
    assert page.status_code == 200
    assert "Exact bytes" in page.text and "/arxiv-papers-chunks/_search" in page.text
    assert "Attention Is All You Need" in page.text and "recorded" in page.text
