"""The OTLP/HTTP trace receiver: `POST /v1/traces`, protobuf or JSON, optionally gzip-compressed."""

import gzip
import logging
import zlib
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Request, Response
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse

from blackbox.otlp.decode import DecodeFailure, SpanData, decode_json, decode_protobuf

log = logging.getLogger(__name__)

MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_INFLATED_BYTES = 64 * 1024 * 1024

type SpanSink = Callable[[list[SpanData]], Awaitable[None]]


class BodyTooLarge(Exception):
    pass


async def _read_limited(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise BodyTooLarge
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise BodyTooLarge
        chunks.append(chunk)
    return b"".join(chunks)


def _inflate(body: bytes, encoding: str) -> bytes:
    if encoding in ("", "identity"):
        return body
    if encoding == "gzip":
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif encoding == "deflate":
        decompressor = zlib.decompressobj()
    else:
        raise DecodeFailure(f"unsupported content-encoding {encoding!r}")
    try:
        out = decompressor.decompress(body, MAX_INFLATED_BYTES + 1)
    except (zlib.error, gzip.BadGzipFile) as exc:
        raise DecodeFailure(f"invalid {encoding} body: {exc}") from exc
    if len(out) > MAX_INFLATED_BYTES:
        raise BodyTooLarge
    return out


def _error(status: int, message: str, as_json: bool) -> Response:
    if as_json:
        return Response(f'{{"error": "{message}"}}', status_code=status, media_type="application/json")
    return Response(message, status_code=status, media_type="text/plain")


def otlp_router(sink: SpanSink) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/traces")
    async def receive_traces(request: Request) -> Response:
        content_type = request.headers.get("content-type", "application/x-protobuf").split(";")[0].strip().lower()
        as_json = content_type == "application/json"
        if content_type not in ("application/x-protobuf", "application/protobuf", "application/json"):
            return _error(415, f"unsupported content-type {content_type}", as_json)
        try:
            body = await _read_limited(request, MAX_BODY_BYTES)
            body = _inflate(body, request.headers.get("content-encoding", "").strip().lower())
            spans = decode_json(body) if as_json else decode_protobuf(body)
        except BodyTooLarge:
            return _error(413, "body over 16 MB", as_json)
        except DecodeFailure as exc:
            log.warning("rejected OTLP export: %s", exc)
            return _error(400, str(exc).replace('"', "'"), as_json)
        await sink(spans)
        if as_json:
            return Response("{}", media_type="application/json")
        return Response(ExportTraceServiceResponse().SerializeToString(), media_type="application/x-protobuf")

    return router
