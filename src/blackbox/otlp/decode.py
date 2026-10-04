"""Decode OTLP/HTTP trace exports (protobuf or JSON) into plain span records.

Both encodings go through the same protobuf message, so the same trace sent either way gives identical records.
OTLP/JSON writes trace and span ids as hex, not as protobuf JSON's base64; they are converted before parsing.
"""

import base64
import json
from dataclasses import dataclass, field
from typing import Any

from google.protobuf import json_format
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue

SPAN_KINDS = {0: "unspecified", 1: "internal", 2: "server", 3: "client", 4: "producer", 5: "consumer"}
STATUS_CODES = {0: "unset", 1: "ok", 2: "error"}
FLAG_HAS_IS_REMOTE = 0x100
FLAG_IS_REMOTE = 0x200


class DecodeFailure(ValueError):
    pass


@dataclass
class SpanData:
    """One span as BlackBox works with it: ids in lowercase hex, attributes as plain Python values."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str = "internal"
    service: str | None = None
    scope: str | None = None
    start_ns: int = 0
    end_ns: int = 0
    status_code: str = "unset"
    status_message: str | None = None
    flags: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    resource: dict[str, Any] = field(default_factory=dict)

    @property
    def parent_is_remote(self) -> bool | None:
        """True or False when the exporter said whether the parent is remote; None when it didn't."""
        if self.flags & FLAG_HAS_IS_REMOTE:
            return bool(self.flags & FLAG_IS_REMOTE)
        return None

    @property
    def duration_ms(self) -> float:
        return max(0, self.end_ns - self.start_ns) / 1_000_000


def any_value(value: AnyValue) -> Any:
    kind = value.WhichOneof("value")
    if kind is None:
        return None
    if kind == "string_value":
        return value.string_value
    if kind == "bool_value":
        return value.bool_value
    if kind == "int_value":
        return value.int_value
    if kind == "double_value":
        return value.double_value
    if kind == "array_value":
        return [any_value(v) for v in value.array_value.values]
    if kind == "kvlist_value":
        return key_values(value.kvlist_value.values)
    if kind == "bytes_value":
        return base64.b64encode(value.bytes_value).decode("ascii")
    return None


def key_values(items: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    item: KeyValue
    for item in items:
        result[item.key] = any_value(item.value)
    return result


def _hex(raw: bytes) -> str:
    return raw.hex()


def spans_from_request(request: ExportTraceServiceRequest) -> list[SpanData]:
    spans: list[SpanData] = []
    for resource_spans in request.resource_spans:
        resource = key_values(resource_spans.resource.attributes)
        service = resource.get("service.name")
        for scope_spans in resource_spans.scope_spans:
            scope = scope_spans.scope.name or None
            for span in scope_spans.spans:
                if len(span.trace_id) != 16 or len(span.span_id) != 8:
                    raise DecodeFailure(f"span {span.name!r} has an invalid trace or span id")
                parent = _hex(span.parent_span_id) if span.parent_span_id else None
                spans.append(
                    SpanData(
                        trace_id=_hex(span.trace_id),
                        span_id=_hex(span.span_id),
                        parent_span_id=parent if parent and parent != "0" * 16 else None,
                        name=span.name,
                        kind=SPAN_KINDS.get(span.kind, "unspecified"),
                        service=str(service) if service is not None else None,
                        scope=scope,
                        start_ns=span.start_time_unix_nano,
                        end_ns=span.end_time_unix_nano,
                        status_code=STATUS_CODES.get(span.status.code, "unset"),
                        status_message=span.status.message or None,
                        flags=span.flags,
                        attributes=key_values(span.attributes),
                        events=[
                            {
                                "name": event.name,
                                "time_ns": event.time_unix_nano,
                                "attributes": key_values(event.attributes),
                            }
                            for event in span.events
                        ],
                        resource=resource,
                    )
                )
    return spans


def decode_protobuf(body: bytes) -> list[SpanData]:
    request = ExportTraceServiceRequest()
    try:
        request.ParseFromString(body)
    except DecodeError as exc:
        raise DecodeFailure(f"invalid OTLP protobuf: {exc}") from exc
    return spans_from_request(request)


def _hex_ids_to_base64(node: Any) -> None:
    """Rewrite OTLP/JSON's hex ids in place into the base64 that protobuf's JSON mapping expects."""
    if isinstance(node, list):
        for item in node:
            _hex_ids_to_base64(item)
        return
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if key in ("traceId", "spanId", "parentSpanId", "trace_id", "span_id", "parent_span_id") and isinstance(
            value, str
        ):
            if value == "":
                continue
            try:
                node[key] = base64.b64encode(bytes.fromhex(value)).decode("ascii")
            except ValueError as exc:
                raise DecodeFailure(f"{key} {value!r} is not hex") from exc
        else:
            _hex_ids_to_base64(value)


def decode_json(body: bytes) -> list[SpanData]:
    try:
        document = json.loads(body)
    except json.JSONDecodeError as exc:
        raise DecodeFailure(f"invalid JSON: {exc}") from exc
    _hex_ids_to_base64(document)
    request = ExportTraceServiceRequest()
    try:
        json_format.ParseDict(document, request, ignore_unknown_fields=True)
    except json_format.ParseError as exc:
        raise DecodeFailure(f"invalid OTLP JSON: {exc}") from exc
    return spans_from_request(request)


def encode_json(spans: list[SpanData]) -> dict[str, Any]:
    """The OTLP/JSON form of `spans` (hex ids), used to write fixtures and by tests."""
    request = encode_request(spans)
    document: dict[str, Any] = json_format.MessageToDict(
        request, preserving_proto_field_name=False, use_integers_for_enums=True
    )

    def to_hex(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                to_hex(item)
        elif isinstance(node, dict):
            for key, value in node.items():
                if key in ("traceId", "spanId", "parentSpanId") and isinstance(value, str):
                    node[key] = base64.b64decode(value).hex()
                else:
                    to_hex(value)

    to_hex(document)
    return document


def _to_any_value(value: Any) -> AnyValue:
    result = AnyValue()
    if isinstance(value, bool):
        result.bool_value = value
    elif isinstance(value, int):
        result.int_value = value
    elif isinstance(value, float):
        result.double_value = value
    elif isinstance(value, str):
        result.string_value = value
    elif isinstance(value, list | tuple):
        result.array_value.values.extend(_to_any_value(v) for v in value)
    elif isinstance(value, dict):
        result.kvlist_value.values.extend(KeyValue(key=k, value=_to_any_value(v)) for k, v in value.items())
    elif value is not None:
        result.string_value = str(value)
    return result


def encode_request(spans: list[SpanData]) -> ExportTraceServiceRequest:
    """Build an export request from span records (grouped by resource and scope)."""
    from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, Span, Status

    request = ExportTraceServiceRequest()
    groups: dict[tuple[str, str], ResourceSpans] = {}
    scope_index: dict[tuple[str, str, str], Any] = {}
    kinds = {v: k for k, v in SPAN_KINDS.items()}
    codes = {v: k for k, v in STATUS_CODES.items()}
    for span in spans:
        resource_key = json.dumps(span.resource, sort_keys=True)
        group = groups.get((resource_key, ""))
        if group is None:
            group = request.resource_spans.add()
            group.resource.attributes.extend(KeyValue(key=k, value=_to_any_value(v)) for k, v in span.resource.items())
            groups[(resource_key, "")] = group
        scope_key = (resource_key, "", span.scope or "")
        scope_spans = scope_index.get(scope_key)
        if scope_spans is None:
            scope_spans = group.scope_spans.add()
            scope_spans.scope.name = span.scope or ""
            scope_index[scope_key] = scope_spans
        out: Span = scope_spans.spans.add()
        out.trace_id = bytes.fromhex(span.trace_id)
        out.span_id = bytes.fromhex(span.span_id)
        if span.parent_span_id:
            out.parent_span_id = bytes.fromhex(span.parent_span_id)
        out.name = span.name
        out.kind = kinds.get(span.kind, 0)  # type: ignore[assignment]
        out.start_time_unix_nano = span.start_ns
        out.end_time_unix_nano = span.end_ns
        out.flags = span.flags
        out.status.CopyFrom(Status(code=codes.get(span.status_code, 0), message=span.status_message or ""))  # type: ignore[arg-type]
        out.attributes.extend(KeyValue(key=k, value=_to_any_value(v)) for k, v in span.attributes.items())
        for event in span.events:
            added = out.events.add()
            added.name = event.get("name", "")
            added.time_unix_nano = int(event.get("time_ns", 0))
            added.attributes.extend(
                KeyValue(key=k, value=_to_any_value(v)) for k, v in (event.get("attributes") or {}).items()
            )
    return request
