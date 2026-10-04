"""SQLAlchemy models for every table BlackBox stores.

Times are epoch milliseconds (UTC), except span timestamps, which keep OpenTelemetry's nanoseconds. Bodies and other
large payloads live in `blobs`, keyed by the SHA-256 of their uncompressed bytes; rows point at them by hash.
"""

from typing import Any, ClassVar

from sqlalchemy import JSON, BigInteger, ForeignKey, Index, LargeBinary, MetaData, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

Json = dict[str, Any]


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map: ClassVar[dict[Any, Any]] = {
        dict[str, Any]: JSON,
        list[Any]: JSON,
        int: BigInteger,
        bytes: LargeBinary,
    }


class Run(Base):
    """One execution of an agent for one input; one trace id."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(32), unique=True)
    profile: Mapped[str | None]
    status: Mapped[str] = mapped_column(default="open")  # open, complete, failed_assembly
    source: Mapped[str] = mapped_column(default="live")  # live, traffic, suite, replay
    started_ms: Mapped[int | None]
    ended_ms: Mapped[int | None]
    updated_ms: Mapped[int]  # last time anything arrived for this trace
    entry_request_blob: Mapped[str | None]  # the exact request that started the run, when BlackBox started it
    output_blob: Mapped[str | None]
    input_text: Mapped[str | None]  # a short form of the input, for lists
    output_text: Mapped[str | None]
    ending: Mapped[str | None]
    model: Mapped[str | None]
    replayable: Mapped[bool] = mapped_column(default=False)
    session_id: Mapped[str | None]
    replay_of: Mapped[str | None]
    remote_parent_span_id: Mapped[str | None]  # the span id in the traceparent BlackBox sent, when it started the run
    step_count: Mapped[int] = mapped_column(default=0)
    input_tokens: Mapped[int] = mapped_column(default=0)
    output_tokens: Mapped[int] = mapped_column(default=0)
    duration_ms: Mapped[int | None]
    tags: Mapped[dict[str, Any]] = mapped_column(default=dict)

    __table_args__ = (
        Index(None, "profile", "started_ms"),
        Index(None, "status"),
    )


class Span(Base):
    """A raw span as received over OTLP. Attribute values over 4 KB are moved to blobs (listed in `blob_refs`)."""

    __tablename__ = "spans"

    trace_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    span_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    parent_span_id: Mapped[str | None]
    name: Mapped[str]
    kind: Mapped[str]  # internal, server, client, producer, consumer, unspecified
    service: Mapped[str | None]
    scope: Mapped[str | None]
    start_ns: Mapped[int]
    end_ns: Mapped[int]
    status_code: Mapped[str] = mapped_column(default="unset")  # unset, ok, error
    status_message: Mapped[str | None]
    flags: Mapped[int] = mapped_column(default=0)
    attributes: Mapped[dict[str, Any]] = mapped_column(default=dict)
    events: Mapped[list[Any]] = mapped_column(default=list)
    resource: Mapped[dict[str, Any]] = mapped_column(default=dict)
    blob_refs: Mapped[list[Any]] = mapped_column(default=list)
    received_ms: Mapped[int]

    __table_args__ = (Index(None, "trace_id"),)


class Exchange(Base):
    """One recorded HTTP request and response at the proxy."""

    __tablename__ = "exchanges"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    trace_id: Mapped[str | None] = mapped_column(String(32))
    parent_span_id: Mapped[str | None]
    session_id: Mapped[str | None]
    upstream: Mapped[str]
    seq: Mapped[int]  # per-trace order of request start, from 1
    method: Mapped[str]
    path: Mapped[str]
    query: Mapped[str] = mapped_column(default="")
    request_headers: Mapped[dict[str, Any]] = mapped_column(default=dict)  # redacted
    request_blob: Mapped[str | None]
    sent_request_blob: Mapped[str | None]  # what was actually sent, when an override or patch changed it
    request_key: Mapped[str]  # SHA-256 of the canonical request
    status: Mapped[int | None]
    response_headers: Mapped[dict[str, Any]] = mapped_column(default=dict)
    response_blob: Mapped[str | None]
    stream: Mapped[bool] = mapped_column(default=False)
    chunk_times: Mapped[list[Any]] = mapped_column(default=list)  # [[byte_offset, ms_since_start], ...]
    started_ms: Mapped[int]
    first_byte_ms: Mapped[int | None]
    ended_ms: Mapped[int | None]
    error: Mapped[str | None]
    served_from: Mapped[str] = mapped_column(default="live")  # live, tape:<exchange id>, patched
    divergence: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    __table_args__ = (
        Index(None, "trace_id", "seq"),
        Index(None, "session_id"),
    )


class Step(Base):
    """One model or tool call within a run, built when the run completes."""

    __tablename__ = "steps"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    idx: Mapped[int] = mapped_column(primary_key=True)  # from 1, in order of request start
    kind: Mapped[str]  # llm, tool, embedding, other
    node: Mapped[str | None]
    exchange_id: Mapped[str | None]
    span_id: Mapped[str | None]
    model: Mapped[str | None]
    tool_name: Mapped[str | None]
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
    started_ms: Mapped[int | None]
    latency_ms: Mapped[int | None]
    status: Mapped[str] = mapped_column(default="ok")  # ok, error
    view: Mapped[dict[str, Any]] = mapped_column(default=dict)


class Session(Base):
    """One replay or fork of a recorded run."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(32), unique=True)
    source_run_id: Mapped[str]
    mode: Mapped[str]  # exact, fork, auto_fork
    fork_step: Mapped[int | None]
    overrides: Mapped[dict[str, Any]] = mapped_column(default=dict)  # model, patches, speed, lenient
    status: Mapped[str] = mapped_column(default="active")  # active, complete, expired, failed
    created_ms: Mapped[int]
    ended_ms: Mapped[int | None]
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)  # the fidelity report


class RecordedValue(Base):
    """A value an SDK agent drew from its clock or random generator."""

    __tablename__ = "recorded_values"

    trace_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    seq: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str]  # now, uuid, random
    value: Mapped[str]


class Score(Base):
    """Every number or verdict about a run: metrics, judge verdicts, checker results."""

    __tablename__ = "scores"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    kind: Mapped[str]  # metric, judge, checker
    name: Mapped[str]
    version: Mapped[str]
    value: Mapped[float | None]
    label: Mapped[str | None]
    rationale: Mapped[str | None]
    details: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_ms: Mapped[int]

    __table_args__ = (
        Index(None, "run_id"),
        Index(None, "name", "version"),
    )


class JudgeCall(Base):
    """Cache and audit trail of judge calls: the same input is never judged twice by the same version."""

    __tablename__ = "judge_calls"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    judge_name: Mapped[str]
    judge_version: Mapped[str]
    run_id: Mapped[str | None]
    input_hash: Mapped[str]
    prompt_blob: Mapped[str | None]
    response_blob: Mapped[str | None]
    parsed: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    valid: Mapped[bool]
    error: Mapped[str | None]
    attempts: Mapped[int] = mapped_column(default=1)
    latency_ms: Mapped[int | None]
    created_ms: Mapped[int]

    __table_args__ = (Index(None, "judge_version", "input_hash"),)


class Judge(Base):
    __tablename__ = "judges"

    name: Mapped[str] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(primary_key=True)
    prompt_hash: Mapped[str]
    model: Mapped[str]
    options: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_ms: Mapped[int]
    trusted: Mapped[bool] = mapped_column(default=False)
    agreement: Mapped[dict[str, Any]] = mapped_column(default=dict)


class Label(Base):
    """Your own verdict on a run, used to measure a judge's agreement."""

    __tablename__ = "labels"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    question: Mapped[str]
    value: Mapped[str]  # pass, fail, unsure (or a scale value)
    note: Mapped[str | None]
    labeler: Mapped[str] = mapped_column(default="you")
    created_ms: Mapped[int]

    __table_args__ = (Index(None, "run_id", "question"),)


class Failure(Base):
    __tablename__ = "failures"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    profile: Mapped[str]
    signature: Mapped[dict[str, Any]] = mapped_column(default=dict)
    description: Mapped[str | None]
    category: Mapped[str | None]
    where_step: Mapped[int | None]
    embedding: Mapped[bytes | None]  # float32 bytes
    cluster_id: Mapped[str | None]
    created_ms: Mapped[int]

    __table_args__ = (Index(None, "profile", "cluster_id"),)


class Cluster(Base):
    __tablename__ = "clusters"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    profile: Mapped[str]
    title: Mapped[str | None]
    likely_cause: Mapped[str | None]
    evidence: Mapped[list[Any]] = mapped_column(default=list)
    suggested_fix: Mapped[str | None]
    cause_status: Mapped[str | None]  # supported, unsupported, invalid
    centroid: Mapped[bytes | None]
    radius: Mapped[float | None]
    size: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(default="new")  # new, known, fixed, retired
    named_members: Mapped[list[Any]] = mapped_column(default=list)  # members when the name was generated
    created_ms: Mapped[int]
    updated_ms: Mapped[int]


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    profile: Mapped[str]
    rule: Mapped[str]
    severity: Mapped[str] = mapped_column(default="warning")
    status: Mapped[str] = mapped_column(default="open")  # open, resolved
    opened_ms: Mapped[int]
    closed_ms: Mapped[int | None]
    clear_count: Mapped[int] = mapped_column(default=0)
    details: Mapped[dict[str, Any]] = mapped_column(default=dict)
    notified: Mapped[bool] = mapped_column(default=False)

    __table_args__ = (Index(None, "profile", "rule", "status"),)


class Job(Base):
    """Durable work queue."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    kind: Mapped[str]
    lane: Mapped[str] = mapped_column(default="cpu")  # cpu, model
    run_id: Mapped[str | None]
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)
    status: Mapped[str] = mapped_column(default="queued")  # queued, running, done, failed
    attempts: Mapped[int] = mapped_column(default=0)
    available_ms: Mapped[int]
    last_error: Mapped[str | None]
    created_ms: Mapped[int]
    updated_ms: Mapped[int]

    __table_args__ = (Index(None, "status", "available_ms"),)


class SuiteRun(Base):
    __tablename__ = "suite_runs"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    suite: Mapped[str]
    baseline: Mapped[str | None]
    candidate: Mapped[str | None]
    mode: Mapped[str]  # replay, fork, live
    status: Mapped[str] = mapped_column(default="running")
    verdict: Mapped[str | None]
    report_blob: Mapped[str | None]
    details: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_ms: Mapped[int]
    finished_ms: Mapped[int | None]


class SuiteCase(Base):
    __tablename__ = "suite_cases"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    suite_run_id: Mapped[str] = mapped_column(ForeignKey("suite_runs.id", ondelete="CASCADE"))
    case_id: Mapped[str]
    repeat: Mapped[int] = mapped_column(default=0)
    input: Mapped[dict[str, Any]] = mapped_column(default=dict)
    baseline_run_id: Mapped[str | None]
    candidate_run_id: Mapped[str | None]
    outcome: Mapped[str | None]
    details: Mapped[dict[str, Any]] = mapped_column(default=dict)


class Blob(Base):
    """zstd-compressed bytes keyed by the SHA-256 of the uncompressed bytes."""

    __tablename__ = "blobs"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    size: Mapped[int]
    content_type: Mapped[str | None]
    data: Mapped[bytes]
