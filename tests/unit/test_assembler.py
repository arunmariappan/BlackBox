import copy
from pathlib import Path

from sqlalchemy import func, select

from blackbox.config import RunsConfig
from blackbox.events import EventBus
from blackbox.otlp.decode import SpanData, decode_json
from blackbox.profiles import default_registry
from blackbox.runs.assembler import RunAssembler
from blackbox.store import Store
from blackbox.store.models import Job, Run, Span
from tests.factories import span_id, trace_id
from tests.harness import registry_with
from tests.profiles import FakeAgentProfile

FIXTURES = Path(__file__).parent.parent / "fixtures" / "otlp"


class Clock:
    def __init__(self) -> None:
        self.ms = 1_000_000

    def __call__(self) -> int:
        return self.ms

    def advance(self, seconds: float) -> None:
        self.ms += int(seconds * 1000)


def make(store: Store, clock: Clock, *, keep_unmatched: bool = False) -> RunAssembler:
    config = RunsConfig(quiet_seconds=5, orphan_seconds=60, keep_unmatched=keep_unmatched)
    return RunAssembler(store, registry_with(FakeAgentProfile()), EventBus(), config, clock=clock)


def fake_trace() -> list[SpanData]:
    """invoke_agent fake-agent → plan → chat; act → execute_tool; answer → chat."""
    tid = trace_id()
    root, plan, act, answer = span_id(), span_id(), span_id(), span_id()
    t = 1_790_000_000_000_000_000

    def s(sid: str, parent: str | None, name: str, start: int, end: int, **attrs: object) -> SpanData:
        return SpanData(
            tid, sid, parent, name, start_ns=t + start * 10**6, end_ns=t + end * 10**6, attributes=dict(attrs)
        )

    return [
        s(root, None, "invoke_agent fake-agent", 0, 100, **{"gen_ai.operation.name": "invoke_agent"}),
        s(plan, root, "plan", 1, 30),
        s(
            span_id(),
            plan,
            "chat m",
            2,
            29,
            **{
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "m",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 3,
            },
        ),
        s(act, root, "act", 31, 60),
        s(
            span_id(),
            act,
            "execute_tool get_logs",
            32,
            59,
            **{"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "get_logs"},
        ),
        s(answer, root, "answer", 61, 99),
        s(
            span_id(),
            answer,
            "chat m",
            62,
            98,
            **{
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "m",
                "gen_ai.usage.input_tokens": 20,
                "gen_ai.usage.output_tokens": 7,
            },
        ),
    ]


async def test_out_of_order_batches_make_one_run(store: Store) -> None:
    clock = Clock()
    assembler = make(store, clock)
    spans = fake_trace()
    # Children first, root last, in three batches, as exporters send them.
    await assembler.ingest([spans[4], spans[6]])
    await assembler.ingest([spans[2], spans[5], spans[3]])
    clock.advance(10)
    assert await assembler.tick() == []  # root span hasn't ended (arrived) yet
    await assembler.ingest([spans[1], spans[0]])
    clock.advance(4.9)
    assert await assembler.tick() == []  # not quiet long enough
    clock.advance(0.2)
    results = await assembler.tick()
    assert [r.outcome for r in results] == ["complete"]
    run = await store.reader.run_by_trace(spans[0].trace_id)
    assert run is not None and run.status == "complete" and run.profile == "fake-agent"
    steps = await store.reader.steps(run.id)
    assert [(s.idx, s.kind, s.node, s.tool_name) for s in steps] == [
        (1, "llm", "plan", None),
        (2, "tool", "act", "get_logs"),
        (3, "llm", "answer", None),
    ]
    assert (run.step_count, run.input_tokens, run.output_tokens, run.duration_ms) == (3, 30, 10, 100)
    assert run.replayable is False
    async with store.read() as s:
        jobs = (await s.execute(select(Job))).scalars().all()
    assert [(j.kind, j.run_id) for j in jobs] == [("run_completed", run.id)]


async def test_root_open_blocks_completion_until_it_arrives(store: Store) -> None:
    clock = Clock()
    assembler = make(store, clock)
    spans = fake_trace()
    await assembler.ingest(spans[1:])
    for _ in range(5):
        clock.advance(10)
        assert await assembler.tick() == []
    await assembler.ingest(spans[:1])
    clock.advance(5)
    assert [r.outcome for r in await assembler.tick()] == ["complete"]


async def test_in_flight_call_blocks_completion(store: Store) -> None:
    clock = Clock()
    assembler = make(store, clock)
    spans = fake_trace()
    await assembler.ingest(spans)
    assembler.begin_call(spans[0].trace_id)
    clock.advance(30)
    assert await assembler.tick() == []
    assembler.end_call(spans[0].trace_id)
    clock.advance(5)
    assert [r.outcome for r in await assembler.tick()] == ["complete"]


async def test_unmatched_trace_is_deleted(store: Store) -> None:
    clock = Clock()
    assembler = make(store, clock)
    spans = fake_trace()
    for span in spans:
        span.name = span.name.replace("fake-agent", "someone-else")
    await assembler.ingest(spans)
    clock.advance(6)
    assert [r.outcome for r in await assembler.tick()] == ["deleted"]
    async with store.read() as s:
        assert (await s.execute(select(func.count()).select_from(Run))).scalar_one() == 0
        assert (await s.execute(select(func.count()).select_from(Span))).scalar_one() == 0


async def test_unmatched_trace_is_kept_when_configured(store: Store) -> None:
    clock = Clock()
    assembler = make(store, clock, keep_unmatched=True)
    spans = fake_trace()
    for span in spans:
        span.name = span.name.replace("fake-agent", "someone-else")
    await assembler.ingest(spans)
    clock.advance(6)
    assert [r.outcome for r in await assembler.tick()] == ["complete"]
    run = await store.reader.run_by_trace(spans[0].trace_id)
    assert run is not None and run.profile is None and run.step_count == 3


async def test_late_span_is_stored_but_does_not_reopen(store: Store) -> None:
    clock = Clock()
    assembler = make(store, clock)
    spans = fake_trace()
    late = copy.copy(spans[2])
    late.span_id = span_id()
    await assembler.ingest(spans)
    clock.advance(6)
    await assembler.tick()
    await assembler.ingest([late])
    assert not assembler.is_open(spans[0].trace_id)
    assert len(await store.reader.spans(spans[0].trace_id)) == 8


async def test_orphan_trace_completes_after_orphan_timeout(store: Store) -> None:
    clock = Clock()
    assembler = make(store, clock, keep_unmatched=True)
    spans = fake_trace()[1:]  # the root never arrives
    await assembler.ingest(spans)
    clock.advance(59)
    assert await assembler.tick() == []
    clock.advance(2)
    assert [r.outcome for r in await assembler.tick()] == ["complete"]


async def test_paperpilot_fixture(store: Store) -> None:
    clock = Clock()
    assembler = RunAssembler(store, default_registry(), EventBus(), RunsConfig(), clock=clock)
    spans = decode_json((FIXTURES / "paperpilot-agentic.json").read_bytes())
    await assembler.ingest(spans)
    clock.advance(6)
    assert [r.outcome for r in await assembler.tick()] == ["complete"]
    run = await store.reader.run_by_trace(spans[0].trace_id)
    assert run is not None
    assert run.profile == "paperpilot"
    assert run.ending == "answered"
    assert run.input_text == "What are transformer architectures?"
    assert run.output_text is not None and run.output_text.startswith("Transformer architectures replace recurrence")
    node_spans = {s.name for s in spans} & {
        "guardrail_validation",
        "document_retrieval_initiation",
        "document_grading",
        "query_rewriting",
        "answer_generation",
    }
    assert len(node_spans) == 5
    steps = await store.reader.steps(run.id)
    assert [(s.kind, s.node) for s in steps] == [
        ("llm", "guardrail_validation"),
        ("llm", "document_grading"),
        ("llm", "query_rewriting"),
        ("llm", "document_grading"),
        ("llm", "answer_generation"),
    ]
    assert steps[0].view["messages"][0]["role"] == "system"
    assert run.input_tokens == 210 + 150 + 90 + 260 + 420
    assert run.duration_ms == 9000


async def test_restart_completes_open_runs(store: Store) -> None:
    clock = Clock()
    first = make(store, clock)
    spans = fake_trace()
    await first.ingest(spans)
    await first.stop()  # "crash" before completion
    clock.advance(10)
    second = make(store, clock)
    await second.recover()
    assert second.is_open(spans[0].trace_id)
    assert [r.outcome for r in await second.tick()] == ["complete"]
