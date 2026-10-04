import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import httpx
import pytest
from sqlalchemy import select

from blackbox.config import OllamaConfig
from blackbox.judges.framework import JudgeError, load_judges, safe_eval
from blackbox.judges.inputs import BUILDERS, last_search_hits
from blackbox.judges.runner import JudgeRunner
from blackbox.llm.client import OllamaJSON
from blackbox.runs.context import load_run_context
from blackbox.store import Store
from blackbox.store.models import JudgeCall, Score
from tests.fixtures import paperpilot


class FakeOllama:
    """An httpx MockTransport that answers /api/chat with `reply(body)`, counting calls."""

    def __init__(self, reply: Callable[[dict[str, Any]], str]) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append(body)
        return httpx.Response(
            200,
            json={"model": body["model"], "message": {"role": "assistant", "content": self.reply(body)}, "done": True},
        )

    def client(self) -> OllamaJSON:
        config = OllamaConfig()
        return OllamaJSON(
            config, httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(self.handler))
        )


def passing(body: dict[str, Any]) -> str:
    return json.dumps({"unsupported_claims": [], "rationale": "Every claim is in the excerpts.", "verdict": "pass"})


JUDGES = load_judges("qwen3.5:4b")


def test_versions_change_with_any_part() -> None:
    judge = JUDGES["pp_faithfulness"]
    assert len(judge.version) == 12
    assert replace(judge, template=judge.template + ".").version != judge.version
    assert replace(judge, model="qwen3.5:9b").version != judge.version
    assert replace(judge, options={"temperature": 0.3}).version != judge.version
    assert replace(judge, input_builder=replace(judge.input_builder, version="2")).version != judge.version
    assert replace(judge).version == judge.version


def test_safe_eval() -> None:
    assert safe_eval("ending == 'answered'", {"ending": "answered"}) is True
    assert safe_eval("ending in ('a', 'b') and source != 'replay'", {"ending": "b", "source": "live"}) is True
    with pytest.raises(JudgeError):
        safe_eval("__import__('os')", {"ending": "x"})
    with pytest.raises(JudgeError):
        safe_eval("nope == 1", {"ending": "x"})


async def test_input_builder_uses_the_last_search_step(store: Store) -> None:
    run_id = await paperpilot.insert_run(store)
    run = await store.reader.run(run_id)
    assert run is not None
    ctx = await load_run_context(store, run)
    hits = last_search_hits(ctx)
    assert hits is not None and [h["paper_id"] for h in hits] == ["1706.03762", "1810.04805"]
    inputs = BUILDERS["pp_faithfulness"].build(ctx)
    assert inputs is not None
    assert inputs["question"] == paperpilot.QUESTION and inputs["answer"] == paperpilot.ANSWER
    for hit in paperpilot.HITS_SECOND:
        assert hit["text"] in inputs["context"]
    assert paperpilot.HITS_FIRST[0]["text"] not in inputs["context"]  # the earlier search's excerpt isn't there
    assert inputs["context"].startswith("[1] arXiv:1706.03762 Attention Is All You Need")


async def test_cache_means_no_second_model_call(store: Store) -> None:
    run_id = await paperpilot.insert_run(store)
    run = await store.reader.run(run_id)
    assert run is not None
    fake = FakeOllama(passing)
    runner = JudgeRunner(store, fake.client())
    judge = JUDGES["pp_faithfulness"]
    first = await runner.judge_run(judge, run)
    assert (first.status, first.label) == ("judged", "pass")
    second = await runner.judge_run(judge, run)
    assert (second.status, second.label) == ("cached", "pass")
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["think"] is False and call["stream"] is False and call["options"]["temperature"] == 0
    assert call["format"]["properties"]["verdict"]["enum"] == ["pass", "fail"]
    assert paperpilot.QUESTION in call["messages"][0]["content"]
    changed = replace(judge, template=judge.template + "\nBe strict.")
    await runner.judge_run(changed, run)
    assert len(fake.calls) == 2  # a new version judges again
    scores = await store.reader.scores(run.id)
    assert {(s.name, s.version, s.label) for s in scores} == {
        ("pp_faithfulness", judge.version, "pass"),
        ("pp_faithfulness", changed.version, "pass"),
    }


async def test_invalid_reply_is_retried_once_then_invalid(store: Store) -> None:
    run_id = await paperpilot.insert_run(store)
    run = await store.reader.run(run_id)
    assert run is not None
    fake = FakeOllama(lambda body: '{"verdict": "maybe"}')
    outcome = await JudgeRunner(store, fake.client()).judge_run(JUDGES["pp_relevance"], run)
    assert outcome.status == "invalid" and outcome.label is None
    assert len(fake.calls) == 2
    async with store.read() as s:
        [call] = (await s.execute(select(JudgeCall))).scalars().all()
        [score] = (await s.execute(select(Score))).scalars().all()
    assert (call.valid, call.attempts) == (False, 2) and call.error
    assert (score.label, score.value) == ("invalid", None)


async def test_scope_judge_compares_with_the_ending(store: Store) -> None:
    run_id = await paperpilot.insert_run(store)
    run = await store.reader.run(run_id)
    assert run is not None
    judge = JUDGES["pp_scope"]
    says_no = FakeOllama(lambda body: json.dumps({"rationale": "about ML", "should_answer": "no"}))
    outcome = await JudgeRunner(store, says_no.client()).judge_run(judge, run)
    assert outcome.label == "fail"  # the run answered a question the judge would have refused
    says_yes = FakeOllama(lambda body: json.dumps({"rationale": "about ML", "should_answer": "yes"}))
    outcome = await JudgeRunner(store, says_yes.client()).judge_run(
        replace(judge, options={"temperature": 0.0, "x": 1}), run
    )
    assert outcome.label == "pass"


async def test_majority_vote(store: Store) -> None:
    run_id = await paperpilot.insert_run(store)
    run = await store.reader.run(run_id)
    assert run is not None
    answers = iter(["pass", "fail", "pass"])
    fake = FakeOllama(lambda body: json.dumps({"rationale": "r", "verdict": next(answers)}))
    judge = replace(JUDGES["pp_relevance"], options={"temperature": 0, "samples": 3})
    outcome = await JudgeRunner(store, fake.client()).judge_run(judge, run)
    assert outcome.label == "pass" and outcome.details["samples"] == ["pass", "fail", "pass"]
    assert outcome.details["samples_disagree"] is True
    assert [c["options"]["temperature"] for c in fake.calls] == [0.3, 0.3, 0.3]


async def test_judge_report_in_replays_uses_the_cache(store: Store) -> None:
    from types import SimpleNamespace

    from blackbox.config import Settings
    from blackbox.judges.replay import judge_report

    run_id = await paperpilot.insert_run(store)
    run = await store.reader.run(run_id)
    assert run is not None
    fake = FakeOllama(
        lambda body: json.dumps({"unsupported_claims": [], "rationale": "r", "verdict": "pass", "should_answer": "yes"})
    )
    services = SimpleNamespace(settings=Settings(), judges=JUDGES, judge_runner=JudgeRunner(store, fake.client()))
    ctx = await load_run_context(store, run)
    report = await judge_report(services, None, ctx, ctx)  # type: ignore[arg-type]
    assert set(report["judges"]) == {"pp_faithfulness", "pp_relevance", "pp_scope"}
    assert report["judges"]["pp_faithfulness"] == {
        "source": {"status": "judged", "label": "pass", "rationale": "r"},
        "replay": {"status": "cached", "label": "pass", "rationale": "r"},
        "change": "same",
    }
    assert len(fake.calls) == 3  # one per judge; the "replay" (identical inputs) came from the cache
