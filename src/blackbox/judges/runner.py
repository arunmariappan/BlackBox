"""Running judges on runs: inputs from the run, the `judge_calls` cache (the same input is never judged twice by the
same version), Ollama with a JSON schema, and a `judge` score naming the version."""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.judges.framework import Judge, input_hash
from blackbox.judges.inputs import Inputs
from blackbox.llm.client import LLMUnavailable, OllamaJSON
from blackbox.runs.context import load_run_context
from blackbox.store import PreparedBlob, Store, insert_blob
from blackbox.store.models import Judge as JudgeRow
from blackbox.store.models import JudgeCall, Run, Score
from blackbox.util import new_id, now_ms

log = logging.getLogger(__name__)


@dataclass
class Verdict:
    """One judge decision on one input (possibly a majority of several samples)."""

    valid: bool
    parsed: dict[str, Any] | None
    call_ids: list[str] = field(default_factory=list)
    cached: bool = False
    samples: list[str] = field(default_factory=list)  # each sample's raw verdict, for majority votes
    error: str | None = None


@dataclass
class JudgeOutcome:
    run_id: str
    judge: str
    version: str
    status: str  # judged, cached, invalid, not_applicable, missing_input
    label: str | None = None
    rationale: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _verdict_field(parsed: dict[str, Any]) -> str | None:
    for key in ("verdict", "should_answer", "preferred"):
        if key in parsed:
            return str(parsed[key])
    return None


class JudgeRunner:
    def __init__(self, store: Store, llm: OllamaJSON) -> None:
        self.store = store
        self.llm = llm

    async def register(self, judge: Judge) -> None:
        """Record this judge version in `judges` (kept trusted state and agreement if it exists)."""

        async def op(session: AsyncSession) -> None:
            await session.execute(
                insert(JudgeRow)
                .values(
                    name=judge.name,
                    version=judge.version,
                    prompt_hash=judge.prompt_hash,
                    model=judge.model,
                    options=judge.options,
                    created_ms=now_ms(),
                )
                .on_conflict_do_nothing()
            )

        await self.store.write(op)

    async def cached(self, judge: Judge, digest: str) -> list[JudgeCall]:
        async with self.store.read() as s:
            query = (
                select(JudgeCall)
                .where(JudgeCall.judge_version == judge.version, JudgeCall.input_hash == digest)
                .order_by(JudgeCall.created_ms)
            )
            return list((await s.execute(query)).scalars())

    async def decide(
        self,
        judge: Judge,
        inputs: Inputs,
        *,
        run_id: str | None = None,
        use_cache: bool = True,
        options: dict[str, Any] | None = None,
        samples: int | None = None,
    ) -> Verdict:
        """Judge `inputs`. With the cache on, an input this version already judged validly costs no model call."""
        digest = input_hash(inputs)
        wanted = samples or judge.samples
        if use_cache:
            previous = [call for call in await self.cached(judge, digest) if call.valid and call.parsed is not None]
            if len(previous) >= wanted:
                return _majority(
                    [c.parsed for c in previous[:wanted] if c.parsed], [c.id for c in previous[:wanted]], True
                )
        parsed_samples: list[dict[str, Any]] = []
        call_ids: list[str] = []
        error = None
        llm_options = {**judge.llm_options, **(options or {})}
        if wanted > 1 and "temperature" not in (options or {}):
            llm_options["temperature"] = judge.options.get("sample_temperature", 0.3)
        for _ in range(wanted):
            result = await self.llm.call(judge.messages(inputs), judge.output, model=judge.model, options=llm_options)
            call_id = await self._store_call(
                judge,
                digest,
                run_id,
                result.request,
                result.raw,
                result.parsed,
                result.valid,
                result.error,
                result.attempts,
                result.latency_ms,
            )
            call_ids.append(call_id)
            if result.valid and result.parsed is not None:
                parsed_samples.append(result.parsed.model_dump())
            else:
                error = result.error
        if not parsed_samples:
            return Verdict(False, None, call_ids, error=error or "invalid reply")
        verdict = _majority(parsed_samples, call_ids, False)
        if len(parsed_samples) < wanted:
            verdict.error = f"{wanted - len(parsed_samples)} of {wanted} samples invalid"
        return verdict

    async def _store_call(
        self,
        judge: Judge,
        digest: str,
        run_id: str | None,
        request: dict[str, Any],
        raw: str,
        parsed: BaseModel | None,
        valid: bool,
        error: str | None,
        attempts: int,
        latency_ms: int,
    ) -> str:
        prompt = PreparedBlob.of(json.dumps(request, ensure_ascii=False).encode(), "application/json")
        response = PreparedBlob.of(raw.encode(), "text/plain")
        call_id = new_id()

        async def op(session: AsyncSession) -> None:
            await insert_blob(session, prompt)
            await insert_blob(session, response)
            session.add(
                JudgeCall(
                    id=call_id,
                    judge_name=judge.name,
                    judge_version=judge.version,
                    run_id=run_id,
                    input_hash=digest,
                    prompt_blob=prompt.sha256,
                    response_blob=response.sha256,
                    parsed=parsed.model_dump() if parsed is not None else None,
                    valid=valid,
                    error=error,
                    attempts=attempts,
                    latency_ms=latency_ms,
                    created_ms=now_ms(),
                )
            )

        await self.store.write(op)
        return call_id

    async def judge_run(self, judge: Judge, run: Run, *, use_cache: bool = True) -> JudgeOutcome:
        if not judge.applies(run):
            return JudgeOutcome(run.id, judge.name, judge.version, "not_applicable")
        ctx = await load_run_context(self.store, run)
        inputs = judge.input_builder.build(ctx)
        if inputs is None:
            return JudgeOutcome(run.id, judge.name, judge.version, "missing_input")
        await self.register(judge)
        verdict = await self.decide(judge, inputs, run_id=run.id, use_cache=use_cache)
        label: str | None = None
        rationale = None
        details: dict[str, Any] = {
            "calls": verdict.call_ids,
            "label_question": judge.label_question,
            "cached": verdict.cached,
        }
        if verdict.valid and verdict.parsed is not None:
            parsed_model = judge.output.model_validate(verdict.parsed)
            label = judge.to_label(parsed_model, run)
            rationale = verdict.parsed.get("rationale")
            details["parsed"] = verdict.parsed
            if len(verdict.samples) > 1:
                details["samples"] = verdict.samples
                details["samples_disagree"] = len(set(verdict.samples)) > 1
        else:
            details["error"] = verdict.error
        await self.write_score(judge, run.id, label, rationale, details)
        status = "invalid" if label is None else ("cached" if verdict.cached else "judged")
        return JudgeOutcome(run.id, judge.name, judge.version, status, label, rationale, details)

    async def write_score(
        self, judge: Judge, run_id: str, label: str | None, rationale: str | None, details: dict[str, Any]
    ) -> None:
        async def op(session: AsyncSession) -> None:
            await session.execute(
                delete(Score).where(
                    Score.run_id == run_id,
                    Score.kind == "judge",
                    Score.name == judge.name,
                    Score.version == judge.version,
                )
            )
            session.add(
                Score(
                    id=new_id(),
                    run_id=run_id,
                    kind="judge",
                    name=judge.name,
                    version=judge.version,
                    value=None if label is None else (1.0 if label == "pass" else 0.0),
                    label=label or "invalid",
                    rationale=rationale,
                    details=details,
                    created_ms=now_ms(),
                )
            )

        await self.store.write(op)


def _majority(parsed: list[dict[str, Any]], call_ids: list[str], cached: bool) -> Verdict:
    labels = [_verdict_field(p) or "" for p in parsed]
    winner, _ = Counter(labels).most_common(1)[0]
    chosen = next(p for p, label in zip(parsed, labels, strict=True) if label == winner)
    return Verdict(True, chosen, call_ids, cached, labels)


async def judge_many(
    runner: JudgeRunner, judge: Judge, runs: list[Run], *, use_cache: bool = True
) -> list[JudgeOutcome]:
    outcomes = []
    for run in runs:
        try:
            outcomes.append(await runner.judge_run(judge, run, use_cache=use_cache))
        except LLMUnavailable as exc:
            log.error("judge %s stopped: %s", judge.name, exc)
            raise
    return outcomes
