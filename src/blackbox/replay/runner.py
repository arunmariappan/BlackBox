"""The replay runner: create a session with a fresh trace id, start the source run's input through the profile with a
`traceparent` carrying it, wait for the new run to complete, and write the fidelity report."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.net import new_trace_id
from blackbox.profiles.base import StartRequest
from blackbox.proxy.matching import match_key
from blackbox.proxy.sessions import MODES, ReplaySession, SessionManager, SessionSpec, TapeEntry
from blackbox.replay.report import build_report
from blackbox.runs.context import load_run_context
from blackbox.runs.starter import create_run, send_entry_request
from blackbox.store.models import Run, Session
from blackbox.util import new_id, now_ms

if TYPE_CHECKING:
    from blackbox.services import Services

log = logging.getLogger(__name__)

SESSION_BAGGAGE = "blackbox.session"


class ReplayError(Exception):
    pass


@dataclass
class PreparedReplay:
    session: ReplaySession
    request: StartRequest
    rebuilt: bool
    source: Run


def session_manager(services: Services) -> SessionManager:
    proxy = services.proxy
    if proxy is None or not isinstance(proxy.sessions, SessionManager):
        raise ReplayError("replay needs the proxy; it isn't running (no upstreams configured?)")
    return proxy.sessions


async def prepare_replay(services: Services, spec: SessionSpec) -> PreparedReplay:
    if spec.mode not in MODES:
        raise ReplayError(f"unknown mode {spec.mode!r} (use {', '.join(MODES)})")
    if spec.mode == "exact" and spec.patches:
        raise ReplayError("exact replay with patches is refused: a patched request can't match the tape; use auto_fork")
    if spec.mode == "fork" and (spec.fork_step is None or spec.fork_step < 1):
        raise ReplayError("fork needs fork_step (a step number from 1)")
    manager = session_manager(services)
    source = await services.store.reader.find_run(spec.source_run_id)
    if source is None:
        raise ReplayError(f"no run {spec.source_run_id}")
    if source.status != "complete":
        raise ReplayError(f"run {source.id} is {source.status}; only complete runs can be replayed")
    if not source.replayable:
        raise ReplayError(f"run {source.id} has no recorded exchanges, so it can't be replayed")
    profile = services.profiles.find(source.profile)
    if profile is None:
        raise ReplayError(f"run {source.id} has no known profile ({source.profile!r})")
    ctx = await load_run_context(services.store, source)
    rebuilt = False
    if ctx.entry_request is not None:
        request = StartRequest.from_envelope(ctx.entry_request)
    else:
        rebuilt_request = profile.rebuild_input(ctx)
        if rebuilt_request is None:
            raise ReplayError(f"run {source.id} has no entry request and profile {profile.name} can't rebuild one")
        request, rebuilt = rebuilt_request, True
    steps_by_exchange = {step.exchange_id: step for step in ctx.steps if step.exchange_id}
    tape = []
    for exchange in ctx.exchanges:
        step = steps_by_exchange.get(exchange.id)
        if step is None:
            continue
        row = exchange.row
        key = match_key(row.method, row.path, row.query, exchange.request_body, profile.normalisers(row.upstream))
        tape.append(TapeEntry(exchange, step.idx, step.node, key))
    if spec.fork_step is not None and spec.fork_step > len(ctx.steps):
        raise ReplayError(f"run {source.id} has {len(ctx.steps)} steps; can't fork from step {spec.fork_step}")
    session = ReplaySession(
        new_id(),
        new_trace_id(),
        spec,
        profile,
        tape,
        ttl_seconds=services.settings.proxy.session_ttl_seconds,
    )
    session.state["rebuilt"] = rebuilt

    async def op(db: AsyncSession) -> None:
        db.add(
            Session(
                id=session.id,
                trace_id=session.trace_id,
                source_run_id=source.id,
                mode=spec.mode,
                fork_step=spec.fork_step,
                overrides=spec.overrides(),
                status="active",
                created_ms=now_ms(),
            )
        )

    await services.store.write(op)
    manager.register(session)
    try:
        await profile.prepare(session, services)
    except Exception as exc:
        manager.end(session.trace_id)
        await finish_session(services, session, "failed", {"error": f"prepare failed: {exc}"})
        raise ReplayError(f"preparing the replay failed: {exc}") from exc
    return PreparedReplay(session, profile.replay_request(request, session), rebuilt, source)


async def run_replay(
    services: Services, prepared: PreparedReplay, *, timeout_seconds: float | None = None
) -> dict[str, Any]:
    session, source = prepared.session, prepared.source
    manager = session_manager(services)
    profile_name = source.profile or ""
    started = await create_run(
        services,
        profile_name,
        prepared.request,
        source="replay",
        trace_id=session.trace_id,
        session_id=session.id,
        replay_of=source.id,
        tags={"replay_mode": session.spec.mode},
    )
    completion = services.assembler.completion(session.trace_id)
    services.bus.publish("session.updated", session_id=session.id, status="running", run_id=started.run_id)
    report: dict[str, Any]
    try:
        sent = await send_entry_request(
            services, started, prepared.request, extra_headers={"baggage": f"{SESSION_BAGGAGE}={session.id}"}
        )
        limit = timeout_seconds or services.settings.proxy.session_ttl_seconds
        async with asyncio.timeout(limit):
            await completion
        report = await build_report(
            services, session, source.id, started.run_id, rebuilt=prepared.rebuilt, entry_error=sent.error
        )
        status = "complete"
    except TimeoutError:
        report = {"error": "the replay run did not complete in time", "replay_run": started.run_id}
        status = "failed"
    except Exception as exc:
        log.exception("replay %s failed", session.id)
        report = {"error": f"{type(exc).__name__}: {exc}", "replay_run": started.run_id}
        status = "failed"
    finally:
        manager.end(session.trace_id)
    await finish_session(services, session, status, report)
    return report


async def finish_session(services: Services, session: ReplaySession, status: str, report: dict[str, Any]) -> None:
    async def op(db: AsyncSession) -> None:
        await db.execute(
            update(Session)
            .where(Session.id == session.id)
            .values(status=status, result=json.loads(json.dumps(report, default=str)), ended_ms=now_ms())
        )

    await services.store.write(op)
    services.bus.publish("session.updated", session_id=session.id, status=status)


async def start_replay(
    services: Services, spec: SessionSpec, *, wait: bool = False
) -> tuple[PreparedReplay, dict[str, Any] | None]:
    prepared = await prepare_replay(services, spec)
    if wait:
        return prepared, await run_replay(services, prepared)
    services.spawn(run_replay(services, prepared), name=f"replay-{prepared.session.id}")
    return prepared, None
