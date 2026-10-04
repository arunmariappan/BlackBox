"""Read helpers. Reads use their own pool of read-only connections; WAL lets them run during writes."""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from blackbox.store.models import Exchange, Label, RecordedValue, Run, Score, Session, Span, Step


class StoreReader:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def run(self, run_id: str) -> Run | None:
        async with self.sessions() as s:
            return await s.get(Run, run_id)

    async def run_by_trace(self, trace_id: str) -> Run | None:
        async with self.sessions() as s:
            return (await s.execute(select(Run).where(Run.trace_id == trace_id))).scalar_one_or_none()

    async def find_run(self, ref: str) -> Run | None:
        """A run by id, trace id, or a unique prefix of its id."""
        run = await self.run(ref) or await self.run_by_trace(ref)
        if run is not None or len(ref) < 6:
            return run
        async with self.sessions() as s:
            rows = (await s.execute(select(Run).where(Run.id.startswith(ref.upper())).limit(2))).scalars().all()
        return rows[0] if len(rows) == 1 else None

    async def runs(
        self,
        *,
        profile: str | None = None,
        ending: str | None = None,
        source: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Run]:
        query = select(Run).order_by(Run.started_ms.desc().nulls_last(), Run.id.desc())
        if profile:
            query = query.where(Run.profile == profile)
        if ending:
            query = query.where(Run.ending == ending)
        if source:
            query = query.where(Run.source == source)
        if status:
            query = query.where(Run.status == status)
        async with self.sessions() as s:
            return (await s.execute(query.limit(limit).offset(offset))).scalars().all()

    async def spans(self, trace_id: str) -> Sequence[Span]:
        async with self.sessions() as s:
            query = select(Span).where(Span.trace_id == trace_id).order_by(Span.start_ns, Span.span_id)
            return (await s.execute(query)).scalars().all()

    async def exchanges(self, trace_id: str) -> Sequence[Exchange]:
        async with self.sessions() as s:
            query = select(Exchange).where(Exchange.trace_id == trace_id).order_by(Exchange.seq, Exchange.id)
            return (await s.execute(query)).scalars().all()

    async def exchange(self, exchange_id: str) -> Exchange | None:
        async with self.sessions() as s:
            return await s.get(Exchange, exchange_id)

    async def steps(self, run_id: str) -> Sequence[Step]:
        async with self.sessions() as s:
            return (await s.execute(select(Step).where(Step.run_id == run_id).order_by(Step.idx))).scalars().all()

    async def scores(self, run_id: str) -> Sequence[Score]:
        async with self.sessions() as s:
            query = select(Score).where(Score.run_id == run_id).order_by(Score.kind, Score.name, Score.created_ms)
            return (await s.execute(query)).scalars().all()

    async def labels(self, run_id: str) -> Sequence[Label]:
        async with self.sessions() as s:
            query = select(Label).where(Label.run_id == run_id).order_by(Label.created_ms)
            return (await s.execute(query)).scalars().all()

    async def recorded_values(self, trace_id: str) -> Sequence[RecordedValue]:
        async with self.sessions() as s:
            query = select(RecordedValue).where(RecordedValue.trace_id == trace_id).order_by(RecordedValue.seq)
            return (await s.execute(query)).scalars().all()

    async def session(self, session_id: str) -> Session | None:
        async with self.sessions() as s:
            return await s.get(Session, session_id)
