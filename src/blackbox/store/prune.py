"""`blackbox db prune`: delete old runs (except baseline and labelled ones), then every blob nothing references."""

from dataclasses import dataclass

from sqlalchemy import delete, exists, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.store import Store
from blackbox.store.models import Exchange, Label, RecordedValue, Run, Session, Span, SuiteCase

# Every place a blob hash can be referenced from. Keep in step with the models.
_BLOB_REFERENCES = """
    SELECT entry_request_blob FROM runs WHERE entry_request_blob IS NOT NULL
    UNION SELECT output_blob FROM runs WHERE output_blob IS NOT NULL
    UNION SELECT request_blob FROM exchanges WHERE request_blob IS NOT NULL
    UNION SELECT response_blob FROM exchanges WHERE response_blob IS NOT NULL
    {extra}
    UNION SELECT prompt_blob FROM judge_calls WHERE prompt_blob IS NOT NULL
    UNION SELECT response_blob FROM judge_calls WHERE response_blob IS NOT NULL
    UNION SELECT report_blob FROM suite_runs WHERE report_blob IS NOT NULL
    UNION SELECT json_each.value FROM spans, json_each(spans.blob_refs)
"""


@dataclass
class PruneResult:
    runs: int
    blobs: int


async def _has_column(session: AsyncSession, table: str, column: str) -> bool:
    rows = (await session.execute(text(f"PRAGMA table_info({table})"))).all()
    return any(row[1] == column for row in rows)


async def prune(store: Store, *, cutoff_ms: int) -> PruneResult:
    async def op(session: AsyncSession) -> PruneResult:
        labelled = exists().where(Label.run_id == Run.id)
        in_suite = exists().where(SuiteCase.baseline_run_id == Run.id)
        old = (
            await session.execute(
                select(Run.id, Run.trace_id).where(
                    func.coalesce(Run.started_ms, Run.updated_ms) < cutoff_ms,
                    ~labelled,
                    ~in_suite,
                    func.json_extract(Run.tags, "$.baseline").is_(None),
                )
            )
        ).all()
        run_ids = [row[0] for row in old]
        trace_ids = [row[1] for row in old]
        for chunk_start in range(0, len(run_ids), 500):
            ids = run_ids[chunk_start : chunk_start + 500]
            traces = trace_ids[chunk_start : chunk_start + 500]
            await session.execute(delete(Span).where(Span.trace_id.in_(traces)))
            await session.execute(delete(Exchange).where(Exchange.trace_id.in_(traces)))
            await session.execute(delete(RecordedValue).where(RecordedValue.trace_id.in_(traces)))
            await session.execute(delete(Session).where(Session.trace_id.in_(traces)))
            await session.execute(delete(Run).where(Run.id.in_(ids)))  # steps, scores, failures cascade
        extra = ""
        if await _has_column(session, "exchanges", "sent_request_blob"):
            extra = "UNION SELECT sent_request_blob FROM exchanges WHERE sent_request_blob IS NOT NULL"
        result = await session.execute(
            text(f"DELETE FROM blobs WHERE sha256 NOT IN ({_BLOB_REFERENCES.format(extra=extra)})")
        )
        return PruneResult(runs=len(run_ids), blobs=int(getattr(result, "rowcount", 0) or 0))

    return await store.write(op)
