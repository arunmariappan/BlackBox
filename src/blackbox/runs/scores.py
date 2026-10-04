"""Writing scores: one row per (run, kind, name, version); writing again replaces it."""

from typing import Any

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.store import Store
from blackbox.store.models import Score
from blackbox.util import new_id, now_ms


async def write_score(
    store: Store,
    run_id: str,
    *,
    kind: str,
    name: str,
    version: str,
    value: float | None = None,
    label: str | None = None,
    rationale: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    async def op(session: AsyncSession) -> None:
        await session.execute(
            delete(Score).where(
                Score.run_id == run_id, Score.kind == kind, Score.name == name, Score.version == version
            )
        )
        session.add(
            Score(
                id=new_id(),
                run_id=run_id,
                kind=kind,
                name=name,
                version=version,
                value=value,
                label=label,
                rationale=rationale,
                details=details or {},
                created_ms=now_ms(),
            )
        )

    await store.write(op)
