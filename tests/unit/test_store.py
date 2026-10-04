import asyncio
import shutil
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.store import Store, db
from blackbox.store.bundles import export_run, import_bundle, run_bundle_data
from blackbox.store.models import Base, Blob, Label, Run, Score
from blackbox.store.prune import prune
from blackbox.util import new_id, now_ms
from tests.factories import add_full_run, add_run, trace_id


def test_migrations_match_models(migrated_db: Path) -> None:
    engine = db.make_sync_engine(migrated_db)
    with engine.connect() as connection:
        diffs = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    engine.dispose()
    assert diffs == []


async def test_pragmas(store: Store) -> None:
    async with store.read() as s:
        assert (await s.execute(select(func.count()).select_from(Run))).scalar_one() == 0
        mode = (await s.connection()).exec_driver_sql
        assert (await mode("PRAGMA journal_mode")).scalar_one() == "wal"
        assert (await mode("PRAGMA foreign_keys")).scalar_one() == 1
        assert (await mode("PRAGMA busy_timeout")).scalar_one() == 5000


async def test_thousand_concurrent_writes_all_land(store: Store) -> None:
    async def worker(n: int) -> None:
        for i in range(50):
            run = Run(id=new_id(), trace_id=f"{n:016x}{i:016x}", updated_ms=now_ms())

            async def op(session: AsyncSession, run: Run = run) -> None:
                session.add(run)

            await store.write(op)

    async def reader() -> None:
        for _ in range(20):
            await store.reader.runs(limit=5)
            await asyncio.sleep(0)

    await asyncio.gather(*(worker(n) for n in range(20)), reader(), reader())
    async with store.read() as s:
        assert (await s.execute(select(func.count()).select_from(Run))).scalar_one() == 1000
    assert store.writer.batches < 1000  # operations were batched


async def test_failing_operation_fails_only_its_caller(store: Store) -> None:
    existing = await add_run(store)

    def make(i: int) -> object:
        async def op(session: AsyncSession) -> int:
            if i == 4:
                raise ValueError("planted")
            tid = existing.trace_id if i == 7 else trace_id()  # a unique-constraint violation
            session.add(Run(id=new_id(), trace_id=tid, updated_ms=now_ms()))
            await session.flush()
            return i

        return op

    results = await asyncio.gather(*(store.write(make(i)) for i in range(10)), return_exceptions=True)  # type: ignore[arg-type]
    assert isinstance(results[4], ValueError)
    assert isinstance(results[7], IntegrityError)
    assert [r for i, r in enumerate(results) if i not in (4, 7)] == [0, 1, 2, 3, 5, 6, 8, 9]
    async with store.read() as s:
        assert (await s.execute(select(func.count()).select_from(Run))).scalar_one() == 9


async def test_blob_stored_once(store: Store) -> None:
    data = b"x" * 10_000
    first = await store.blobs.put(data, "text/plain")
    second = await store.blobs.put(data, "text/plain")
    assert first == second
    async with store.read() as s:
        assert (await s.execute(select(func.count()).select_from(Blob))).scalar_one() == 1
        size = (await s.execute(select(func.length(Blob.data)))).scalar_one()
    assert size < 1000  # compressed
    assert await store.blobs.get(first) == data
    with pytest.raises(KeyError):
        await store.blobs.get("0" * 64)


async def test_export_import_round_trip(store: Store, tmp_path: Path, db_path: Path, migrated_db: Path) -> None:
    run = await add_full_run(store)
    bundle = tmp_path / "bundle"
    await export_run(store, run.id, bundle)
    assert (bundle / "run.json").read_text().startswith('{\n  "blobs"')
    assert len(list((bundle / "blobs").glob("*.zst"))) == 3
    original = await run_bundle_data(store, run.id)

    other_path = tmp_path / "other.db"
    await asyncio.to_thread(shutil.copy, migrated_db, other_path)
    other = await Store.open(other_path, migrate=False)
    try:
        imported_id = await import_bundle(other, bundle)
        again = await import_bundle(other, bundle)  # idempotent
        assert imported_id == again == run.id
        assert await run_bundle_data(other, run.id) == original
        async with other.read() as s:
            assert (await s.execute(select(func.count()).select_from(Run))).scalar_one() == 1
            assert (await s.execute(select(func.count()).select_from(Score))).scalar_one() == 1
    finally:
        await other.close()


async def test_prune_keeps_labelled_and_baseline_runs(store: Store) -> None:
    old = now_ms() - 40 * 86_400_000
    doomed = await add_full_run(store, started_ms=old)
    labelled = await add_full_run(store, started_ms=old)
    baseline = await add_run(store, started_ms=old, tags={"baseline": "opsdesk-core/v1"})
    recent = await add_full_run(store)
    orphan = await store.blobs.put(b"nobody points at me")

    async def label(session: AsyncSession) -> None:
        session.add(Label(id=new_id(), run_id=labelled.id, question="faithful", value="pass", created_ms=now_ms()))

    await store.write(label)
    result = await prune(store, cutoff_ms=now_ms() - 30 * 86_400_000)
    assert result.runs == 1
    remaining = {r.id for r in await store.reader.runs(limit=10)}
    assert remaining == {labelled.id, baseline.id, recent.id}
    assert doomed.id not in remaining
    with pytest.raises(KeyError):
        await store.blobs.get(orphan)
    # Blobs still referenced by the kept runs survive, including a span's blob ref.
    data = await run_bundle_data(store, recent.id)
    for sha in data["blobs"]:
        assert await store.blobs.get(sha)
