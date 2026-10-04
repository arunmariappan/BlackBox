"""BlackBox's store: one SQLite file, one writer task, read-only reader connections and content-addressed blobs."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from blackbox.store import db
from blackbox.store.blobs import BlobStore, PreparedBlob, insert_blob
from blackbox.store.reader import StoreReader
from blackbox.store.writer import StoreWriter, WriteOp

__all__ = ["BlobStore", "PreparedBlob", "Store", "StoreReader", "StoreWriter", "WriteOp", "insert_blob"]


class Store:
    def __init__(self, path: Path, write_engine: AsyncEngine, read_engine: AsyncEngine) -> None:
        self.path = path
        self._write_engine = write_engine
        self._read_engine = read_engine
        self.writer = StoreWriter(write_engine)
        self.read_sessions = async_sessionmaker(read_engine, expire_on_commit=False)
        self.reader = StoreReader(self.read_sessions)
        self.blobs = BlobStore(self.writer, self.read_sessions)

    @classmethod
    async def open(cls, path: Path, *, migrate: bool = True, reader_pool: int = 4) -> Store:
        """Open (and by default create or upgrade) the database at `path`, and start its writer."""
        if migrate:
            await asyncio.to_thread(db.upgrade, path)
        store = cls(
            path,
            db.make_async_engine(path, readonly=False, pool_size=1),
            db.make_async_engine(path, readonly=True, pool_size=reader_pool),
        )
        await store.writer.start()
        return store

    async def close(self) -> None:
        await self.writer.stop()
        await self._write_engine.dispose()
        await self._read_engine.dispose()

    async def write[T](self, fn: WriteOp[T]) -> T:
        return await self.writer.submit(fn)

    @asynccontextmanager
    async def read(self) -> AsyncIterator[AsyncSession]:
        async with self.read_sessions() as session:
            yield session
