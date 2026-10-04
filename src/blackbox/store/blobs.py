"""Content-addressed blobs: zstd-compressed bytes keyed by the SHA-256 of the uncompressed bytes."""

from collections import OrderedDict
from compression import zstd
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from blackbox.store.models import Blob
from blackbox.store.writer import StoreWriter
from blackbox.util import sha256_hex

ZSTD_LEVEL = 3


@dataclass(frozen=True, slots=True)
class PreparedBlob:
    """A blob hashed and compressed outside the writer, ready to insert inside any write operation."""

    sha256: str
    size: int
    content_type: str | None
    data: bytes  # compressed

    @classmethod
    def of(cls, raw: bytes, content_type: str | None = None) -> PreparedBlob:
        return cls(sha256_hex(raw), len(raw), content_type, zstd.compress(raw, level=ZSTD_LEVEL))


def decompress(data: bytes) -> bytes:
    return zstd.decompress(data)


async def insert_blob(session: AsyncSession, blob: PreparedBlob) -> str:
    """Insert `blob` unless a blob with the same hash exists; returns its hash."""
    await session.execute(
        insert(Blob)
        .values(sha256=blob.sha256, size=blob.size, content_type=blob.content_type, data=blob.data)
        .on_conflict_do_nothing(index_elements=["sha256"])
    )
    return blob.sha256


class BlobStore:
    def __init__(self, writer: StoreWriter, reader_sessions: async_sessionmaker[AsyncSession], cache_size: int = 256):
        self._writer = writer
        self._reader_sessions = reader_sessions
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._cache_size = cache_size

    async def put(self, raw: bytes, content_type: str | None = None) -> str:
        prepared = PreparedBlob.of(raw, content_type)

        async def op(session: AsyncSession) -> str:
            return await insert_blob(session, prepared)

        return await self._writer.submit(op)

    async def get(self, sha256: str) -> bytes:
        cached = self._cache.get(sha256)
        if cached is not None:
            self._cache.move_to_end(sha256)
            return cached
        async with self._reader_sessions() as session:
            data = (await session.execute(select(Blob.data).where(Blob.sha256 == sha256))).scalar_one_or_none()
        if data is None:
            raise KeyError(f"no blob {sha256}")
        raw = decompress(data)
        if len(raw) <= 1_000_000:
            self._cache[sha256] = raw
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return raw

    async def get_optional(self, sha256: str | None) -> bytes | None:
        if sha256 is None:
            return None
        try:
            return await self.get(sha256)
        except KeyError:
            return None

    async def get_compressed(self, sha256: str) -> tuple[bytes, str | None]:
        async with self._reader_sessions() as session:
            row = (
                await session.execute(select(Blob.data, Blob.content_type).where(Blob.sha256 == sha256))
            ).one_or_none()
        if row is None:
            raise KeyError(f"no blob {sha256}")
        return row[0], row[1]
