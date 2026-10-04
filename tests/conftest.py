import shutil
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from blackbox.store import Store, db


@pytest.fixture(scope="session")
def migrated_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A database upgraded to the latest revision once per test session; tests copy it."""
    path = tmp_path_factory.mktemp("template") / "template.db"
    db.upgrade(path)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    return path


@pytest.fixture
def db_path(tmp_path: Path, migrated_db: Path) -> Path:
    path = tmp_path / "blackbox.db"
    shutil.copy(migrated_db, path)
    return path


@pytest.fixture
async def store(db_path: Path) -> AsyncIterator[Store]:
    s = await Store.open(db_path, migrate=False)
    try:
        yield s
    finally:
        await s.close()
