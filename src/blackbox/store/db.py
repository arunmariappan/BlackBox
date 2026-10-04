"""SQLite engines and migrations.

The database runs in WAL mode with `synchronous=NORMAL`, `foreign_keys=ON` and `busy_timeout=5000`. The writer engine
has exactly one connection; the reader engine has a small pool of connections that refuse writes (`query_only`).
"""

from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _set_pragmas(dbapi_connection: Any, *, readonly: bool) -> None:
    # Let SQLAlchemy, not the driver, decide when transactions begin, so SAVEPOINTs work (see the "begin" hook).
    dbapi_connection.isolation_level = None
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    if readonly:
        cursor.execute("PRAGMA query_only=ON")
    cursor.close()


def _install_hooks(engine: Engine, *, readonly: bool) -> None:
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection: Any, _record: Any) -> None:
        _set_pragmas(dbapi_connection, readonly=readonly)

    @event.listens_for(engine, "begin")
    def _on_begin(connection: Any) -> None:
        connection.exec_driver_sql("BEGIN IMMEDIATE" if not readonly else "BEGIN")


def make_async_engine(path: Path, *, readonly: bool, pool_size: int = 1) -> AsyncEngine:
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}", pool_size=pool_size, max_overflow=0)
    _install_hooks(engine.sync_engine, readonly=readonly)
    return engine


def make_sync_engine(path: Path) -> Engine:
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}")
    _install_hooks(engine, readonly=False)
    return engine


def alembic_config(path: Path) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    config.attributes["db_path"] = path
    return config


def upgrade(path: Path, revision: str = "head") -> None:
    """Create or upgrade the database at `path` to `revision` (synchronous; call it in a thread from async code)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    command.upgrade(alembic_config(path), revision)


def current_revision(path: Path) -> str | None:
    from alembic.runtime.migration import MigrationContext

    engine = make_sync_engine(path)
    try:
        with engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        engine.dispose()
