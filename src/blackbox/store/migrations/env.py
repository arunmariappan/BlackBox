"""Alembic environment: migrations run synchronously against the SQLite file, with batch mode for ALTERs."""

from pathlib import Path

from alembic import context

from blackbox.store.db import make_sync_engine
from blackbox.store.models import Base

config = context.config
target_metadata = Base.metadata


def run_migrations_online() -> None:
    path = config.attributes.get("db_path")
    if path is None:
        url = config.get_main_option("sqlalchemy.url") or ""
        path = Path(url.removeprefix("sqlite:///"))
    engine = make_sync_engine(Path(path))
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
            with context.begin_transaction():
                context.run_migrations()
            connection.commit()
    finally:
        engine.dispose()


if context.is_offline_mode():
    raise SystemExit("offline migrations are not supported; run `blackbox db upgrade`")
run_migrations_online()
