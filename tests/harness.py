"""Start a whole BlackBox in-process for integration tests: real sockets on free ports, a migrated database copy."""

import shutil
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from blackbox.config import (
    ClustersConfig,
    OllamaConfig,
    ProxyConfig,
    RunsConfig,
    ServerConfig,
    Settings,
    StoreConfig,
    UpstreamConfig,
)
from blackbox.profiles import ProfileRegistry, default_registry
from blackbox.profiles.base import Profile
from blackbox.server import Running, start_blackbox


def make_settings(
    db: Path,
    *,
    upstreams: Sequence[UpstreamConfig] = (),
    quiet: float = 0.3,
    keep_unmatched: bool = False,
    **extra: Any,
) -> Settings:
    return Settings(
        server=ServerConfig(port=0),
        store=StoreConfig(path=db),
        runs=RunsConfig(quiet_seconds=quiet, tick_seconds=0.05, orphan_seconds=3, keep_unmatched=keep_unmatched),
        proxy=ProxyConfig(upstreams=list(upstreams)),
        **{
            "clusters": ClustersConfig(embedder="hashing"),
            "ollama": OllamaConfig(base_url="http://127.0.0.1:9"),
            **extra,
        },
    )


def registry_with(*profiles: Profile) -> ProfileRegistry:
    registry = default_registry()
    for profile in profiles:
        registry.register(profile)
    return registry


@asynccontextmanager
async def running_blackbox(
    tmp_path: Path,
    migrated_db: Path,
    *,
    profiles: Sequence[Profile] = (),
    upstreams: Sequence[UpstreamConfig] = (),
    quiet: float = 0.3,
    keep_unmatched: bool = False,
    **extra: Any,
) -> AsyncIterator[Running]:
    db = tmp_path / "blackbox.db"
    if not db.exists():
        shutil.copy(migrated_db, db)
    settings = make_settings(db, upstreams=upstreams, quiet=quiet, keep_unmatched=keep_unmatched, **extra)
    running = await start_blackbox(settings, profiles=registry_with(*profiles), migrate=False)
    try:
        yield running
    finally:
        await running.stop()
