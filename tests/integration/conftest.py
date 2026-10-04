from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from tests.opsdesk_stack import Ops, ops_stack


@pytest.fixture
async def ops(tmp_path: Path, migrated_db: Path) -> AsyncIterator[Ops]:
    async for stack in ops_stack(tmp_path, migrated_db):
        yield stack
