"""exchanges.sent_request_blob: the body actually sent when a model override or patch changed it

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-04 20:10:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("exchanges", schema=None) as batch_op:
        batch_op.add_column(sa.Column("sent_request_blob", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("exchanges", schema=None) as batch_op:
        batch_op.drop_column("sent_request_blob")
