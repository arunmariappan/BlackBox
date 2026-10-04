"""markers and live_patches: timeline notes for alerts and charts, and patches applied to live traffic

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-04 23:10:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "markers",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("profile", sa.String(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("created_ms", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_markers")),
    )
    op.create_index(op.f("ix_markers_profile_created_ms"), "markers", ["profile", "created_ms"])
    op.create_table(
        "live_patches",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("hits", sa.BigInteger(), nullable=False),
        sa.Column("created_ms", sa.BigInteger(), nullable=False),
        sa.Column("expires_ms", sa.BigInteger(), nullable=False),
        sa.Column("ended_ms", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_live_patches")),
    )
    op.create_index(op.f("ix_live_patches_status"), "live_patches", ["status"])


def downgrade() -> None:
    op.drop_index(op.f("ix_live_patches_status"), table_name="live_patches")
    op.drop_table("live_patches")
    op.drop_index(op.f("ix_markers_profile_created_ms"), table_name="markers")
    op.drop_table("markers")
