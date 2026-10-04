"""cluster_spaces: the feature space each profile's failure clusters were made in

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-04 20:55:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "cluster_spaces",
        sa.Column("profile", sa.String(), nullable=False),
        sa.Column("vocabulary", sa.JSON(), nullable=False),
        sa.Column("weight", sa.Double(), nullable=False),
        sa.Column("embedder", sa.String(), nullable=False),
        sa.Column("dimensions", sa.BigInteger(), nullable=False),
        sa.Column("clustered_ms", sa.BigInteger(), nullable=False),
        sa.Column("stats", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("profile", name=op.f("pk_cluster_spaces")),
    )


def downgrade() -> None:
    op.drop_table("cluster_spaces")
