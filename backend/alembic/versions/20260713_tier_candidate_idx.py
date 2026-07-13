"""Add composite index for tier migration candidate lookups

Revision ID: 20260713_tier_candidate_idx
Revises: 20260209_add_storage_tiers
Create Date: 2026-07-13

The tier migration monitor repeatedly runs
``WHERE storage_tier = ? AND status = 'completed' ORDER BY start_time``.
A composite index lets that resolve without scanning/sorting the whole
tier, which matters once a tier holds tens of thousands of rows.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260713_tier_candidate_idx"
down_revision: Union[str, None] = "20260209_add_storage_tiers"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_recordings_tier_status_start",
        "recordings",
        ["storage_tier", "status", "start_time"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_recordings_tier_status_start",
        table_name="recordings",
    )
