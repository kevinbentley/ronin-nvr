"""Add storage tier columns to recordings

Revision ID: 20260209_add_storage_tiers
Revises: 20260120_add_object_event_clips
Create Date: 2026-02-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260209_add_storage_tiers"
down_revision: Union[str, None] = "20260120_add_object_event_clips"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add storage_tier column with default 'hot' for existing recordings
    op.add_column(
        "recordings",
        sa.Column(
            "storage_tier",
            sa.String(20),
            nullable=False,
            server_default="hot",
        ),
    )

    # Add storage_path column for warm/cold storage paths
    # This is null for hot storage (uses file_path instead)
    op.add_column(
        "recordings",
        sa.Column("storage_path", sa.String(1024), nullable=True),
    )

    # Add migrated_at timestamp
    op.add_column(
        "recordings",
        sa.Column("migrated_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )

    # Create index on storage_tier for efficient filtering
    op.create_index(
        "ix_recordings_storage_tier",
        "recordings",
        ["storage_tier"],
    )


def downgrade() -> None:
    op.drop_index("ix_recordings_storage_tier", table_name="recordings")
    op.drop_column("recordings", "migrated_at")
    op.drop_column("recordings", "storage_path")
    op.drop_column("recordings", "storage_tier")
