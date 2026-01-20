"""Add video clip columns to object_events

Revision ID: 20260120_add_object_event_clips
Revises: 8b2e9f4c3a1d
Create Date: 2026-01-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20260120_add_object_event_clips'
down_revision: Union[str, None] = '8b2e9f4c3a1d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add video_clip_path column for storing the relative path to extracted clips
    op.add_column(
        "object_events",
        sa.Column("video_clip_path", sa.String(500), nullable=True),
    )
    # Add video_clip_status column for tracking extraction status
    # Values: pending, extracting, ready, failed
    op.add_column(
        "object_events",
        sa.Column(
            "video_clip_status",
            sa.String(20),
            nullable=False,
            server_default="pending",
        ),
    )


def downgrade() -> None:
    op.drop_column("object_events", "video_clip_status")
    op.drop_column("object_events", "video_clip_path")
