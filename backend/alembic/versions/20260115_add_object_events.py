"""Add object_events table for FSM state transitions.

Revision ID: 20260115_add_object_events
Revises: 20260115_add_class_thresholds
Create Date: 2026-01-15
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260115_add_object_events"
down_revision = "20260115_add_class_thresholds"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "object_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_type", sa.String(32), nullable=False, index=True),
        sa.Column("class_name", sa.String(64), nullable=False, index=True),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("old_state", sa.String(32), nullable=True),
        sa.Column("new_state", sa.String(32), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.Column("snapshot_path", sa.String(512), nullable=True),
        sa.Column(
            "camera_id",
            sa.Integer(),
            sa.ForeignKey("cameras.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "event_time",
            sa.DateTime(timezone=True),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # Create composite index for common query patterns
    op.create_index(
        "ix_object_events_camera_time",
        "object_events",
        ["camera_id", "event_time"],
    )


def downgrade() -> None:
    op.drop_index("ix_object_events_camera_time", "object_events")
    op.drop_table("object_events")
