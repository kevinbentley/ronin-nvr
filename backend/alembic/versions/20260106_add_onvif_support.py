"""Add ONVIF protocol support.

Revision ID: 20260106_onvif
Revises: a15875f2b4cd
Create Date: 2026-01-06

This migration adds:
1. ONVIF fields to cameras table for protocol configuration
2. event_source field to detections table for unified timeline
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision = "20260106_onvif"
down_revision = "a15875f2b4cd"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add ONVIF support columns."""
    # Camera ONVIF fields
    op.add_column(
        "cameras",
        sa.Column("onvif_port", sa.Integer(), nullable=True),
    )
    op.add_column(
        "cameras",
        sa.Column(
            "onvif_enabled",
            sa.Boolean(),
            server_default="false",
            nullable=False,
        ),
    )
    op.add_column(
        "cameras",
        sa.Column("onvif_profile_token", sa.String(100), nullable=True),
    )
    op.add_column(
        "cameras",
        sa.Column("onvif_device_info", JSONB, nullable=True),
    )
    op.add_column(
        "cameras",
        sa.Column(
            "onvif_events_enabled",
            sa.Boolean(),
            server_default="false",
            nullable=False,
        ),
    )

    # Detection event_source field for distinguishing ML vs camera events
    op.add_column(
        "detections",
        sa.Column(
            "event_source",
            sa.String(20),
            server_default="ml",
            nullable=False,
        ),
    )

    # Index for event_source queries
    op.create_index(
        "ix_detections_event_source",
        "detections",
        ["event_source"],
    )


def downgrade() -> None:
    """Remove ONVIF support columns."""
    # Remove detection event_source
    op.drop_index("ix_detections_event_source", table_name="detections")
    op.drop_column("detections", "event_source")

    # Remove camera ONVIF fields
    op.drop_column("cameras", "onvif_events_enabled")
    op.drop_column("cameras", "onvif_device_info")
    op.drop_column("cameras", "onvif_profile_token")
    op.drop_column("cameras", "onvif_enabled")
    op.drop_column("cameras", "onvif_port")
