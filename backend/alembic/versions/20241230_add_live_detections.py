"""Add live_detections table for real-time detection from streams.

Revision ID: 20241230_live
Revises: 20241230_utc
Create Date: 2024-12-30

This migration adds the live_detections table for storing real-time
object detections from live camera streams. Unlike the detections table
which is tied to recordings, live_detections are standalone events
with their own snapshot images and optional LLM descriptions.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TIMESTAMP


# revision identifiers, used by Alembic.
revision = '20241230_live'
down_revision = '20241230_utc'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create live_detections table."""
    op.create_table(
        'live_detections',
        sa.Column('id', sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column(
            'camera_id',
            sa.Integer(),
            sa.ForeignKey('cameras.id', ondelete='CASCADE'),
            nullable=False,
        ),
        sa.Column('class_name', sa.String(100), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('bbox_x', sa.Float(), nullable=False),
        sa.Column('bbox_y', sa.Float(), nullable=False),
        sa.Column('bbox_width', sa.Float(), nullable=False),
        sa.Column('bbox_height', sa.Float(), nullable=False),
        sa.Column('model_name', sa.String(100), nullable=False),
        sa.Column(
            'detected_at',
            TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column('notified', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('snapshot_path', sa.String(500), nullable=True),
        sa.Column('llm_description', sa.Text(), nullable=True),
    )

    # Index for querying detections by camera and time (most common query)
    op.create_index(
        'ix_live_detections_camera_detected',
        'live_detections',
        ['camera_id', 'detected_at'],
    )

    # Index for filtering by class name
    op.create_index(
        'ix_live_detections_class_name',
        'live_detections',
        ['class_name'],
    )

    # Index for retention cleanup (delete old detections)
    op.create_index(
        'ix_live_detections_detected_at',
        'live_detections',
        ['detected_at'],
    )


def downgrade() -> None:
    """Drop live_detections table."""
    op.drop_index('ix_live_detections_detected_at', table_name='live_detections')
    op.drop_index('ix_live_detections_class_name', table_name='live_detections')
    op.drop_index('ix_live_detections_camera_detected', table_name='live_detections')
    op.drop_table('live_detections')
