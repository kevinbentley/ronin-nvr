"""Add live detection support to detections table.

Revision ID: 20241230_live
Revises: 20241230_utc
Create Date: 2024-12-30

This migration enhances the existing detections table to support real-time
detection from live streams:
- Makes recording_id nullable (live detections correlate by timestamp later)
- Adds detected_at for actual detection timestamp
- Adds snapshot_path for event thumbnails
- Adds llm_description for future Vision LLM integration

This unified approach allows:
- Real-time alerts with 2-5 second latency
- Same detections visible in playback timeline
- Single retention policy
- Historical worker becomes optional
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
    """Add live detection columns to detections table."""
    # Make recording_id nullable for live detections
    # Live detections can correlate to recording by timestamp later
    op.alter_column(
        'detections',
        'recording_id',
        existing_type=sa.Integer(),
        nullable=True,
    )

    # Add detected_at - the actual time the detection occurred
    # This is different from created_at (when the row was inserted)
    # For live detection: detected_at = now
    # For historical: detected_at = recording start + timestamp_ms
    op.add_column(
        'detections',
        sa.Column(
            'detected_at',
            TIMESTAMP(timezone=True),
            nullable=True,
        ),
    )

    # Add snapshot_path for event thumbnails (JPG with bounding boxes)
    op.add_column(
        'detections',
        sa.Column('snapshot_path', sa.String(500), nullable=True),
    )

    # Add llm_description for future Vision LLM scene descriptions
    op.add_column(
        'detections',
        sa.Column('llm_description', sa.Text(), nullable=True),
    )

    # Index for querying by detected_at (useful for live detection timeline)
    op.create_index(
        'ix_detections_detected_at',
        'detections',
        ['detected_at'],
    )

    # Composite index for camera + detected_at (common query pattern)
    op.create_index(
        'ix_detections_camera_detected_at',
        'detections',
        ['camera_id', 'detected_at'],
    )


def downgrade() -> None:
    """Remove live detection columns from detections table."""
    op.drop_index('ix_detections_camera_detected_at', table_name='detections')
    op.drop_index('ix_detections_detected_at', table_name='detections')

    op.drop_column('detections', 'llm_description')
    op.drop_column('detections', 'snapshot_path')
    op.drop_column('detections', 'detected_at')

    # Restore recording_id as NOT NULL
    # Note: This will fail if there are rows with NULL recording_id
    op.alter_column(
        'detections',
        'recording_id',
        existing_type=sa.Integer(),
        nullable=False,
    )
