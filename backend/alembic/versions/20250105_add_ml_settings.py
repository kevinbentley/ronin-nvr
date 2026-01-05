"""Add ml_settings table for runtime configuration.

Revision ID: 20250105_settings
Revises: 20241230_live
Create Date: 2025-01-05

This migration adds the ml_settings table which stores runtime-configurable
ML settings. This table uses a singleton pattern (always has id=1) and is
polled by workers to pick up configuration changes without restarts.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TIMESTAMP


# revision identifiers, used by Alembic.
revision = '20250105_settings'
down_revision = '20241230_live'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ml_settings table with default row."""
    # Create the table
    op.create_table(
        'ml_settings',
        sa.Column('id', sa.Integer(), primary_key=True),
        # Live detection settings
        sa.Column('live_detection_enabled', sa.Boolean(), nullable=False, default=True),
        sa.Column('live_detection_fps', sa.Float(), nullable=False, default=1.0),
        sa.Column('live_detection_cooldown', sa.Float(), nullable=False, default=30.0),
        sa.Column('live_detection_confidence', sa.Float(), nullable=False, default=0.6),
        sa.Column('live_detection_classes', sa.String(), nullable=False, default='person,car,truck'),
        # Historical processing settings
        sa.Column('historical_confidence', sa.Float(), nullable=False, default=0.5),
        sa.Column('historical_classes', sa.String(), nullable=False,
                  default='person,car,truck,bus,motorcycle,bicycle,dog,cat'),
        # Timestamps
        sa.Column('updated_at', TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )

    # Insert the default singleton row
    op.execute("""
        INSERT INTO ml_settings (
            id,
            live_detection_enabled,
            live_detection_fps,
            live_detection_cooldown,
            live_detection_confidence,
            live_detection_classes,
            historical_confidence,
            historical_classes,
            updated_at
        ) VALUES (
            1,
            true,
            1.0,
            30.0,
            0.6,
            'person,car,truck',
            0.5,
            'person,car,truck,bus,motorcycle,bicycle,dog,cat',
            NOW()
        )
    """)


def downgrade() -> None:
    """Drop ml_settings table."""
    op.drop_table('ml_settings')
