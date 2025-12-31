"""Convert datetime columns to timestamp with timezone.

Revision ID: 20241230_utc
Revises: bfc3f1b7eb27
Create Date: 2024-12-30

This migration converts all TIMESTAMP columns to TIMESTAMP WITH TIME ZONE
for proper UTC timezone handling. PostgreSQL stores timestamps internally
as UTC, so existing data is preserved correctly.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TIMESTAMP


# revision identifiers, used by Alembic.
revision = '20241230_utc'
down_revision = 'bfc3f1b7eb27'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Convert TIMESTAMP columns to TIMESTAMP WITH TIME ZONE."""
    # cameras table
    op.alter_column('cameras', 'last_seen',
                    type_=TIMESTAMP(timezone=True),
                    existing_type=sa.DateTime())
    op.alter_column('cameras', 'created_at',
                    type_=TIMESTAMP(timezone=True),
                    existing_type=sa.DateTime())
    op.alter_column('cameras', 'updated_at',
                    type_=TIMESTAMP(timezone=True),
                    existing_type=sa.DateTime())

    # recordings table
    op.alter_column('recordings', 'start_time',
                    type_=TIMESTAMP(timezone=True),
                    existing_type=sa.DateTime())
    op.alter_column('recordings', 'end_time',
                    type_=TIMESTAMP(timezone=True),
                    existing_type=sa.DateTime())
    op.alter_column('recordings', 'created_at',
                    type_=TIMESTAMP(timezone=True),
                    existing_type=sa.DateTime())

    # users table
    op.alter_column('users', 'created_at',
                    type_=TIMESTAMP(timezone=True),
                    existing_type=sa.DateTime())

    # detections table
    op.alter_column('detections', 'created_at',
                    type_=TIMESTAMP(timezone=True),
                    existing_type=sa.DateTime())

    # ml_jobs table
    op.alter_column('ml_jobs', 'started_at',
                    type_=TIMESTAMP(timezone=True),
                    existing_type=sa.DateTime())
    op.alter_column('ml_jobs', 'completed_at',
                    type_=TIMESTAMP(timezone=True),
                    existing_type=sa.DateTime())
    op.alter_column('ml_jobs', 'last_heartbeat',
                    type_=TIMESTAMP(timezone=True),
                    existing_type=sa.DateTime())
    op.alter_column('ml_jobs', 'created_at',
                    type_=TIMESTAMP(timezone=True),
                    existing_type=sa.DateTime())

    # ml_models table
    op.alter_column('ml_models', 'created_at',
                    type_=TIMESTAMP(timezone=True),
                    existing_type=sa.DateTime())
    op.alter_column('ml_models', 'updated_at',
                    type_=TIMESTAMP(timezone=True),
                    existing_type=sa.DateTime())


def downgrade() -> None:
    """Revert to TIMESTAMP without timezone."""
    # cameras table
    op.alter_column('cameras', 'last_seen',
                    type_=sa.DateTime(),
                    existing_type=TIMESTAMP(timezone=True))
    op.alter_column('cameras', 'created_at',
                    type_=sa.DateTime(),
                    existing_type=TIMESTAMP(timezone=True))
    op.alter_column('cameras', 'updated_at',
                    type_=sa.DateTime(),
                    existing_type=TIMESTAMP(timezone=True))

    # recordings table
    op.alter_column('recordings', 'start_time',
                    type_=sa.DateTime(),
                    existing_type=TIMESTAMP(timezone=True))
    op.alter_column('recordings', 'end_time',
                    type_=sa.DateTime(),
                    existing_type=TIMESTAMP(timezone=True))
    op.alter_column('recordings', 'created_at',
                    type_=sa.DateTime(),
                    existing_type=TIMESTAMP(timezone=True))

    # users table
    op.alter_column('users', 'created_at',
                    type_=sa.DateTime(),
                    existing_type=TIMESTAMP(timezone=True))

    # detections table
    op.alter_column('detections', 'created_at',
                    type_=sa.DateTime(),
                    existing_type=TIMESTAMP(timezone=True))

    # ml_jobs table
    op.alter_column('ml_jobs', 'started_at',
                    type_=sa.DateTime(),
                    existing_type=TIMESTAMP(timezone=True))
    op.alter_column('ml_jobs', 'completed_at',
                    type_=sa.DateTime(),
                    existing_type=TIMESTAMP(timezone=True))
    op.alter_column('ml_jobs', 'last_heartbeat',
                    type_=sa.DateTime(),
                    existing_type=TIMESTAMP(timezone=True))
    op.alter_column('ml_jobs', 'created_at',
                    type_=sa.DateTime(),
                    existing_type=TIMESTAMP(timezone=True))

    # ml_models table
    op.alter_column('ml_models', 'created_at',
                    type_=sa.DateTime(),
                    existing_type=TIMESTAMP(timezone=True))
    op.alter_column('ml_models', 'updated_at',
                    type_=sa.DateTime(),
                    existing_type=TIMESTAMP(timezone=True))
