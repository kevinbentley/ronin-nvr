"""add unique constraint on ml_jobs recording_id and model_name

Revision ID: a15875f2b4cd
Revises: 20250105_settings
Create Date: 2026-01-05 17:24:05.763066

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a15875f2b4cd'
down_revision: Union[str, None] = '20250105_settings'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # First delete any duplicate jobs (keep the one with lowest id)
    op.execute("""
        DELETE FROM ml_jobs a USING ml_jobs b
        WHERE a.id > b.id
        AND a.recording_id = b.recording_id
        AND a.model_name = b.model_name
    """)
    # Add unique constraint
    op.create_unique_constraint(
        'ml_jobs_recording_model_unique',
        'ml_jobs',
        ['recording_id', 'model_name']
    )


def downgrade() -> None:
    op.drop_constraint('ml_jobs_recording_model_unique', 'ml_jobs', type_='unique')
