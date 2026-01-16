"""add_snapshot_path_to_object_events

Revision ID: 996f920084d2
Revises: 20260115_add_object_events
Create Date: 2026-01-15 17:49:57.964167

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '996f920084d2'
down_revision: Union[str, None] = '20260115_add_object_events'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "object_events",
        sa.Column("snapshot_path", sa.String(512), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("object_events", "snapshot_path")
