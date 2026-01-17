"""add_camera_scene_description

Revision ID: 7ad1350c109b
Revises: 996f920084d2
Create Date: 2026-01-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7ad1350c109b'
down_revision: Union[str, None] = '996f920084d2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "cameras",
        sa.Column("scene_description", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("cameras", "scene_description")
