"""add_detection_mosaic_path

Revision ID: 8b2e9f4c3a1d
Revises: 7ad1350c109b
Create Date: 2026-01-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8b2e9f4c3a1d'
down_revision: Union[str, None] = '7ad1350c109b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "detections",
        sa.Column("mosaic_path", sa.String(500), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("detections", "mosaic_path")
