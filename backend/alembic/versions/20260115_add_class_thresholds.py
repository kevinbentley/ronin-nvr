"""Add class_thresholds column to ml_settings.

Revision ID: 20260115_add_class_thresholds
Revises: 20260106_add_onvif_support
Create Date: 2026-01-15
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260115_add_class_thresholds"
down_revision = "20260106_onvif"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add class_thresholds column
    op.add_column(
        "ml_settings",
        sa.Column("class_thresholds", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    # Remove class_thresholds column
    op.drop_column("ml_settings", "class_thresholds")
