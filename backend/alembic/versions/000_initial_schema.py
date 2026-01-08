"""Initial schema - create base tables

Revision ID: 000_initial_schema
Revises:
Create Date: 2024-12-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '000_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table('users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('is_admin', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username')
    )

    # Create cameras table
    op.create_table('cameras',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('host', sa.String(length=255), nullable=False),
        sa.Column('port', sa.Integer(), nullable=False, server_default='554'),
        sa.Column('path', sa.String(length=512), nullable=False, server_default='/cam/realmonitor'),
        sa.Column('username', sa.String(length=255), nullable=True),
        sa.Column('password', sa.String(length=255), nullable=True),
        sa.Column('transport', sa.String(length=10), nullable=False, server_default='tcp'),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='unknown'),
        sa.Column('last_seen', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('recording_enabled', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )

    # Create recordings table
    op.create_table('recordings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('camera_id', sa.Integer(), nullable=False),
        sa.Column('file_path', sa.String(length=1024), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=True),
        sa.Column('start_time', postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('end_time', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='recording'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('codec', sa.String(length=50), nullable=True),
        sa.Column('resolution', sa.String(length=20), nullable=True),
        sa.Column('fps', sa.Float(), nullable=True),
        sa.Column('ml_processed', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('file_path')
    )
    op.create_index('ix_recordings_camera_id', 'recordings', ['camera_id'], unique=False)
    op.create_index('ix_recordings_start_time', 'recordings', ['start_time'], unique=False)
    op.create_index('ix_recordings_status', 'recordings', ['status'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_recordings_status', table_name='recordings')
    op.drop_index('ix_recordings_start_time', table_name='recordings')
    op.drop_index('ix_recordings_camera_id', table_name='recordings')
    op.drop_table('recordings')
    op.drop_table('cameras')
    op.drop_table('users')
