"""Alembic environment configuration for async migrations."""

import asyncio
import re
from datetime import datetime
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.config import get_settings
from app.database import Base
from app.models import Camera, Detection, MLJob, MLModel, Recording, User  # noqa: F401

config = context.config
settings = get_settings()

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

config.set_main_option("sqlalchemy.url", settings.database_url)


def process_revision_directives(context, revision, directives):
    """Generate date-based revision IDs matching filename convention.

    This ensures the revision ID matches the filename pattern (YYYYMMDD_description),
    preventing the common mistake of referencing filenames instead of revision IDs.
    """
    if not directives:
        return

    script = directives[0]
    if script.rev_id is None:
        return

    # Get the message/slug from the revision
    message = script.message or "migration"
    # Sanitize the message for use in revision ID
    slug = re.sub(r"[^a-z0-9_]", "_", message.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")

    # Generate date-based revision ID: YYYYMMDD_slug
    date_prefix = datetime.now().strftime("%Y%m%d")
    new_rev_id = f"{date_prefix}_{slug}"

    script.rev_id = new_rev_id


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        process_revision_directives=process_revision_directives,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        process_revision_directives=process_revision_directives,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
