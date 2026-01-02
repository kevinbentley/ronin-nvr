"""Database connection and session management."""

import logging
from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Alias for backwards compatibility
AsyncSessionLocal = async_session_maker


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that provides a database session."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def run_migrations() -> None:
    """Run Alembic migrations to upgrade database schema.

    This runs synchronously since Alembic's command API is synchronous.
    Called during application startup before async operations begin.
    """
    from alembic import command
    from alembic.config import Config

    # Find alembic.ini relative to this file
    backend_dir = Path(__file__).parent.parent
    alembic_ini = backend_dir / "alembic.ini"

    if not alembic_ini.exists():
        logger.warning(f"alembic.ini not found at {alembic_ini}, skipping migrations")
        return

    try:
        alembic_cfg = Config(str(alembic_ini))
        # Set the script location relative to the ini file
        alembic_cfg.set_main_option("script_location", str(backend_dir / "alembic"))

        logger.info("Running database migrations...")
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations completed successfully")
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        raise


async def init_db() -> None:
    """Initialize database and verify connection.

    Note: Migrations are handled by docker-entrypoint.sh before the
    application starts. We skip running migrations here because Alembic's
    asyncio.run() conflicts with FastAPI's already-running event loop.
    """
    # Verify connection works
    async with engine.begin() as conn:
        # Just verify we can connect - migrations handle schema
        await conn.execute(text("SELECT 1"))


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()
