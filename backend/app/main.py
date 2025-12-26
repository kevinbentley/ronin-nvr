"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import api_router
from app.config import get_settings
from app.database import close_db, init_db
from app.services.camera_stream import stream_manager
from app.services.retention import retention_monitor
from app.services.status_monitor import status_monitor

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    db_available = False

    # Startup - try to connect to database but don't fail if unavailable
    try:
        await init_db()
        logger.info("Database initialized successfully")
        db_available = True
    except Exception as e:
        logger.warning(f"Database unavailable at startup: {e}")
        logger.warning("App will run without database - configure DATABASE_URL in .env")

    # Start camera status monitor if database is available
    if db_available:
        try:
            await status_monitor.start()
        except Exception as e:
            logger.warning(f"Failed to start status monitor: {e}")

    # Start retention monitor
    try:
        await retention_monitor.start()
    except Exception as e:
        logger.warning(f"Failed to start retention monitor: {e}")

    yield

    # Shutdown
    try:
        await stream_manager.stop_all()
        logger.info("All streams stopped")
    except Exception:
        pass

    try:
        await retention_monitor.stop()
    except Exception:
        pass

    try:
        await status_monitor.stop()
    except Exception:
        pass

    try:
        await close_db()
    except Exception:
        pass


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    lifespan=lifespan,
)

# Add CORS middleware for frontend
# Note: Allow all origins in development for HLS streaming
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API info."""
    return {
        "name": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }
