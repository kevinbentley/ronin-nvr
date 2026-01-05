"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app import __version__
from app.api import api_router
from app.config import get_settings
from app.database import close_db, get_db, init_db
from app.rate_limiter import limiter
from app.services.stream_client import stream_manager
from app.services.retention import retention_monitor
from app.services.status_monitor import status_monitor
from app.services.startup import auto_start_recording_cameras
from app.services.ml import ml_coordinator, recording_watcher
from app.services.ml.live_detection_listener import live_detection_listener

settings = get_settings()
logger = logging.getLogger(__name__)


async def create_default_admin() -> None:
    """Create default admin user if no users exist."""
    from app.database import AsyncSessionLocal
    from app.schemas.auth import UserCreate
    from app.services.auth import AuthService
    from app.services.encryption import generate_random_password

    async with AsyncSessionLocal() as db:
        auth_service = AuthService(db)
        user_count = await auth_service.get_user_count()

        if user_count == 0:
            # Generate or use configured password
            password = settings.default_admin_password or generate_random_password()

            user_data = UserCreate(
                username="admin",
                password=password,
                is_admin=True,
            )
            await auth_service.create_user(user_data)

            logger.info("=" * 60)
            logger.info("Default admin user created:")
            logger.info(f"  Username: admin")
            if not settings.default_admin_password:
                logger.info(f"  Password: {password}")
                logger.info("  (Set DEFAULT_ADMIN_PASSWORD in .env to specify password)")
            else:
                logger.info("  Password: (from DEFAULT_ADMIN_PASSWORD)")
            logger.info("=" * 60)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Handle rate limit exceeded errors."""
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


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

        # Create default admin user if no users exist
        try:
            await create_default_admin()
        except Exception as e:
            logger.warning(f"Failed to create default admin: {e}")

        # Auto-start streams for cameras with recording enabled
        try:
            await auto_start_recording_cameras()
        except Exception as e:
            logger.warning(f"Failed to auto-start camera streams: {e}")

    # Start retention monitor
    try:
        await retention_monitor.start()
    except Exception as e:
        logger.warning(f"Failed to start retention monitor: {e}")

    # Start ML coordinator and recording watcher if database available and ML enabled
    if db_available and settings.ml_enabled:
        try:
            from app.database import AsyncSessionLocal
            ml_coordinator.set_session_factory(AsyncSessionLocal)
            recording_watcher.set_session_factory(AsyncSessionLocal)
            await ml_coordinator.start()
            await recording_watcher.start()
            logger.info("ML coordinator and recording watcher started")
        except Exception as e:
            logger.warning(f"Failed to start ML services: {e}")

        # Start live detection listener for real-time SSE notifications
        if settings.live_detection_enabled:
            try:
                await live_detection_listener.start()
                logger.info("Live detection listener started")
            except Exception as e:
                logger.warning(f"Failed to start live detection listener: {e}")

    yield

    # Shutdown
    try:
        await stream_manager.stop_all()
        logger.info("All streams stopped")
    except Exception:
        pass

    try:
        await recording_watcher.stop()
    except Exception:
        pass

    try:
        await ml_coordinator.stop()
    except Exception:
        pass

    try:
        await live_detection_listener.stop()
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

# Add rate limiter to app state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Add CORS middleware for frontend
# Parse origins from comma-separated config string
cors_origins = [origin.strip() for origin in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,  # Required for Authorization headers
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
