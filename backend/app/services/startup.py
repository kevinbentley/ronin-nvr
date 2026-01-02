"""Startup tasks for the application."""

import asyncio
import logging

from sqlalchemy import select

from app.database import async_session_maker
from app.models import Camera
from app.services.camera_stream import stream_manager

logger = logging.getLogger(__name__)


async def auto_start_recording_cameras() -> None:
    """Auto-start streams for all cameras with recording_enabled=true.

    This ensures that after a backend restart, cameras resume recording
    and producing HLS segments for live detection.
    """
    async with async_session_maker() as db:
        result = await db.execute(
            select(Camera).where(Camera.recording_enabled == True)  # noqa: E712
        )
        cameras = list(result.scalars().all())

    if not cameras:
        logger.info("No cameras with recording enabled found")
        return

    logger.info(f"Auto-starting streams for {len(cameras)} cameras with recording enabled")

    # Start streams concurrently but with some delay between each
    # to avoid overwhelming the cameras/network
    started = 0
    failed = 0

    for camera in cameras:
        try:
            success = await stream_manager.start_stream(camera, recording_enabled=True)
            if success:
                started += 1
                logger.info(f"Auto-started stream for camera '{camera.name}'")
            else:
                failed += 1
                logger.warning(f"Failed to auto-start stream for camera '{camera.name}'")
        except Exception as e:
            failed += 1
            logger.error(f"Error auto-starting stream for camera '{camera.name}': {e}")

        # Small delay between starting each camera to avoid network congestion
        await asyncio.sleep(0.5)

    logger.info(f"Auto-start complete: {started} started, {failed} failed")
