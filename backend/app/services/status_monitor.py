"""Background task for monitoring camera status."""

import asyncio
import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_maker
from app.models import Camera
from app.models.camera import CameraStatus
from app.services.camera import test_camera_connection

logger = logging.getLogger(__name__)


class CameraStatusMonitor:
    """Background service to periodically check camera status."""

    def __init__(self, poll_interval_seconds: int = 60):
        self.poll_interval = poll_interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the background monitoring task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            f"Camera status monitor started (interval: {self.poll_interval}s)"
        )

    async def stop(self) -> None:
        """Stop the background monitoring task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Camera status monitor stopped")

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await self._check_all_cameras()
            except Exception:
                logger.exception("Error in camera status check")

            # Wait for next poll interval
            try:
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break

    async def _check_all_cameras(self) -> None:
        """Check status of all cameras."""
        async with async_session_maker() as db:
            result = await db.execute(select(Camera))
            cameras = list(result.scalars().all())

            if not cameras:
                return

            logger.debug(f"Checking status of {len(cameras)} cameras")

            for camera in cameras:
                await self._check_camera(db, camera)

            await db.commit()

    async def _check_camera(self, db: AsyncSession, camera: Camera) -> None:
        """Check status of a single camera."""
        try:
            result = await test_camera_connection(camera)

            if result.success:
                if camera.status != CameraStatus.ONLINE.value:
                    logger.info(f"Camera '{camera.name}' is now ONLINE")
                camera.status = CameraStatus.ONLINE.value
                camera.error_message = None
                from datetime import datetime
                camera.last_seen = datetime.utcnow()
            else:
                if camera.status != CameraStatus.ERROR.value:
                    logger.warning(
                        f"Camera '{camera.name}' is now ERROR: {result.message}"
                    )
                camera.status = CameraStatus.ERROR.value
                camera.error_message = result.message

        except Exception as e:
            logger.error(f"Error checking camera '{camera.name}': {e}")
            camera.status = CameraStatus.ERROR.value
            camera.error_message = str(e)


# Global monitor instance
status_monitor = CameraStatusMonitor()
