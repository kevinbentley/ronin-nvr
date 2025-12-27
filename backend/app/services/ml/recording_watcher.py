"""Recording Watcher for auto-processing new recordings."""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.ml_job import JobStatus, MLJob
from app.models.recording import Recording, RecordingStatus
from app.services.ml.coordinator import MLCoordinator, ml_coordinator
from app.services.playback import playback_service

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

logger = logging.getLogger(__name__)


class RecordingWatcher:
    """Watch for new recordings and auto-queue them for ML processing.

    Periodically scans for completed recordings that haven't been
    processed yet and submits them to the ML coordinator.
    """

    def __init__(
        self,
        coordinator: Optional[MLCoordinator] = None,
        check_interval: int = 30,
        session_factory: Optional["async_sessionmaker[AsyncSession]"] = None,
    ):
        """Initialize recording watcher.

        Args:
            coordinator: ML coordinator for job submission
            check_interval: Seconds between checks
            session_factory: SQLAlchemy async session factory
        """
        self.coordinator = coordinator or ml_coordinator
        self.check_interval = check_interval
        self.session_factory = session_factory

        settings = get_settings()
        self._enabled = settings.ml_enabled and settings.ml_auto_process
        self._default_model = settings.ml_default_model

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_check: Optional[datetime] = None

    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running

    def set_session_factory(
        self, session_factory: "async_sessionmaker[AsyncSession]"
    ) -> None:
        """Set the session factory for database access.

        Args:
            session_factory: SQLAlchemy async session factory
        """
        self.session_factory = session_factory

    async def start(self) -> None:
        """Start the recording watcher."""
        if not self._enabled:
            logger.info("Recording watcher disabled (ml_enabled or ml_auto_process is False)")
            return

        if self._running:
            logger.warning("Recording watcher already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Recording watcher started (check every {self.check_interval}s)")

    async def stop(self) -> None:
        """Stop the recording watcher."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Recording watcher stopped")

    async def _run_loop(self) -> None:
        """Main watcher loop."""
        while self._running:
            try:
                await self._check_new_recordings()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Recording watcher error: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)

    async def _check_new_recordings(self) -> None:
        """Check for new recordings and submit for processing."""
        if not self.session_factory:
            logger.debug("No session factory - skipping check")
            return

        now = datetime.utcnow()

        # Look for recordings completed in the last check interval + buffer
        # (or since startup if first check)
        if self._last_check:
            since = self._last_check - timedelta(seconds=10)  # Small overlap buffer
        else:
            # On first run, check recordings from the last hour
            since = now - timedelta(hours=1)

        self._last_check = now

        async with self.session_factory() as session:
            # Find recordings that:
            # 1. Are completed
            # 2. Were modified recently (file completed)
            # 3. Don't have a pending/processing ML job

            # Get recordings that don't have an ML job yet
            subquery = (
                select(MLJob.recording_id)
                .where(
                    MLJob.model_name == self._default_model,
                    MLJob.status.in_([
                        JobStatus.PENDING.value,
                        JobStatus.QUEUED.value,
                        JobStatus.PROCESSING.value,
                        JobStatus.COMPLETED.value,
                    ]),
                )
            )

            result = await session.execute(
                select(Recording)
                .where(
                    Recording.status == RecordingStatus.COMPLETED.value,
                    Recording.created_at >= since,
                    ~Recording.id.in_(subquery),
                )
                .order_by(Recording.created_at.asc())
                .limit(10)  # Process at most 10 at a time
            )
            recordings = result.scalars().all()

            if recordings:
                logger.info(f"Found {len(recordings)} new recordings to process")

            for recording in recordings:
                try:
                    job = await self.coordinator.process_recording(
                        recording_id=recording.id,
                        model_name=self._default_model,
                        priority=0,  # Normal priority for auto-detected
                    )
                    if job:
                        logger.info(
                            f"Queued recording {recording.id} "
                            f"({recording.file_path}) for processing"
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to queue recording {recording.id}: {e}"
                    )

    async def process_existing_recordings(
        self,
        camera_id: Optional[int] = None,
        limit: int = 100,
    ) -> int:
        """Process existing recordings that haven't been analyzed.

        Useful for initial setup or reprocessing.

        Args:
            camera_id: Optional camera to filter by
            limit: Maximum number of recordings to queue

        Returns:
            Number of recordings queued
        """
        if not self.session_factory:
            return 0

        async with self.session_factory() as session:
            # Find completed recordings without ML jobs
            subquery = (
                select(MLJob.recording_id)
                .where(
                    MLJob.model_name == self._default_model,
                    MLJob.status.in_([
                        JobStatus.PENDING.value,
                        JobStatus.QUEUED.value,
                        JobStatus.PROCESSING.value,
                        JobStatus.COMPLETED.value,
                    ]),
                )
            )

            query = (
                select(Recording)
                .where(
                    Recording.status == RecordingStatus.COMPLETED.value,
                    ~Recording.id.in_(subquery),
                )
            )

            if camera_id:
                query = query.where(Recording.camera_id == camera_id)

            query = query.order_by(Recording.created_at.desc()).limit(limit)

            result = await session.execute(query)
            recordings = result.scalars().all()

            queued = 0
            for recording in recordings:
                try:
                    job = await self.coordinator.process_recording(
                        recording_id=recording.id,
                        model_name=self._default_model,
                        priority=-1,  # Lower priority for batch processing
                    )
                    if job:
                        queued += 1
                except Exception as e:
                    logger.error(f"Failed to queue recording {recording.id}: {e}")

            logger.info(f"Queued {queued} existing recordings for processing")
            return queued


# Global watcher instance
recording_watcher = RecordingWatcher()
