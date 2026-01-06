"""Recording Watcher for auto-processing new recordings."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Set, TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.camera import Camera
from app.models.ml_job import JobStatus, MLJob
from app.models.recording import Recording, RecordingStatus
from app.services.ml.coordinator import MLCoordinator, ml_coordinator
from app.services.playback import playback_service, RecordingFile


def _safe_camera_name(name: str) -> str:
    """Convert camera name to filesystem-safe version (matches CameraStream)."""
    return "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in name
    )

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

logger = logging.getLogger(__name__)


class RecordingWatcher:
    """Watch for new recordings and auto-queue them for ML processing.

    Scans the filesystem for completed recordings and submits them
    to the ML coordinator for processing.
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
        self._segment_duration = settings.segment_duration_minutes * 60

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._processed_files: Set[str] = set()  # Track files we've already queued

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

        # Load already-processed files from database
        await self._load_processed_files()

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

    async def _load_processed_files(self) -> None:
        """Load list of already-processed file paths from database."""
        if not self.session_factory:
            return

        async with self.session_factory() as session:
            # Get all recording file paths that have ML jobs
            result = await session.execute(
                select(Recording.file_path).join(
                    MLJob, MLJob.recording_id == Recording.id
                ).where(
                    MLJob.model_name == self._default_model
                )
            )
            paths = result.scalars().all()
            self._processed_files = set(paths)
            logger.info(f"Loaded {len(self._processed_files)} already-processed recordings")

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
        """Check for new recordings on filesystem and submit for processing."""
        if not self.session_factory:
            logger.debug("No session factory - skipping check")
            return

        # Scan filesystem for all recordings
        all_recordings = playback_service.scan_recordings()

        # Filter to completed recordings (not currently being written)
        # A recording is considered complete if it's older than segment duration + buffer
        now = datetime.now(timezone.utc)
        min_age = timedelta(seconds=self._segment_duration + 60)  # Segment duration + 1 min buffer

        new_recordings = []
        for rec in all_recordings:
            # Skip if already processed
            if str(rec.path) in self._processed_files:
                continue

            # Skip if too recent (still being written)
            file_age = now - rec.start_time
            if file_age < min_age:
                continue

            new_recordings.append(rec)

        if not new_recordings:
            return

        logger.info(f"Found {len(new_recordings)} new recordings to process")

        async with self.session_factory() as session:
            # Get safe_camera_name -> camera_id mapping
            result = await session.execute(select(Camera))
            cameras = {_safe_camera_name(cam.name): cam.id for cam in result.scalars().all()}

            for rec_file in new_recordings[:10]:  # Process at most 10 at a time
                try:
                    await self._process_recording_file(session, rec_file, cameras)
                except Exception as e:
                    logger.error(f"Failed to process {rec_file.path}: {e}")

    async def _process_recording_file(
        self,
        session: AsyncSession,
        rec_file: RecordingFile,
        cameras: dict[str, int],
    ) -> None:
        """Create database record and submit for ML processing."""
        # Get camera ID
        camera_id = cameras.get(rec_file.camera_name)
        if not camera_id:
            logger.warning(f"Camera '{rec_file.camera_name}' not found in database")
            return

        # Check if recording already exists in database
        result = await session.execute(
            select(Recording).where(Recording.file_path == str(rec_file.path))
        )
        recording = result.scalars().first()

        if not recording:
            # Create new recording record
            recording = Recording(
                camera_id=camera_id,
                file_path=str(rec_file.path),
                file_size=rec_file.size,
                start_time=rec_file.start_time,
                end_time=rec_file.start_time + timedelta(seconds=rec_file.duration_seconds or self._segment_duration),
                duration_seconds=rec_file.duration_seconds or self._segment_duration,
                status=RecordingStatus.COMPLETED.value,
            )
            session.add(recording)
            try:
                await session.commit()
                await session.refresh(recording)
                logger.info(f"Created recording record for {rec_file.path}")
            except IntegrityError:
                # Recording was inserted by another process between our check and insert
                await session.rollback()
                result = await session.execute(
                    select(Recording).where(Recording.file_path == str(rec_file.path))
                )
                recording = result.scalars().first()
                if not recording:
                    logger.error(f"Recording {rec_file.path} not found after IntegrityError")
                    return
                logger.debug(f"Recording {rec_file.path} already exists (race condition)")

        # Submit for ML processing
        job = await self.coordinator.process_recording(
            recording_id=recording.id,
            model_name=self._default_model,
            priority=0,
        )

        if job:
            self._processed_files.add(str(rec_file.path))
            logger.info(f"Queued {rec_file.path} for ML processing (job {job.id})")

    async def process_existing_recordings(
        self,
        camera_name: Optional[str] = None,
        limit: int = 100,
    ) -> int:
        """Process existing recordings that haven't been analyzed.

        Useful for initial setup or reprocessing.

        Args:
            camera_name: Optional camera to filter by
            limit: Maximum number of recordings to queue

        Returns:
            Number of recordings queued
        """
        if not self.session_factory:
            return 0

        # Scan filesystem
        all_recordings = playback_service.scan_recordings(camera_name=camera_name)

        # Filter out already processed
        new_recordings = [
            rec for rec in all_recordings
            if str(rec.path) not in self._processed_files
        ]

        # Limit
        new_recordings = new_recordings[:limit]

        if not new_recordings:
            logger.info("No unprocessed recordings found")
            return 0

        async with self.session_factory() as session:
            # Get safe_camera_name -> camera_id mapping
            result = await session.execute(select(Camera))
            cameras = {_safe_camera_name(cam.name): cam.id for cam in result.scalars().all()}

            queued = 0
            for rec_file in new_recordings:
                try:
                    await self._process_recording_file(session, rec_file, cameras)
                    queued += 1
                except Exception as e:
                    logger.error(f"Failed to process {rec_file.path}: {e}")

            logger.info(f"Queued {queued} existing recordings for processing")
            return queued


# Global watcher instance
recording_watcher = RecordingWatcher()
