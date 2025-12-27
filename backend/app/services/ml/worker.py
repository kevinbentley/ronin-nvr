"""ML Worker for processing inference jobs."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.detection import Detection
from app.models.ml_job import JobStatus, MLJob
from app.models.recording import Recording
from app.services.ml.detection_service import DetectionService, detection_service
from app.services.ml.events import ml_event_service
from app.services.ml.frame_extractor import FrameExtractor
from app.services.ml.job_queue import MLJobQueue, job_queue

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

logger = logging.getLogger(__name__)


class MLWorker:
    """Worker that processes ML inference jobs.

    Each worker runs in its own asyncio task, pulling jobs from the queue
    and processing them.
    """

    def __init__(
        self,
        worker_id: int,
        queue: Optional[MLJobQueue] = None,
        detector: Optional[DetectionService] = None,
        session_factory: Optional["async_sessionmaker[AsyncSession]"] = None,
    ):
        """Initialize worker.

        Args:
            worker_id: Unique identifier for this worker
            queue: Job queue to pull from
            detector: Detection service for inference
            session_factory: SQLAlchemy session factory
        """
        self.worker_id = worker_id
        self.queue = queue or job_queue
        self.detector = detector or detection_service
        self.session_factory = session_factory

        self._running = False
        self._current_job_id: Optional[int] = None
        self._task: Optional[asyncio.Task] = None

        settings = get_settings()
        self.frame_extractor = FrameExtractor(
            fps=settings.ml_default_fps,
            max_dimension=640,
        )

    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running

    @property
    def current_job_id(self) -> Optional[int]:
        """Get ID of job currently being processed."""
        return self._current_job_id

    async def start(self) -> None:
        """Start the worker loop."""
        if self._running:
            logger.warning(f"Worker {self.worker_id} already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Worker {self.worker_id} started")

    async def stop(self) -> None:
        """Stop the worker loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info(f"Worker {self.worker_id} stopped")

    async def _run_loop(self) -> None:
        """Main worker loop."""
        logger.info(f"Worker {self.worker_id} loop started")

        while self._running:
            try:
                # Wait for next job with timeout
                job_id = await self.queue.get_next(timeout=5.0)

                if job_id is None:
                    continue

                self._current_job_id = job_id

                # Process the job
                await self._process_job(job_id)

            except asyncio.CancelledError:
                logger.debug(f"Worker {self.worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(1)
            finally:
                self._current_job_id = None

        logger.info(f"Worker {self.worker_id} loop ended")

    async def _process_job(self, job_id: int) -> None:
        """Process a single job.

        Args:
            job_id: ID of job to process
        """
        if not self.session_factory:
            logger.error("No session factory configured")
            self.queue.mark_complete(job_id)
            return

        async with self.session_factory() as session:
            try:
                # Load job and recording
                job = await session.get(MLJob, job_id)
                if not job:
                    logger.error(f"Job {job_id} not found")
                    self.queue.mark_complete(job_id)
                    return

                recording = await session.get(Recording, job.recording_id)
                if not recording:
                    await self._fail_job(
                        session, job, f"Recording {job.recording_id} not found"
                    )
                    return

                # Mark job as processing
                self.queue.mark_active(job_id, job)
                job.status = JobStatus.PROCESSING.value
                job.started_at = datetime.utcnow()
                await session.commit()

                # Emit job started event (total_frames updated later)
                await ml_event_service.emit_job_started(job_id, 0)

                # Process the recording
                await self._process_recording(session, job, recording)

            except Exception as e:
                logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
                try:
                    job = await session.get(MLJob, job_id)
                    if job:
                        await self._fail_job(session, job, str(e))
                except Exception:
                    pass
            finally:
                self.queue.mark_complete(job_id)

    async def _process_recording(
        self,
        session: AsyncSession,
        job: MLJob,
        recording: Recording,
    ) -> None:
        """Process a recording for ML inference.

        Args:
            session: Database session
            job: MLJob being processed
            recording: Recording to process
        """
        video_path = Path(recording.file_path)
        if not video_path.exists():
            await self._fail_job(session, job, f"Video file not found: {video_path}")
            return

        model_name = job.model_name
        camera_id = recording.camera_id

        # Get video info for progress tracking
        video_info = await self.frame_extractor.get_video_info(video_path)
        if not video_info:
            await self._fail_job(session, job, "Could not get video info")
            return

        expected_frames = int(video_info.duration_seconds * self.frame_extractor.fps)
        job.total_frames = expected_frames
        await session.commit()

        # Emit updated job started with correct frame count
        await ml_event_service.emit_job_started(job.id, expected_frames)

        logger.info(
            f"Processing job {job.id}: {video_path.name}, "
            f"~{expected_frames} frames at {self.frame_extractor.fps} fps"
        )

        frames_processed = 0
        detections_count = 0
        detections_batch: list[Detection] = []
        batch_size = 50  # Commit detections in batches

        try:
            async for frame, frame_num, timestamp_ms in self.frame_extractor.extract_frames(
                video_path
            ):
                # Run detection
                results = self.detector.detect(frame, model_name)

                # Create detection records
                for result in results:
                    detection = Detection(
                        recording_id=recording.id,
                        camera_id=camera_id,
                        class_name=result.class_name,
                        confidence=result.confidence,
                        timestamp_ms=int(timestamp_ms),
                        frame_number=frame_num,
                        bbox_x=result.x,
                        bbox_y=result.y,
                        bbox_width=result.width,
                        bbox_height=result.height,
                        model_name=model_name,
                    )
                    detections_batch.append(detection)
                    detections_count += 1

                frames_processed += 1

                # Batch commit detections
                if len(detections_batch) >= batch_size:
                    session.add_all(detections_batch)
                    await session.commit()
                    detections_batch = []

                # Update progress periodically
                if frames_processed % 10 == 0:
                    progress = (frames_processed / expected_frames * 100) if expected_frames > 0 else 0
                    job.progress_percent = min(progress, 99.9)
                    job.frames_processed = frames_processed
                    job.detections_count = detections_count
                    await session.commit()

                    # Emit progress event
                    await ml_event_service.emit_job_progress(
                        job_id=job.id,
                        progress_percent=job.progress_percent,
                        frames_processed=frames_processed,
                        detections_count=detections_count,
                    )

            # Commit remaining detections
            if detections_batch:
                session.add_all(detections_batch)
                await session.commit()

            # Mark job complete
            job.status = JobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            job.progress_percent = 100.0
            job.frames_processed = frames_processed
            job.detections_count = detections_count
            if job.started_at:
                job.processing_time_seconds = (
                    job.completed_at - job.started_at
                ).total_seconds()
            await session.commit()

            logger.info(
                f"Job {job.id} complete: {frames_processed} frames, "
                f"{detections_count} detections in {job.processing_time_seconds:.1f}s"
            )

            # Emit job completed event
            await ml_event_service.emit_job_completed(
                job_id=job.id,
                frames_processed=frames_processed,
                detections_count=detections_count,
                processing_time_seconds=job.processing_time_seconds or 0.0,
            )

        except Exception as e:
            await self._fail_job(session, job, str(e))
            raise

    async def _fail_job(
        self,
        session: AsyncSession,
        job: MLJob,
        error_message: str,
    ) -> None:
        """Mark a job as failed.

        Args:
            session: Database session
            job: Job to fail
            error_message: Error description
        """
        job.status = JobStatus.FAILED.value
        job.error_message = error_message
        job.completed_at = datetime.utcnow()
        if job.started_at:
            job.processing_time_seconds = (
                job.completed_at - job.started_at
            ).total_seconds()
        await session.commit()
        logger.error(f"Job {job.id} failed: {error_message}")

        # Emit job failed event
        await ml_event_service.emit_job_failed(job.id, error_message)
