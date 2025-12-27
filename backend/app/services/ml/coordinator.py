"""ML Coordinator for managing workers and job processing."""

import asyncio
import logging
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.ml_job import JobStatus, MLJob
from app.models.recording import Recording
from app.services.ml.events import ml_event_service
from app.services.ml.job_queue import MLJobQueue, job_queue
from app.services.ml.model_manager import ModelManager, model_manager
from app.services.ml.worker import MLWorker

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

logger = logging.getLogger(__name__)


class MLCoordinator:
    """Coordinate ML processing across multiple workers.

    Manages a pool of workers, the job queue, and provides methods
    for submitting jobs and monitoring status.
    """

    def __init__(
        self,
        num_workers: Optional[int] = None,
        queue: Optional[MLJobQueue] = None,
        model_mgr: Optional[ModelManager] = None,
        session_factory: Optional["async_sessionmaker[AsyncSession]"] = None,
    ):
        """Initialize coordinator.

        Args:
            num_workers: Number of worker processes (default from settings)
            queue: Job queue instance
            model_mgr: Model manager instance
            session_factory: SQLAlchemy async session factory
        """
        settings = get_settings()
        self.num_workers = num_workers or settings.ml_workers
        self.queue = queue or job_queue
        self.model_manager = model_mgr or model_manager
        self.session_factory = session_factory

        self._workers: list[MLWorker] = []
        self._running = False
        self._default_model = settings.ml_default_model

    @property
    def is_running(self) -> bool:
        """Check if coordinator is running."""
        return self._running

    def set_session_factory(
        self, session_factory: "async_sessionmaker[AsyncSession]"
    ) -> None:
        """Set the session factory for database access.

        Args:
            session_factory: SQLAlchemy async session factory
        """
        self.session_factory = session_factory
        # Update existing workers
        for worker in self._workers:
            worker.session_factory = session_factory

    async def start(self) -> None:
        """Start the coordinator and all workers."""
        if self._running:
            logger.warning("Coordinator already running")
            return

        if not self.session_factory:
            logger.error("Cannot start coordinator: no session factory configured")
            return

        logger.info(f"Starting ML coordinator with {self.num_workers} workers")

        # Create and start workers
        self._workers = []
        for i in range(self.num_workers):
            worker = MLWorker(
                worker_id=i,
                queue=self.queue,
                session_factory=self.session_factory,
            )
            self._workers.append(worker)
            await worker.start()

        self._running = True
        logger.info("ML coordinator started")

    async def stop(self) -> None:
        """Stop the coordinator and all workers."""
        if not self._running:
            return

        logger.info("Stopping ML coordinator")

        # Stop all workers
        for worker in self._workers:
            await worker.stop()

        self._workers = []
        self._running = False

        # Unload all models
        self.model_manager.unload_all()

        logger.info("ML coordinator stopped")

    async def process_recording(
        self,
        recording_id: int,
        model_name: Optional[str] = None,
        priority: int = 0,
    ) -> Optional[MLJob]:
        """Submit a recording for ML processing.

        Args:
            recording_id: ID of recording to process
            model_name: Model to use (default from settings)
            priority: Job priority (higher = more urgent)

        Returns:
            Created MLJob or None if submission failed
        """
        if not self.session_factory:
            logger.error("Cannot process: no session factory")
            return None

        model_name = model_name or self._default_model

        async with self.session_factory() as session:
            # Verify recording exists
            recording = await session.get(Recording, recording_id)
            if not recording:
                logger.error(f"Recording {recording_id} not found")
                return None

            # Check if job already exists for this recording/model
            existing = await session.execute(
                select(MLJob).where(
                    MLJob.recording_id == recording_id,
                    MLJob.model_name == model_name,
                    MLJob.status.in_([
                        JobStatus.PENDING.value,
                        JobStatus.QUEUED.value,
                        JobStatus.PROCESSING.value,
                    ]),
                )
            )
            if existing.scalar_one_or_none():
                logger.warning(
                    f"Job already exists for recording {recording_id} with model {model_name}"
                )
                return None

            # Create job
            job = MLJob(
                recording_id=recording_id,
                model_name=model_name,
                status=JobStatus.PENDING.value,
                priority=priority,
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)

            # Add to queue
            job.status = JobStatus.QUEUED.value
            await session.commit()

            if not await self.queue.enqueue(job):
                job.status = JobStatus.FAILED.value
                job.error_message = "Queue is full"
                await session.commit()
                return None

            logger.info(f"Created job {job.id} for recording {recording_id}")

            # Emit job created event
            await ml_event_service.emit_job_created(
                job_id=job.id,
                recording_id=recording_id,
                model_name=model_name,
            )

            return job

    async def cancel_job(self, job_id: int) -> bool:
        """Cancel a pending job.

        Args:
            job_id: ID of job to cancel

        Returns:
            True if job was cancelled
        """
        if not self.session_factory:
            return False

        # Try to cancel in queue
        if not self.queue.cancel_job(job_id):
            return False

        # Update database
        async with self.session_factory() as session:
            job = await session.get(MLJob, job_id)
            if job and job.status in [JobStatus.PENDING.value, JobStatus.QUEUED.value]:
                job.status = JobStatus.CANCELLED.value
                job.completed_at = datetime.utcnow()
                await session.commit()
                logger.info(f"Job {job_id} cancelled")
                return True

        return False

    async def get_job(self, job_id: int) -> Optional[MLJob]:
        """Get a job by ID.

        Args:
            job_id: Job ID

        Returns:
            MLJob or None
        """
        if not self.session_factory:
            return None

        async with self.session_factory() as session:
            return await session.get(MLJob, job_id)

    def get_status(self) -> dict:
        """Get coordinator status summary.

        Returns:
            Dict with status information
        """
        worker_status = []
        for worker in self._workers:
            worker_status.append({
                "id": worker.worker_id,
                "running": worker.is_running,
                "current_job": worker.current_job_id,
            })

        return {
            "running": self._running,
            "workers": len(self._workers),
            "worker_status": worker_status,
            "queue": self.queue.get_status(),
            "models_loaded": self.model_manager.list_available_models(),
        }


# Global coordinator instance
ml_coordinator = MLCoordinator()
