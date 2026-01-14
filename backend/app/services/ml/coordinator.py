"""ML Coordinator for managing job creation and status.

This simplified coordinator only creates jobs in the database.
Actual job processing is handled by standalone worker processes
that connect directly to PostgreSQL.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.ml_job import JobStatus, MLJob
from app.models.recording import Recording
from app.services.ml.events import ml_event_service
from app.services.ml.model_manager import ModelManager, model_manager

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

logger = logging.getLogger(__name__)

# Channel name for job notifications (must match job_service.py)
JOB_NOTIFY_CHANNEL = "ml_jobs_notify"


class MLCoordinator:
    """Coordinate ML job creation and status queries.

    This coordinator creates jobs in the database and sends PostgreSQL
    NOTIFY to wake up any listening workers. Job processing is handled
    by standalone worker processes.
    """

    def __init__(
        self,
        model_mgr: Optional[ModelManager] = None,
        session_factory: Optional["async_sessionmaker[AsyncSession]"] = None,
    ):
        """Initialize coordinator.

        Args:
            model_mgr: Model manager instance
            session_factory: SQLAlchemy async session factory
        """
        settings = get_settings()
        self.model_manager = model_mgr or model_manager
        self.session_factory = session_factory

        self._running = False
        self._default_model = settings.ml_default_model
        self._num_workers = settings.ml_workers

    @property
    def is_running(self) -> bool:
        """Check if coordinator is 'running' (enabled for job creation).

        Note: With standalone workers, this just indicates whether
        job creation is enabled, not whether workers are active.
        """
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
        """Enable the coordinator for job creation.

        With standalone workers, this just sets the running flag.
        Workers are started separately via ml_worker.py.
        """
        if self._running:
            logger.warning("Coordinator already running")
            return

        if not self.session_factory:
            logger.error("Cannot start coordinator: no session factory configured")
            return

        logger.info("ML coordinator started (job creation enabled)")
        self._running = True

    async def stop(self) -> None:
        """Disable the coordinator.

        With standalone workers, this just sets the running flag.
        Workers continue running independently.
        """
        if not self._running:
            return

        self._running = False
        logger.info("ML coordinator stopped (job creation disabled)")

    async def process_recording(
        self,
        recording_id: int,
        model_name: Optional[str] = None,
        priority: int = 0,
    ) -> Optional[MLJob]:
        """Submit a recording for ML processing.

        Creates a job in the database with PENDING status.
        Standalone workers will pick up the job and process it.

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
                logger.debug(
                    f"Job already exists for recording {recording_id} with model {model_name}"
                )
                return None

            # Create job in PENDING state
            job = MLJob(
                recording_id=recording_id,
                model_name=model_name,
                status=JobStatus.PENDING.value,
                priority=priority,
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)

            # Send PostgreSQL NOTIFY to wake up workers
            await session.execute(
                text(f"NOTIFY {JOB_NOTIFY_CHANNEL}, 'new_job'")
            )
            await session.commit()

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

        async with self.session_factory() as session:
            job = await session.get(MLJob, job_id)
            if not job:
                return False

            if job.status not in [JobStatus.PENDING.value, JobStatus.QUEUED.value]:
                logger.warning(f"Cannot cancel job {job_id} with status {job.status}")
                return False

            job.status = JobStatus.CANCELLED.value
            job.completed_at = datetime.utcnow()
            await session.commit()
            logger.info(f"Job {job_id} cancelled")
            return True

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

    async def get_status(self) -> dict:
        """Get coordinator status summary.

        Queries the database for job counts and active workers.

        Returns:
            Dict with status information
        """
        if not self.session_factory:
            return {
                "running": self._running,
                "workers": 0,
                "worker_status": [],
                "queue": {
                    "pending": 0,
                    "active": 0,
                    "max_size": 100,
                    "active_jobs": [],
                },
                "models_loaded": [],
            }

        async with self.session_factory() as session:
            # Count pending jobs
            pending_result = await session.execute(
                select(func.count()).where(MLJob.status == JobStatus.PENDING.value)
            )
            pending_count = pending_result.scalar() or 0

            # Count processing jobs
            processing_result = await session.execute(
                select(func.count()).where(MLJob.status == JobStatus.PROCESSING.value)
            )
            processing_count = processing_result.scalar() or 0

            # Get active workers (jobs with recent heartbeat)
            two_minutes_ago = datetime.utcnow() - timedelta(minutes=2)
            workers_result = await session.execute(
                select(
                    MLJob.worker_id,
                    MLJob.id,
                    MLJob.progress_percent,
                )
                .where(MLJob.status == JobStatus.PROCESSING.value)
                .where(MLJob.worker_id.isnot(None))
                .where(MLJob.last_heartbeat > two_minutes_ago)
            )
            active_workers = workers_result.all()

            # Build worker status
            worker_status = []
            seen_workers = set()
            for row in active_workers:
                worker_id = row.worker_id
                if worker_id and worker_id not in seen_workers:
                    seen_workers.add(worker_id)
                    worker_status.append({
                        "id": len(seen_workers) - 1,  # Numeric ID for compatibility
                        "running": True,
                        "current_job": row.id,
                    })

            # Add placeholder workers if we have configured workers
            while len(worker_status) < self._num_workers:
                worker_status.append({
                    "id": len(worker_status),
                    "running": False,
                    "current_job": None,
                })

            active_job_ids = [w["current_job"] for w in worker_status if w["current_job"]]

            return {
                "running": self._running,
                "workers": len(seen_workers),
                "worker_status": worker_status,
                "queue": {
                    "pending": pending_count,
                    "active": processing_count,
                    "max_size": 100,  # No longer relevant with database queue
                    "active_jobs": active_job_ids,
                },
                "models_loaded": self.model_manager.list_available_models(),
            }


# Global coordinator instance
ml_coordinator = MLCoordinator()
