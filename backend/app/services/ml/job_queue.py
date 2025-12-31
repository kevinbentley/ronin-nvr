"""ML Job Queue for managing processing jobs."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.models.ml_job import JobStatus, MLJob

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PriorityJob:
    """Wrapper for priority queue ordering.

    Jobs are sorted by priority (higher = more urgent), then by creation time.
    """

    priority: int
    created_at: datetime = field(compare=False)
    job_id: int = field(compare=False)

    def __post_init__(self):
        # Negate priority so higher priority comes first
        self.priority = -self.priority


class MLJobQueue:
    """In-process job queue for ML processing.

    Provides a priority-based asyncio queue for managing ML processing jobs.
    Jobs are persisted to the database, with the queue providing the runtime
    ordering and distribution to workers.
    """

    def __init__(self, max_size: int = 100):
        """Initialize the job queue.

        Args:
            max_size: Maximum number of pending jobs
        """
        self.max_size = max_size
        self._queue: asyncio.PriorityQueue[PriorityJob] = asyncio.PriorityQueue(
            maxsize=max_size
        )
        self._active_jobs: dict[int, MLJob] = {}
        self._cancelled_jobs: set[int] = set()
        self._lock = asyncio.Lock()

    async def enqueue(self, job: MLJob) -> bool:
        """Add a job to the queue.

        Args:
            job: MLJob instance to queue

        Returns:
            True if queued successfully, False if queue is full
        """
        if self._queue.full():
            logger.warning(f"Queue is full, cannot add job {job.id}")
            return False

        priority_job = PriorityJob(
            priority=job.priority,
            created_at=job.created_at,
            job_id=job.id,
        )

        try:
            self._queue.put_nowait(priority_job)
            logger.debug(f"Job {job.id} added to queue (priority={job.priority})")
            return True
        except asyncio.QueueFull:
            logger.warning(f"Queue full, could not add job {job.id}")
            return False

    async def get_next(self, timeout: Optional[float] = None) -> Optional[int]:
        """Get the next job ID to process.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Job ID or None if timeout/empty
        """
        try:
            if timeout:
                priority_job = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=timeout,
                )
            else:
                priority_job = await self._queue.get()

            job_id = priority_job.job_id

            # Check if job was cancelled while waiting
            if job_id in self._cancelled_jobs:
                self._cancelled_jobs.discard(job_id)
                self._queue.task_done()
                return await self.get_next(timeout)

            return job_id

        except asyncio.TimeoutError:
            return None
        except asyncio.CancelledError:
            raise

    def mark_active(self, job_id: int, job: MLJob) -> None:
        """Mark a job as being actively processed.

        Args:
            job_id: Job ID
            job: MLJob instance
        """
        self._active_jobs[job_id] = job

    def mark_complete(self, job_id: int) -> None:
        """Mark a job as complete.

        Args:
            job_id: Job ID
        """
        self._active_jobs.pop(job_id, None)
        self._queue.task_done()

    def cancel_job(self, job_id: int) -> bool:
        """Cancel a pending job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if job was cancelled
        """
        # If job is active, can't cancel (worker is processing)
        if job_id in self._active_jobs:
            logger.warning(f"Cannot cancel active job {job_id}")
            return False

        # Mark as cancelled so it gets skipped when dequeued
        self._cancelled_jobs.add(job_id)
        logger.info(f"Job {job_id} marked for cancellation")
        return True

    @property
    def pending_count(self) -> int:
        """Number of jobs waiting in queue."""
        return self._queue.qsize()

    @property
    def active_count(self) -> int:
        """Number of jobs currently being processed."""
        return len(self._active_jobs)

    @property
    def active_job_ids(self) -> list[int]:
        """List of active job IDs."""
        return list(self._active_jobs.keys())

    def get_status(self) -> dict:
        """Get queue status summary.

        Returns:
            Dict with queue statistics
        """
        return {
            "pending": self.pending_count,
            "active": self.active_count,
            "max_size": self.max_size,
            "active_jobs": self.active_job_ids,
        }

    async def clear(self) -> int:
        """Clear all pending jobs from queue.

        Returns:
            Number of jobs cleared
        """
        count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                count += 1
            except asyncio.QueueEmpty:
                break
        logger.info(f"Cleared {count} jobs from queue")
        return count


# Global queue instance
job_queue = MLJobQueue()
