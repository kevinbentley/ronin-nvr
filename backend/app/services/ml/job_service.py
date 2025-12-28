"""Job service for database-level job claiming and management.

Uses PostgreSQL FOR UPDATE SKIP LOCKED for atomic job claiming
and LISTEN/NOTIFY for instant job notifications.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)


# Channel name for job notifications
JOB_NOTIFY_CHANNEL = "ml_jobs_notify"


@dataclass
class JobInfo:
    """Job information returned from database."""

    id: int
    recording_id: int
    model_name: str
    priority: int
    file_path: str  # From recording table
    camera_id: int  # From recording table


class JobService:
    """Database-level job claiming and management service.

    Uses PostgreSQL FOR UPDATE SKIP LOCKED for atomic job claiming,
    ensuring multiple workers can safely compete for jobs.
    """

    def __init__(self, database_url: str):
        """Initialize job service.

        Args:
            database_url: PostgreSQL connection URL (postgresql://...)
        """
        # Convert asyncpg URL format if needed
        self.database_url = database_url.replace(
            "postgresql+asyncpg://", "postgresql://"
        )
        self.pool: Optional[asyncpg.Pool] = None
        self._listener_connection: Optional[asyncpg.Connection] = None
        self._notification_queue: asyncio.Queue[str] = asyncio.Queue()

    async def connect(self) -> None:
        """Connect to database and set up connection pool."""
        if self.pool is not None:
            return

        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=2,
            max_size=10,
        )
        logger.info("JobService connected to database")

    async def disconnect(self) -> None:
        """Disconnect from database."""
        if self._listener_connection:
            await self._listener_connection.close()
            self._listener_connection = None

        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("JobService disconnected from database")

    async def start_listening(self) -> None:
        """Start listening for job notifications.

        Notifications are put into a queue that workers can wait on.
        """
        if self._listener_connection:
            return

        self._listener_connection = await asyncpg.connect(self.database_url)

        def notification_handler(
            connection: asyncpg.Connection,
            pid: int,
            channel: str,
            payload: str,
        ) -> None:
            """Handle incoming notifications."""
            try:
                self._notification_queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass  # Queue full, notification dropped

        await self._listener_connection.add_listener(
            JOB_NOTIFY_CHANNEL, notification_handler
        )
        logger.info(f"Listening for notifications on channel: {JOB_NOTIFY_CHANNEL}")

    async def stop_listening(self) -> None:
        """Stop listening for notifications."""
        if self._listener_connection:
            await self._listener_connection.remove_listener(
                JOB_NOTIFY_CHANNEL, lambda *args: None
            )
            await self._listener_connection.close()
            self._listener_connection = None

    async def wait_for_notification(self, timeout: float = 5.0) -> bool:
        """Wait for a job notification.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if notification received, False if timeout
        """
        try:
            await asyncio.wait_for(
                self._notification_queue.get(),
                timeout=timeout,
            )
            return True
        except asyncio.TimeoutError:
            return False

    async def notify_new_job(self) -> None:
        """Send notification that a new job is available.

        Call this after creating a new job to wake up idle workers.
        """
        if not self.pool:
            return

        async with self.pool.acquire() as conn:
            await conn.execute(f"NOTIFY {JOB_NOTIFY_CHANNEL}, 'new_job'")

    async def claim_next_job(self, worker_id: str) -> Optional[JobInfo]:
        """Atomically claim the next available pending job.

        Uses FOR UPDATE SKIP LOCKED to prevent race conditions between workers.

        Args:
            worker_id: Unique identifier for this worker

        Returns:
            JobInfo if a job was claimed, None if no jobs available
        """
        if not self.pool:
            raise RuntimeError("JobService not connected")

        async with self.pool.acquire() as conn:
            # Use a transaction with FOR UPDATE SKIP LOCKED
            async with conn.transaction():
                # Find and lock the next pending job
                row = await conn.fetchrow(
                    """
                    UPDATE ml_jobs
                    SET
                        status = 'processing',
                        started_at = NOW(),
                        worker_id = $1,
                        last_heartbeat = NOW()
                    WHERE id = (
                        SELECT id FROM ml_jobs
                        WHERE status = 'pending'
                        ORDER BY priority DESC, created_at ASC
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING id, recording_id, model_name, priority
                    """,
                    worker_id,
                )

                if not row:
                    return None

                # Get recording info
                recording = await conn.fetchrow(
                    "SELECT file_path, camera_id FROM recordings WHERE id = $1",
                    row["recording_id"],
                )

                if not recording:
                    # Recording was deleted, mark job as failed
                    await conn.execute(
                        """
                        UPDATE ml_jobs
                        SET status = 'failed',
                            error_message = 'Recording not found',
                            completed_at = NOW()
                        WHERE id = $1
                        """,
                        row["id"],
                    )
                    return None

                return JobInfo(
                    id=row["id"],
                    recording_id=row["recording_id"],
                    model_name=row["model_name"],
                    priority=row["priority"],
                    file_path=recording["file_path"],
                    camera_id=recording["camera_id"],
                )

    async def update_progress(
        self,
        job_id: int,
        progress_percent: float,
        frames_processed: int,
        detections_count: int,
    ) -> None:
        """Update job progress and heartbeat.

        Args:
            job_id: Job ID
            progress_percent: Progress percentage (0-100)
            frames_processed: Number of frames processed
            detections_count: Number of detections found
        """
        if not self.pool:
            return

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ml_jobs
                SET progress_percent = $2,
                    frames_processed = $3,
                    detections_count = $4,
                    last_heartbeat = NOW()
                WHERE id = $1
                """,
                job_id,
                progress_percent,
                frames_processed,
                detections_count,
            )

    async def heartbeat(self, job_id: int) -> None:
        """Update job heartbeat to prevent orphan detection.

        Args:
            job_id: Job ID
        """
        if not self.pool:
            return

        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE ml_jobs SET last_heartbeat = NOW() WHERE id = $1",
                job_id,
            )

    async def complete_job(
        self,
        job_id: int,
        frames_processed: int,
        detections_count: int,
        total_frames: int,
    ) -> None:
        """Mark a job as completed.

        Args:
            job_id: Job ID
            frames_processed: Total frames processed
            detections_count: Total detections found
            total_frames: Total frames in video
        """
        if not self.pool:
            return

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ml_jobs
                SET status = 'completed',
                    completed_at = NOW(),
                    progress_percent = 100.0,
                    frames_processed = $2,
                    detections_count = $3,
                    total_frames = $4,
                    processing_time_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))
                WHERE id = $1
                """,
                job_id,
                frames_processed,
                detections_count,
                total_frames,
            )

    async def fail_job(self, job_id: int, error_message: str) -> None:
        """Mark a job as failed.

        Args:
            job_id: Job ID
            error_message: Error description
        """
        if not self.pool:
            return

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ml_jobs
                SET status = 'failed',
                    error_message = $2,
                    completed_at = NOW(),
                    processing_time_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))
                WHERE id = $1
                """,
                job_id,
                error_message,
            )

    async def reset_orphaned_jobs(self, timeout_minutes: int = 5) -> int:
        """Reset jobs that have been processing too long without heartbeat.

        Args:
            timeout_minutes: Minutes since last heartbeat to consider orphaned

        Returns:
            Number of jobs reset
        """
        if not self.pool:
            return 0

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE ml_jobs
                SET status = 'pending',
                    started_at = NULL,
                    worker_id = NULL,
                    last_heartbeat = NULL,
                    progress_percent = 0,
                    frames_processed = 0
                WHERE status = 'processing'
                  AND last_heartbeat < NOW() - INTERVAL '%s minutes'
                """ % timeout_minutes,
            )
            # Parse result to get row count
            count = int(result.split()[-1]) if result else 0
            if count > 0:
                logger.info(f"Reset {count} orphaned jobs")
            return count

    async def get_pending_count(self) -> int:
        """Get count of pending jobs."""
        if not self.pool:
            return 0

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) FROM ml_jobs WHERE status = 'pending'"
            )
            return row[0] if row else 0

    async def get_processing_count(self) -> int:
        """Get count of jobs currently processing."""
        if not self.pool:
            return 0

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) FROM ml_jobs WHERE status = 'processing'"
            )
            return row[0] if row else 0

    async def get_worker_status(self) -> list[dict]:
        """Get status of active workers (based on recent heartbeats).

        Returns:
            List of worker status dicts
        """
        if not self.pool:
            return []

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT worker_id, id as job_id, progress_percent, last_heartbeat
                FROM ml_jobs
                WHERE status = 'processing'
                  AND worker_id IS NOT NULL
                  AND last_heartbeat > NOW() - INTERVAL '2 minutes'
                ORDER BY worker_id
                """
            )
            return [
                {
                    "worker_id": row["worker_id"],
                    "current_job": row["job_id"],
                    "progress": row["progress_percent"],
                    "last_heartbeat": row["last_heartbeat"].isoformat()
                    if row["last_heartbeat"]
                    else None,
                }
                for row in rows
            ]

    async def add_detections(
        self,
        detections: list[dict],
    ) -> None:
        """Batch insert detections.

        Args:
            detections: List of detection dicts with keys:
                - recording_id, camera_id, class_name, confidence,
                - timestamp_ms, frame_number, bbox_x, bbox_y,
                - bbox_width, bbox_height, model_name
        """
        if not self.pool or not detections:
            return

        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO detections (
                    recording_id, camera_id, class_name, confidence,
                    timestamp_ms, frame_number, bbox_x, bbox_y,
                    bbox_width, bbox_height, model_name
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                [
                    (
                        d["recording_id"],
                        d["camera_id"],
                        d["class_name"],
                        d["confidence"],
                        d["timestamp_ms"],
                        d["frame_number"],
                        d["bbox_x"],
                        d["bbox_y"],
                        d["bbox_width"],
                        d["bbox_height"],
                        d["model_name"],
                    )
                    for d in detections
                ],
            )
