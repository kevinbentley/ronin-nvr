#!/usr/bin/env python3
"""Standalone ML worker process for processing video analysis jobs.

This script runs independently of the main backend server, connecting
directly to PostgreSQL to claim and process jobs. Multiple workers can
run concurrently, on the same machine or on remote servers.

Usage:
    ./ml_worker.py                      # Run with defaults from .env
    ./ml_worker.py --workers 4          # Run 4 worker processes
    ./ml_worker.py --database-url "postgresql://..." --workers 2

Workers use PostgreSQL LISTEN/NOTIFY for instant job notifications,
falling back to polling every 5 seconds if no notifications arrive.
"""

import argparse
import asyncio
import logging
import multiprocessing
import os
import signal
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add the backend directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.config import get_settings
from app.services.ml.detection_service import DetectionService
from app.services.ml.frame_extractor import FrameExtractor
from app.services.ml.job_service import JobInfo, JobService
from app.services.ml.model_manager import ModelManager
from app.services.ml.motion_detector import MotionDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ml_worker")


class StandaloneMLWorker:
    """Standalone ML worker that processes jobs from PostgreSQL.

    Each worker instance runs in its own process, connecting to the
    database to claim and process jobs.
    """

    def __init__(
        self,
        worker_id: str,
        database_url: str,
        fps: float = 1.0,
        confidence_threshold: float = 0.5,
        motion_enabled: bool = True,
    ):
        """Initialize worker.

        Args:
            worker_id: Unique identifier for this worker
            database_url: PostgreSQL connection URL
            fps: Frames per second to extract from videos
            confidence_threshold: Detection confidence threshold
            motion_enabled: Whether to run motion detection
        """
        self.worker_id = worker_id
        self.database_url = database_url
        self.fps = fps
        self.confidence_threshold = confidence_threshold
        self.motion_enabled = motion_enabled

        self._running = False
        self._job_service: Optional[JobService] = None
        self._current_job_id: Optional[int] = None

        # Initialize processing components
        self.model_manager = ModelManager()
        self.detector = DetectionService(model_mgr=self.model_manager)
        self.frame_extractor = FrameExtractor(fps=fps, max_dimension=640)

        # Class filter - only save detections for these classes
        settings = get_settings()
        filter_str = settings.ml_class_filter.strip()
        if filter_str:
            self.class_filter = set(c.strip().lower() for c in filter_str.split(","))
            logger.info(f"Class filter enabled: {self.class_filter}")
        else:
            self.class_filter = None  # No filter, accept all classes

    async def start(self) -> None:
        """Start the worker and begin processing jobs."""
        logger.info(f"Worker {self.worker_id} starting...")

        # Connect to database
        self._job_service = JobService(self.database_url)
        await self._job_service.connect()
        await self._job_service.start_listening()

        self._running = True

        # Reset any orphaned jobs from previous runs
        await self._job_service.reset_orphaned_jobs()

        logger.info(f"Worker {self.worker_id} ready, waiting for jobs...")

        try:
            await self._run_loop()
        except asyncio.CancelledError:
            logger.info(f"Worker {self.worker_id} cancelled")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        self._running = False

        if self._job_service:
            await self._job_service.stop_listening()
            await self._job_service.disconnect()
            self._job_service = None

        # Unload models
        self.model_manager.unload_all()

        logger.info(f"Worker {self.worker_id} stopped")

    async def _run_loop(self) -> None:
        """Main worker loop: claim jobs, process them, repeat."""
        assert self._job_service is not None

        while self._running:
            try:
                # Try to claim a job
                job = await self._job_service.claim_next_job(self.worker_id)

                if job:
                    self._current_job_id = job.id
                    logger.info(
                        f"Worker {self.worker_id} claimed job {job.id} "
                        f"for recording {job.recording_id}"
                    )
                    await self._process_job(job)
                    self._current_job_id = None
                else:
                    # No jobs available, wait for notification or timeout
                    await self._job_service.wait_for_notification(timeout=5.0)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}", exc_info=True)
                self._current_job_id = None
                await asyncio.sleep(1)

    async def _process_job(self, job: JobInfo) -> None:
        """Process a single ML job.

        Args:
            job: Job information from database
        """
        assert self._job_service is not None

        video_path = Path(job.file_path)
        if not video_path.exists():
            await self._job_service.fail_job(
                job.id, f"Video file not found: {video_path}"
            )
            return

        # Get video info
        video_info = await self.frame_extractor.get_video_info(video_path)
        if not video_info:
            await self._job_service.fail_job(job.id, "Could not get video info")
            return

        expected_frames = int(video_info.duration_seconds * self.fps)
        logger.info(
            f"Processing job {job.id}: {video_path.name}, "
            f"~{expected_frames} frames at {self.fps} fps"
        )

        # Create fresh motion detector for this video
        motion_detector: Optional[MotionDetector] = None
        if self.motion_enabled:
            motion_detector = MotionDetector.from_settings()

        frames_processed = 0
        detections_count = 0
        detections_batch: list[dict] = []
        batch_size = 50
        heartbeat_interval = 10  # Update heartbeat every N frames

        try:
            async for frame, frame_num, timestamp_ms in self.frame_extractor.extract_frames(
                video_path
            ):
                # Run YOLO object detection
                results = self.detector.detect(frame, job.model_name)

                # Create detection records (filtered by class if configured)
                for result in results:
                    # Apply class filter if configured
                    if self.class_filter and result.class_name.lower() not in self.class_filter:
                        continue

                    detections_batch.append({
                        "recording_id": job.recording_id,
                        "camera_id": job.camera_id,
                        "class_name": result.class_name,
                        "confidence": result.confidence,
                        "timestamp_ms": int(timestamp_ms),
                        "frame_number": frame_num,
                        "bbox_x": result.x,
                        "bbox_y": result.y,
                        "bbox_width": result.width,
                        "bbox_height": result.height,
                        "model_name": job.model_name,
                    })
                    detections_count += 1

                # Run motion detection
                if motion_detector is not None:
                    motion_result = motion_detector.detect(frame)
                    if motion_result.has_motion:
                        if motion_result.bounding_boxes:
                            bbox = motion_result.bounding_boxes[0]
                        else:
                            bbox = (0.0, 0.0, 1.0, 1.0)

                        detections_batch.append({
                            "recording_id": job.recording_id,
                            "camera_id": job.camera_id,
                            "class_name": "motion",
                            "confidence": min(motion_result.motion_percent / 100.0, 1.0),
                            "timestamp_ms": int(timestamp_ms),
                            "frame_number": frame_num,
                            "bbox_x": bbox[0],
                            "bbox_y": bbox[1],
                            "bbox_width": bbox[2],
                            "bbox_height": bbox[3],
                            "model_name": "motion_detector",
                        })
                        detections_count += 1

                frames_processed += 1

                # Batch commit detections
                if len(detections_batch) >= batch_size:
                    await self._job_service.add_detections(detections_batch)
                    detections_batch = []

                # Update progress and heartbeat periodically
                if frames_processed % heartbeat_interval == 0:
                    progress = (
                        (frames_processed / expected_frames * 100)
                        if expected_frames > 0
                        else 0
                    )
                    await self._job_service.update_progress(
                        job.id,
                        min(progress, 99.9),
                        frames_processed,
                        detections_count,
                    )

            # Commit remaining detections
            if detections_batch:
                await self._job_service.add_detections(detections_batch)

            # Mark job complete
            await self._job_service.complete_job(
                job.id,
                frames_processed,
                detections_count,
                expected_frames,
            )

            logger.info(
                f"Job {job.id} complete: {frames_processed} frames, "
                f"{detections_count} detections"
            )

        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}", exc_info=True)
            await self._job_service.fail_job(job.id, str(e))
            raise


def run_worker_process(
    worker_num: int,
    database_url: str,
    fps: float,
    confidence: float,
    motion_enabled: bool,
) -> None:
    """Entry point for worker subprocess.

    Args:
        worker_num: Worker number (0-based)
        database_url: PostgreSQL connection URL
        fps: Frames per second to extract
        confidence: Detection confidence threshold
        motion_enabled: Whether to run motion detection
    """
    # Generate unique worker ID
    hostname = socket.gethostname()[:16]
    pid = os.getpid()
    worker_id = f"{hostname}-{pid}-{worker_num}"

    # Configure this process's logger
    logger = logging.getLogger("ml_worker")
    logger.info(f"Starting worker process: {worker_id}")

    # Create worker
    worker = StandaloneMLWorker(
        worker_id=worker_id,
        database_url=database_url,
        fps=fps,
        confidence_threshold=confidence,
        motion_enabled=motion_enabled,
    )

    # Handle shutdown signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    shutdown_event = asyncio.Event()

    def signal_handler(sig: int, frame: object) -> None:
        logger.info(f"Worker {worker_id} received signal {sig}, shutting down...")
        loop.call_soon_threadsafe(shutdown_event.set)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    async def run_with_shutdown() -> None:
        """Run worker until shutdown signal."""
        worker_task = asyncio.create_task(worker.start())
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        done, pending = await asyncio.wait(
            [worker_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await worker.stop()

    try:
        loop.run_until_complete(run_with_shutdown())
    finally:
        loop.close()


def main() -> None:
    """Main entry point for ML worker CLI."""
    parser = argparse.ArgumentParser(
        description="Standalone ML worker for processing video analysis jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./ml_worker.py                           Run with defaults from .env
  ./ml_worker.py --workers 4               Run 4 worker processes
  ./ml_worker.py --database-url "postgresql://user:pass@host/db" --workers 2
        """,
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of worker processes (default: from ML_WORKERS env or 4)",
    )
    parser.add_argument(
        "--database-url",
        "-d",
        type=str,
        default=None,
        help="PostgreSQL connection URL (default: from DATABASE_URL env)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frames per second to extract (default: from ML_DEFAULT_FPS or 1.0)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=None,
        help="Detection confidence threshold (default: from ML_CONFIDENCE_THRESHOLD or 0.5)",
    )
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="Disable motion detection",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load settings from environment
    settings = get_settings()

    # Resolve configuration from args, env, or defaults
    database_url = args.database_url or settings.database_url
    num_workers = args.workers or settings.ml_workers
    fps = args.fps or settings.ml_default_fps
    confidence = args.confidence or settings.ml_confidence_threshold
    motion_enabled = not args.no_motion and settings.motion_detection_enabled

    # Validate database URL
    if not database_url:
        logger.error("No database URL provided. Set DATABASE_URL or use --database-url")
        sys.exit(1)

    logger.info(f"Starting {num_workers} ML worker(s)")
    logger.info(f"Database: {database_url.split('@')[-1]}")  # Hide credentials
    logger.info(f"FPS: {fps}, Confidence: {confidence}, Motion: {motion_enabled}")

    # Spawn worker processes
    processes: list[multiprocessing.Process] = []

    def shutdown_handler(sig: int, frame: object) -> None:
        """Handle shutdown signal in main process."""
        logger.info(f"Received signal {sig}, shutting down workers...")
        for p in processes:
            if p.is_alive():
                p.terminate()

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    for i in range(num_workers):
        p = multiprocessing.Process(
            target=run_worker_process,
            args=(i, database_url, fps, confidence, motion_enabled),
            name=f"ml-worker-{i}",
        )
        p.start()
        processes.append(p)
        logger.info(f"Started worker process {i} (PID: {p.pid})")

    # Wait for all processes to complete
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logger.info("Interrupted, terminating workers...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)

    logger.info("All workers stopped")


if __name__ == "__main__":
    main()
