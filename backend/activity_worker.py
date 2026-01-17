#!/usr/bin/env python3
"""Activity characterization worker using Vision LLM.

This worker monitors detections and uses a Vision LLM to characterize
activities, distinguishing normal behavior from suspicious activity.

Key features:
- Extracts 4 frames around detection time (1 second apart)
- Creates 2x2 mosaic for temporal context
- Sends to VLLM with camera scene description
- Updates detection with analysis result and concern level
- Supports manual processing of specific recordings

Usage:
    ./activity_worker.py                    # Run in continuous mode
    ./activity_worker.py --recording 123    # Process specific recording
    ./activity_worker.py --video /path.mp4  # Process video file directly
"""

import argparse
import asyncio
import logging
import signal
import socket
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Add the backend directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.config import Settings, get_settings
from app.services.ml.frame_extractor import FrameExtractor
from app.services.vllm.characterization import ActivityCharacterizer, ConcernLevel
from app.services.vllm.client import VLLMClient
from app.services.vllm.mosaic import add_frame_numbers, create_mosaic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("activity_worker")


class ActivityWorker:
    """Worker for VLLM-based activity characterization."""

    def __init__(
        self,
        settings: Settings,
        database_url: Optional[str] = None,
    ):
        """Initialize activity worker.

        Args:
            settings: Application settings
            database_url: Optional database URL override
        """
        self.settings = settings
        self.database_url = database_url or str(settings.database_url)
        self.storage_root = settings.storage_root
        self.worker_id = f"activity-{socket.gethostname()}-{id(self)}"

        # VLLM settings
        self.vllm_endpoint = settings.vllm_endpoint
        self.vllm_timeout = settings.vllm_timeout
        self.frame_count = settings.vllm_frame_count
        self.frame_interval_ms = settings.vllm_frame_interval_ms
        self.poll_interval = settings.vllm_poll_interval
        self.max_age_seconds = settings.vllm_max_age_seconds

        # Components (initialized on start)
        self._pool = None
        self._running = False
        self._vllm_client: Optional[VLLMClient] = None
        self._characterizer: Optional[ActivityCharacterizer] = None
        self._frame_extractor: Optional[FrameExtractor] = None

    async def start(self) -> None:
        """Start the worker in continuous mode."""
        import asyncpg

        logger.info(f"Worker {self.worker_id} starting...")
        logger.info(f"VLLM endpoint: {self.vllm_endpoint}")
        logger.info(
            f"Config: frame_count={self.frame_count}, "
            f"interval={self.frame_interval_ms}ms, "
            f"poll={self.poll_interval}s"
        )

        # Initialize components
        self._vllm_client = VLLMClient(
            endpoint=self.vllm_endpoint,
            timeout=self.vllm_timeout,
        )
        self._characterizer = ActivityCharacterizer(
            client=self._vllm_client,
            frame_count=self.frame_count,
            frame_interval_ms=self.frame_interval_ms,
        )
        self._frame_extractor = FrameExtractor(fps=1.0)

        # Check VLLM connectivity
        if not await self._vllm_client.health_check():
            logger.warning(
                f"VLLM endpoint not responding at {self.vllm_endpoint}, "
                "will retry on each detection"
            )

        # Connect to database
        self._pool = await asyncpg.create_pool(
            self.database_url, min_size=2, max_size=5, ssl=False
        )

        self._running = True
        logger.info(f"Worker {self.worker_id} ready")

        try:
            await self._run_loop()
        except asyncio.CancelledError:
            logger.info(f"Worker {self.worker_id} cancelled")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        self._running = False

        if self._vllm_client:
            await self._vllm_client.close()
            self._vllm_client = None

        if self._pool:
            await self._pool.close()
            self._pool = None

        logger.info(f"Worker {self.worker_id} stopped")

    async def _run_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                # Find detections without LLM analysis
                processed = await self._process_pending_detections()

                if processed == 0:
                    # No work, wait before polling again
                    await asyncio.sleep(self.poll_interval)
                else:
                    # Processed some, check for more immediately
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)

    async def _process_pending_detections(self) -> int:
        """Find and process detections missing LLM analysis.

        Returns:
            Number of detections processed
        """
        if not self._pool:
            return 0

        # Calculate cutoff time
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.max_age_seconds)

        async with self._pool.acquire() as conn:
            # Find detections without LLM description
            rows = await conn.fetch(
                """
                SELECT
                    d.id,
                    d.camera_id,
                    d.recording_id,
                    d.class_name,
                    d.timestamp_ms,
                    d.detected_at,
                    d.snapshot_path,
                    c.name as camera_name,
                    c.scene_description,
                    r.file_path as recording_path
                FROM detections d
                JOIN cameras c ON d.camera_id = c.id
                LEFT JOIN recordings r ON d.recording_id = r.id
                WHERE d.llm_description IS NULL
                    AND d.detected_at > $1
                ORDER BY d.detected_at DESC
                LIMIT 10
                """,
                cutoff,
            )

        processed = 0
        for row in rows:
            try:
                await self._process_detection(row)
                processed += 1
            except Exception as e:
                logger.error(
                    f"Failed to process detection {row['id']}: {e}",
                    exc_info=True,
                )

        return processed

    async def _process_detection(self, row: dict) -> None:
        """Process a single detection with VLLM analysis.

        Args:
            row: Detection row from database query
        """
        detection_id = row["id"]
        camera_id = row["camera_id"]
        camera_name = row["camera_name"]
        class_name = row["class_name"]
        recording_path = row["recording_path"]
        timestamp_ms = row["timestamp_ms"]
        scene_description = row["scene_description"]

        logger.info(
            f"Processing detection {detection_id}: "
            f"camera={camera_name}, class={class_name}"
        )

        # We need a recording to extract frames
        if not recording_path:
            logger.warning(
                f"Detection {detection_id} has no recording, skipping VLLM analysis"
            )
            # Mark as processed with note
            await self._update_detection(
                detection_id,
                "No recording available for frame extraction",
                ConcernLevel.NONE,
            )
            return

        video_path = Path(recording_path)
        if not video_path.is_absolute():
            video_path = self.storage_root / recording_path

        if not video_path.exists():
            logger.error(f"Recording file not found: {video_path}")
            await self._update_detection(
                detection_id,
                f"Recording file not found: {recording_path}",
                ConcernLevel.NONE,
            )
            return

        # Extract frames around the detection time
        frames, timestamps = await self._extract_frames_around(
            video_path, timestamp_ms
        )

        if not frames:
            logger.error(f"Failed to extract frames for detection {detection_id}")
            await self._update_detection(
                detection_id,
                "Failed to extract frames from recording",
                ConcernLevel.NONE,
            )
            return

        # Analyze with VLLM
        try:
            analysis = await self._characterizer.analyze_frames(
                frames=frames,
                timestamps_ms=timestamps,
                scene_description=scene_description,
                detected_class=class_name,
            )

            # Update detection with analysis
            await self._update_detection(
                detection_id,
                analysis.description,
                analysis.concern_level,
                analysis.activity_type,
            )

            # Log result
            level_emoji = {
                ConcernLevel.NONE: "⚪",
                ConcernLevel.LOW: "🟢",
                ConcernLevel.MEDIUM: "🟡",
                ConcernLevel.HIGH: "🟠",
                ConcernLevel.EMERGENCY: "🔴",
            }.get(analysis.concern_level, "⚪")

            logger.info(
                f"{level_emoji} Detection {detection_id} "
                f"[{analysis.concern_level.value.upper()}]: "
                f"{analysis.description[:100]}"
            )

        except Exception as e:
            logger.error(f"VLLM analysis failed for detection {detection_id}: {e}")
            raise

    async def _extract_frames_around(
        self,
        video_path: Path,
        center_timestamp_ms: int,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Extract frames around the detection timestamp.

        Extracts frame_count frames centered on the detection,
        spaced frame_interval_ms apart.

        Args:
            video_path: Path to video file
            center_timestamp_ms: Detection timestamp in milliseconds

        Returns:
            Tuple of (frames list, timestamps list)
        """
        if not self._frame_extractor:
            return [], []

        # Get video info
        video_info = await self._frame_extractor.get_video_info(video_path)
        if not video_info:
            return [], []

        # Calculate time range
        # For 4 frames at 1s intervals, we want: t-1.5s, t-0.5s, t+0.5s, t+1.5s
        # This centers the detection in the sequence
        total_span_ms = (self.frame_count - 1) * self.frame_interval_ms
        start_offset_ms = center_timestamp_ms - (total_span_ms // 2)

        # Clamp to video bounds
        start_offset_ms = max(0, start_offset_ms)
        video_duration_ms = video_info.duration_seconds * 1000
        end_offset_ms = start_offset_ms + total_span_ms
        if end_offset_ms > video_duration_ms:
            start_offset_ms = max(0, video_duration_ms - total_span_ms)

        # Calculate timestamps for each frame
        target_timestamps_ms = [
            start_offset_ms + i * self.frame_interval_ms
            for i in range(self.frame_count)
        ]

        frames = []
        actual_timestamps = []

        # Extract frames at each target timestamp
        for target_ms in target_timestamps_ms:
            start_sec = target_ms / 1000.0

            # Extract single frame at this time
            extractor = FrameExtractor(fps=30.0, max_dimension=1280)

            try:
                async for frame, _, _ in extractor.extract_frames(
                    video_path,
                    start_time=start_sec,
                    duration=0.1,
                ):
                    frames.append(frame)
                    actual_timestamps.append(target_ms)
                    break  # Only need first frame

            except Exception as e:
                logger.warning(f"Failed to extract frame at {target_ms}ms: {e}")

        return frames, actual_timestamps

    async def _update_detection(
        self,
        detection_id: int,
        description: str,
        concern_level: ConcernLevel,
        activity_type: Optional[str] = None,
    ) -> None:
        """Update detection with LLM analysis results.

        Args:
            detection_id: Detection ID to update
            description: LLM description
            concern_level: Concern level classification
            activity_type: Optional activity type
        """
        if not self._pool:
            return

        # Store analysis in extra_data JSON
        import json
        extra_data = {
            "vllm_analysis": {
                "concern_level": concern_level.value,
                "activity_type": activity_type,
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
            }
        }

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE detections
                SET llm_description = $2,
                    extra_data = COALESCE(extra_data, '{}'::jsonb) || $3::jsonb
                WHERE id = $1
                """,
                detection_id,
                description,
                json.dumps(extra_data),
            )

    async def process_recording(self, recording_id: int) -> int:
        """Process all detections for a specific recording.

        Args:
            recording_id: Recording ID to process

        Returns:
            Number of detections processed
        """
        import asyncpg

        logger.info(f"Processing recording {recording_id}")

        # Initialize components if not already
        if not self._vllm_client:
            self._vllm_client = VLLMClient(
                endpoint=self.vllm_endpoint,
                timeout=self.vllm_timeout,
            )
            self._characterizer = ActivityCharacterizer(
                client=self._vllm_client,
                frame_count=self.frame_count,
                frame_interval_ms=self.frame_interval_ms,
            )
            self._frame_extractor = FrameExtractor(fps=1.0)

        # Connect to database
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                self.database_url, min_size=2, max_size=5, ssl=False
            )

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    d.id,
                    d.camera_id,
                    d.recording_id,
                    d.class_name,
                    d.timestamp_ms,
                    d.detected_at,
                    d.snapshot_path,
                    c.name as camera_name,
                    c.scene_description,
                    r.file_path as recording_path
                FROM detections d
                JOIN cameras c ON d.camera_id = c.id
                JOIN recordings r ON d.recording_id = r.id
                WHERE d.recording_id = $1
                ORDER BY d.timestamp_ms
                """,
                recording_id,
            )

        if not rows:
            logger.warning(f"No detections found for recording {recording_id}")
            return 0

        logger.info(f"Found {len(rows)} detections to process")

        processed = 0
        for row in rows:
            try:
                await self._process_detection(row)
                processed += 1
            except Exception as e:
                logger.error(
                    f"Failed to process detection {row['id']}: {e}",
                    exc_info=True,
                )

        logger.info(f"Processed {processed}/{len(rows)} detections")
        return processed

    async def process_video_file(
        self,
        video_path: Path,
        detection_timestamps_ms: Optional[list[int]] = None,
    ) -> None:
        """Process a video file directly without database lookup.

        Used for testing/development with standalone video files.

        Args:
            video_path: Path to video file
            detection_timestamps_ms: Optional list of timestamps to analyze
                                     If not provided, analyzes at 10s intervals
        """
        logger.info(f"Processing video file: {video_path}")

        # Initialize components if not already
        if not self._vllm_client:
            self._vllm_client = VLLMClient(
                endpoint=self.vllm_endpoint,
                timeout=self.vllm_timeout,
            )
            self._characterizer = ActivityCharacterizer(
                client=self._vllm_client,
                frame_count=self.frame_count,
                frame_interval_ms=self.frame_interval_ms,
            )
            self._frame_extractor = FrameExtractor(fps=1.0)

        # Get video info
        video_info = await self._frame_extractor.get_video_info(video_path)
        if not video_info:
            logger.error(f"Could not read video: {video_path}")
            return

        logger.info(
            f"Video info: {video_info.width}x{video_info.height}, "
            f"{video_info.duration_seconds:.1f}s, {video_info.fps:.1f}fps"
        )

        # Generate timestamps if not provided
        if not detection_timestamps_ms:
            # Sample every 10 seconds
            detection_timestamps_ms = [
                int(t * 1000)
                for t in range(
                    5, int(video_info.duration_seconds) - 5, 10
                )
            ]

        if not detection_timestamps_ms:
            # Video too short, just analyze middle
            detection_timestamps_ms = [int(video_info.duration_seconds * 500)]

        logger.info(f"Analyzing {len(detection_timestamps_ms)} timestamps")

        for timestamp_ms in detection_timestamps_ms:
            logger.info(f"\n--- Analyzing at {timestamp_ms/1000:.1f}s ---")

            frames, timestamps = await self._extract_frames_around(
                video_path, timestamp_ms
            )

            if not frames:
                logger.warning(f"No frames extracted at {timestamp_ms}ms")
                continue

            try:
                analysis = await self._characterizer.analyze_frames(
                    frames=frames,
                    timestamps_ms=timestamps,
                    scene_description=None,
                    detected_class="person",  # Default for manual testing
                )

                level_emoji = {
                    ConcernLevel.NONE: "⚪",
                    ConcernLevel.LOW: "🟢",
                    ConcernLevel.MEDIUM: "🟡",
                    ConcernLevel.HIGH: "🟠",
                    ConcernLevel.EMERGENCY: "🔴",
                }.get(analysis.concern_level, "⚪")

                print(f"\n{level_emoji} Concern Level: {analysis.concern_level.value.upper()}")
                print(f"   Activity Type: {analysis.activity_type}")
                print(f"   Description: {analysis.description}")

            except Exception as e:
                logger.error(f"VLLM analysis failed: {e}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Activity characterization worker using Vision LLM"
    )
    parser.add_argument(
        "--recording",
        type=int,
        help="Process specific recording ID (manual mode)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Process video file directly (testing mode)",
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        action="append",
        help="Timestamp in ms to analyze (can specify multiple, requires --video)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        help="Override VLLM endpoint URL",
    )

    args = parser.parse_args()

    # Load settings
    settings = get_settings()

    # Override endpoint if specified
    if args.endpoint:
        settings = Settings(
            **{**settings.model_dump(), "vllm_endpoint": args.endpoint}
        )

    worker = ActivityWorker(settings)

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        loop.create_task(worker.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    if args.video:
        # Manual video file processing
        await worker.process_video_file(
            args.video,
            detection_timestamps_ms=args.timestamp,
        )
    elif args.recording:
        # Process specific recording
        await worker.process_recording(args.recording)
    else:
        # Continuous mode
        await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
