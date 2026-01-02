#!/usr/bin/env python3
"""Process historical recordings through ML detection.

This script processes video recordings that haven't been through ML detection yet,
extracting objects (people, cars, trucks, etc.) and saving them to the database.

Usage:
    # Process all unprocessed recordings
    python scripts/process_historical.py

    # Process specific camera only
    python scripts/process_historical.py --camera-id 1

    # Process recordings from a specific date range
    python scripts/process_historical.py --start-date 2026-01-01 --end-date 2026-01-02

    # Reprocess already-processed recordings
    python scripts/process_historical.py --reprocess

    # Limit number of recordings to process
    python scripts/process_historical.py --limit 10

    # Dry run - show what would be processed without doing it
    python scripts/process_historical.py --dry-run

    # Process every Nth frame (faster but less thorough)
    python scripts/process_historical.py --sample-rate 30
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import asyncpg
import cv2
import numpy as np

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.services.ml.detection_service import DetectionResult, DetectionService
from app.services.ml.model_manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class HistoricalProcessor:
    """Process historical recordings for ML detection."""

    def __init__(
        self,
        database_url: str,
        storage_root: Path,
        model_name: str,
        confidence: float,
        classes: set[str],
        sample_rate: int,
    ):
        self.database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
        self.storage_root = storage_root
        self.model_name = model_name
        self.confidence = confidence
        self.classes = classes
        self.sample_rate = sample_rate

        self.model_manager = ModelManager()
        self.detector = DetectionService(model_mgr=self.model_manager)
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Connect to database."""
        self._pool = await asyncpg.create_pool(
            self.database_url, min_size=2, max_size=5, ssl=False
        )

    async def close(self) -> None:
        """Close connections and cleanup."""
        if self._pool:
            await self._pool.close()
        self.model_manager.unload_all()

    async def get_recordings_to_process(
        self,
        camera_id: Optional[int],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        reprocess: bool,
        limit: Optional[int],
    ) -> list[dict]:
        """Get list of recordings to process."""
        query = """
            SELECT r.id, r.camera_id, r.file_path, r.start_time, r.duration_seconds,
                   c.name as camera_name
            FROM recordings r
            JOIN cameras c ON r.camera_id = c.id
            WHERE r.status = 'completed'
        """
        params = []
        param_idx = 1

        if not reprocess:
            query += f" AND r.ml_processed = false"

        if camera_id:
            query += f" AND r.camera_id = ${param_idx}"
            params.append(camera_id)
            param_idx += 1

        if start_date:
            query += f" AND r.start_time >= ${param_idx}"
            params.append(start_date)
            param_idx += 1

        if end_date:
            query += f" AND r.start_time < ${param_idx}"
            params.append(end_date)
            param_idx += 1

        query += " ORDER BY r.start_time ASC"

        if limit:
            query += f" LIMIT ${param_idx}"
            params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]

    async def process_recording(self, recording: dict) -> dict:
        """Process a single recording and return results."""
        recording_id = recording["id"]
        db_path = recording["file_path"]

        # Handle absolute paths stored in database (e.g., /data/storage/...)
        # by replacing the Docker mount path with the actual storage root
        if db_path.startswith("/data/storage/"):
            relative_path = db_path[len("/data/storage/") :]
            file_path = self.storage_root / relative_path
        elif db_path.startswith("/"):
            # Other absolute path - use as-is
            file_path = Path(db_path)
        else:
            # Relative path - join with storage root
            file_path = self.storage_root / db_path

        if not file_path.exists():
            logger.warning(f"Recording {recording_id}: file not found: {file_path}")
            return {"error": "file_not_found", "detections": 0}

        # Open video
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            logger.warning(f"Recording {recording_id}: failed to open video")
            return {"error": "open_failed", "detections": 0}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(
            f"Recording {recording_id} ({recording['camera_name']}): "
            f"{total_frames} frames, {duration:.1f}s, processing every {self.sample_rate} frames"
        )

        detections: list[dict] = []
        frame_num = 0
        processed_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1

                # Skip frames based on sample rate
                if frame_num % self.sample_rate != 0:
                    continue

                processed_count += 1

                # Run detection
                results = self.detector.detect(
                    frame, self.model_name, confidence_threshold=self.confidence
                )

                # Filter by class
                filtered = [r for r in results if r.class_name.lower() in self.classes]

                # Record detections
                timestamp_ms = int((frame_num / fps) * 1000) if fps > 0 else 0

                for det in filtered:
                    detections.append({
                        "recording_id": recording_id,
                        "camera_id": recording["camera_id"],
                        "class_name": det.class_name,
                        "confidence": det.confidence,
                        "timestamp_ms": timestamp_ms,
                        "frame_number": frame_num,
                        "bbox_x": det.x,
                        "bbox_y": det.y,
                        "bbox_width": det.width,
                        "bbox_height": det.height,
                        "model_name": self.model_name,
                    })

                # Progress update
                if processed_count % 100 == 0:
                    pct = frame_num / total_frames * 100
                    logger.debug(f"  Progress: {pct:.1f}% ({frame_num}/{total_frames})")

        finally:
            cap.release()

        # Save detections to database
        if detections:
            await self._save_detections(detections)

        # Mark recording as processed
        await self._mark_processed(recording_id)

        return {"detections": len(detections), "frames_processed": processed_count}

    async def _save_detections(self, detections: list[dict]) -> None:
        """Save detections to database."""
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO detections (
                    recording_id, camera_id, class_name, confidence,
                    timestamp_ms, frame_number, bbox_x, bbox_y,
                    bbox_width, bbox_height, model_name, detected_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW()
                )
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

    async def _mark_processed(self, recording_id: int) -> None:
        """Mark recording as ML processed."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE recordings SET ml_processed = true WHERE id = $1",
                recording_id,
            )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process historical recordings through ML detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--camera-id",
        type=int,
        default=None,
        help="Only process recordings from this camera",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Process recordings from this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Process recordings until this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess recordings that have already been processed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of recordings to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without doing it",
    )
    parser.add_argument(
        "--sample-rate",
        "-s",
        type=int,
        default=15,
        help="Process every Nth frame (default: 15, ~2 fps for 30fps video)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.6,
        help="Minimum confidence threshold (default: 0.6)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="person,car,truck",
        help="Comma-separated list of classes to detect (default: person,car,truck)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse dates
    start_date = None
    end_date = None

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )

    # Parse classes
    classes = {c.strip().lower() for c in args.classes.split(",") if c.strip()}

    # Get settings
    settings = get_settings()
    storage_root = Path(settings.storage_root)

    processor = HistoricalProcessor(
        database_url=settings.database_url,
        storage_root=storage_root,
        model_name=settings.ml_default_model,
        confidence=args.confidence,
        classes=classes,
        sample_rate=args.sample_rate,
    )

    try:
        await processor.connect()

        # Get recordings to process
        recordings = await processor.get_recordings_to_process(
            camera_id=args.camera_id,
            start_date=start_date,
            end_date=end_date,
            reprocess=args.reprocess,
            limit=args.limit,
        )

        if not recordings:
            logger.info("No recordings to process")
            return

        logger.info(f"Found {len(recordings)} recordings to process")

        if args.dry_run:
            for r in recordings:
                print(
                    f"  {r['id']}: {r['camera_name']} - "
                    f"{r['start_time']} ({r['duration_seconds'] or 0}s)"
                )
            logger.info("Dry run complete - no recordings were processed")
            return

        # Process recordings
        total_detections = 0
        success_count = 0
        error_count = 0

        for i, recording in enumerate(recordings, 1):
            logger.info(f"[{i}/{len(recordings)}] Processing recording {recording['id']}...")

            try:
                result = await processor.process_recording(recording)

                if "error" in result:
                    error_count += 1
                else:
                    success_count += 1
                    total_detections += result["detections"]
                    logger.info(
                        f"  Found {result['detections']} detections "
                        f"in {result['frames_processed']} frames"
                    )

            except Exception as e:
                logger.error(f"  Error processing recording {recording['id']}: {e}")
                error_count += 1

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Processed: {success_count} recordings")
        logger.info(f"Errors: {error_count} recordings")
        logger.info(f"Total detections: {total_detections}")

    finally:
        await processor.close()


if __name__ == "__main__":
    asyncio.run(main())
