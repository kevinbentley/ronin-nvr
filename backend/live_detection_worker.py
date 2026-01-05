#!/usr/bin/env python3
"""Live detection worker for real-time object detection from camera streams.

This worker monitors HLS segments from active camera streams and runs YOLO
inference to detect objects in near real-time (2-5 second latency).

Key features:
- Single process handles multiple cameras (efficient at 1-2 fps)
- Taps existing HLS segments (no extra RTSP connections)
- Debounces notifications to prevent spam
- Saves snapshots with bounding boxes for previews
- Writes to unified detections table

Usage:
    ./live_detection_worker.py              # Run with defaults from .env
    ./live_detection_worker.py --fps 2.0    # 2 frames per second
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add the backend directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.config import Settings, get_settings
from app.services.ml.detection_service import DetectionResult, DetectionService
from app.services.ml.model_manager import ModelManager
from app.services.ml.motion_detector import MotionDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("live_detection")


@dataclass
class CameraState:
    """Per-camera state for live detection."""

    camera_id: int
    camera_name: str
    last_processed_segment: Optional[str] = None
    last_segment_mtime: float = 0.0
    enabled: bool = True


@dataclass
class DebounceTracker:
    """Track detection cooldowns per camera/class to prevent notification spam."""

    cooldown_seconds: float = 30.0
    _last_notified: dict[tuple[int, str], datetime] = field(default_factory=dict)

    def should_notify(self, camera_id: int, class_name: str) -> bool:
        """Check if enough time has passed since last notification."""
        key = (camera_id, class_name)
        last = self._last_notified.get(key)
        if last is None:
            return True
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return elapsed >= self.cooldown_seconds

    def mark_notified(self, camera_id: int, class_name: str) -> None:
        """Record notification time for debouncing."""
        self._last_notified[(camera_id, class_name)] = datetime.now(timezone.utc)


class SnapshotService:
    """Save detection snapshots with bounding boxes drawn."""

    def __init__(self, storage_root: Path, jpeg_quality: int = 85):
        self.storage_root = storage_root
        self.snapshots_dir = storage_root / ".snapshots"
        self.jpeg_quality = jpeg_quality

    def save_snapshot(
        self,
        frame: np.ndarray,
        camera_id: int,
        detections: list[DetectionResult],
        timestamp: datetime,
    ) -> Path:
        """Save frame with detection boxes drawn.

        Args:
            frame: BGR image
            camera_id: Camera that captured the frame
            detections: List of detections to draw
            timestamp: Detection timestamp

        Returns:
            Path to saved JPG file (relative to storage root)
        """
        annotated = self._draw_boxes(frame.copy(), detections)

        # Generate path: .snapshots/{camera_id}/{YYYY-MM-DD}/{HH-MM-SS-fff}.jpg
        date_str = timestamp.strftime("%Y-%m-%d")
        date_dir = self.snapshots_dir / str(camera_id) / date_str
        date_dir.mkdir(parents=True, exist_ok=True)

        filename = timestamp.strftime("%H-%M-%S-%f")[:-3] + ".jpg"
        snapshot_path = date_dir / filename

        cv2.imwrite(
            str(snapshot_path),
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )

        # Return relative path for database storage
        return snapshot_path.relative_to(self.storage_root)

    def _draw_boxes(
        self, frame: np.ndarray, detections: list[DetectionResult]
    ) -> np.ndarray:
        """Draw detection boxes and labels on frame."""
        h, w = frame.shape[:2]

        for det in detections:
            # Convert normalized coords to pixels
            x1 = int(det.x * w)
            y1 = int(det.y * h)
            x2 = int((det.x + det.width) * w)
            y2 = int((det.y + det.height) * h)

            # Draw box
            color = self._get_class_color(det.class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det.class_name} {det.confidence:.0%}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1
            )
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return frame

    def _get_class_color(self, class_name: str) -> tuple[int, int, int]:
        """Get BGR color for detection class."""
        colors = {
            "person": (0, 255, 0),  # Green
            "car": (255, 0, 0),  # Blue
            "truck": (255, 0, 0),
            "bus": (255, 0, 0),
            "motorcycle": (255, 165, 0),  # Orange
            "bicycle": (255, 165, 0),
            "dog": (0, 165, 255),  # Orange
            "cat": (0, 165, 255),
            "motion": (0, 128, 255),  # Orange (BGR) for motion
        }
        return colors.get(class_name, (0, 255, 255))  # Yellow default


class LiveDetectionWorker:
    """Single worker monitoring all active camera HLS streams."""

    def __init__(
        self,
        worker_id: str,
        database_url: str,
        storage_root: Path,
        settings: Settings,
    ):
        self.worker_id = worker_id
        self.database_url = database_url.replace(
            "postgresql+asyncpg://", "postgresql://"
        )
        self.storage_root = storage_root
        self.settings = settings

        # Configuration from environment
        self.fps = float(os.getenv("LIVE_DETECTION_FPS", "1.0"))
        self.cooldown = float(os.getenv("LIVE_DETECTION_COOLDOWN", "30.0"))
        self.confidence = float(os.getenv("LIVE_DETECTION_CONFIDENCE", "0.6"))
        self.model_name = settings.ml_default_model

        # Parse class filter
        class_str = os.getenv(
            "LIVE_DETECTION_CLASSES", "person,car,truck"
        )
        self.class_filter = set(c.strip().lower() for c in class_str.split(","))

        # State
        self.cameras: dict[int, CameraState] = {}
        self.debounce = DebounceTracker(cooldown_seconds=self.cooldown)
        self._running = False
        self._pool = None

        # Motion detection
        self._motion_enabled = settings.motion_detection_enabled
        self._motion_detectors: dict[int, MotionDetector] = {}

        # Initialize ML components
        self.model_manager = ModelManager()
        self.detector = DetectionService(model_mgr=self.model_manager)
        self.snapshot_service = SnapshotService(storage_root)

        # Stream directory
        self.streams_dir = storage_root / ".streams"

    async def start(self) -> None:
        """Start the worker and begin processing streams."""
        import asyncpg

        logger.info(f"Worker {self.worker_id} starting...")
        logger.info(
            f"Config: fps={self.fps}, cooldown={self.cooldown}s, "
            f"confidence={self.confidence}, classes={self.class_filter}"
        )

        # Connect to database
        # Disable SSL for internal Docker network connections
        self._pool = await asyncpg.create_pool(
            self.database_url, min_size=2, max_size=5, ssl=False
        )

        # Load active cameras from database
        await self._load_cameras()

        self._running = True
        logger.info(
            f"Worker {self.worker_id} ready, monitoring {len(self.cameras)} cameras"
        )

        try:
            await self._run_loop()
        except asyncio.CancelledError:
            logger.info(f"Worker {self.worker_id} cancelled")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        self._running = False

        if self._pool:
            await self._pool.close()
            self._pool = None

        self.model_manager.unload_all()
        logger.info(f"Worker {self.worker_id} stopped")

    async def _load_cameras(self) -> None:
        """Load active cameras from database, preserving existing state."""
        if not self._pool:
            return

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name FROM cameras
                WHERE recording_enabled = true
                """
            )

        # Build set of active camera IDs
        active_ids = {row["id"] for row in rows}

        # Remove cameras that are no longer active
        for cam_id in list(self.cameras.keys()):
            if cam_id not in active_ids:
                del self.cameras[cam_id]
                # Clean up motion detector for this camera
                if cam_id in self._motion_detectors:
                    del self._motion_detectors[cam_id]
                    logger.info(f"Removed motion detector for camera {cam_id}")

        # Add new cameras, preserving existing state
        for row in rows:
            if row["id"] not in self.cameras:
                self.cameras[row["id"]] = CameraState(
                    camera_id=row["id"],
                    camera_name=row["name"],
                )

        logger.info(f"Loaded {len(self.cameras)} active cameras")

    def _get_motion_detector(self, camera_id: int) -> MotionDetector:
        """Get or create motion detector for a camera.

        Each camera needs its own detector to maintain background model state.
        """
        if camera_id not in self._motion_detectors:
            self._motion_detectors[camera_id] = MotionDetector.from_settings()
            logger.info(f"Created motion detector for camera {camera_id}")
        return self._motion_detectors[camera_id]

    async def _run_loop(self) -> None:
        """Main worker loop: round-robin through cameras."""
        cycle_time = 1.0 / self.fps if self.fps > 0 else 1.0

        while self._running:
            cycle_start = asyncio.get_event_loop().time()

            # Refresh camera list periodically (every 60 seconds)
            if int(cycle_start) % 60 == 0:
                await self._load_cameras()

            # Process each camera
            for camera_id, state in list(self.cameras.items()):
                if not state.enabled:
                    continue

                try:
                    await self._process_camera(state)
                except Exception as e:
                    logger.error(f"Error processing camera {camera_id}: {e}")

            # Sleep to maintain target FPS
            elapsed = asyncio.get_event_loop().time() - cycle_start
            sleep_time = max(0, cycle_time - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def _process_camera(self, state: CameraState) -> None:
        """Process the latest HLS segment for a camera."""
        camera_dir = self.streams_dir / str(state.camera_id)
        if not camera_dir.exists():
            logger.debug(f"Camera {state.camera_id}: directory doesn't exist")
            return

        # Find the newest segment - handle race condition where segments may be
        # deleted by FFmpeg between glob() and stat() calls
        try:
            segments = []
            for p in camera_dir.glob("segment*.ts"):
                try:
                    segments.append((p, p.stat().st_mtime))
                except FileNotFoundError:
                    # Segment was deleted between glob and stat - skip it
                    continue
            if not segments:
                return
            # Sort by mtime and get the newest
            segments.sort(key=lambda x: x[1])
            newest, newest_mtime = segments[-1]
        except OSError:
            # Directory was deleted or other filesystem error
            return

        # Skip if we already processed this segment
        if (
            state.last_processed_segment == newest.name
            and state.last_segment_mtime == newest_mtime
        ):
            return

        # Wait for segment to be fully written (at least 1 second old)
        age = time.time() - newest_mtime
        if age < 1.0:
            return

        logger.debug(f"Camera {state.camera_id}: processing {newest.name}")

        # Process the segment - pass mtime to avoid additional stat() call
        await self._process_segment(state, newest, newest_mtime)

        state.last_processed_segment = newest.name
        state.last_segment_mtime = newest_mtime

    async def _process_segment(
        self, state: CameraState, segment_path: Path, segment_mtime: float
    ) -> None:
        """Extract frame from segment and run detection.

        Args:
            state: Camera state for this camera
            segment_path: Path to the HLS segment file
            segment_mtime: Modification time of the segment (avoids race condition)
        """
        # Extract a frame from the segment
        frame = await self._extract_frame(segment_path)
        if frame is None:
            logger.debug(f"Camera {state.camera_id}: frame extraction failed")
            return

        logger.debug(f"Camera {state.camera_id}: extracted frame {frame.shape}")

        # Use segment modification time as detection timestamp
        # This is closer to actual video capture time than current time.
        detection_time = datetime.fromtimestamp(segment_mtime, tz=timezone.utc)
        detections_to_save = []
        all_detections: list[DetectionResult] = []
        snapshot_path: Optional[Path] = None

        # Run motion detection (if enabled)
        if self._motion_enabled:
            motion_detector = self._get_motion_detector(state.camera_id)
            motion_result = motion_detector.detect(frame)

            if motion_result.has_motion:
                logger.debug(
                    f"Camera {state.camera_id}: motion detected "
                    f"({motion_result.motion_percent:.1f}%)"
                )

                if self.debounce.should_notify(state.camera_id, "motion"):
                    # Use largest bounding box or full frame if none
                    if motion_result.bounding_boxes:
                        bbox = motion_result.bounding_boxes[0]
                    else:
                        bbox = (0.0, 0.0, 1.0, 1.0)

                    # Create DetectionResult for snapshot drawing
                    motion_det = DetectionResult(
                        class_name="motion",
                        confidence=min(motion_result.motion_percent / 100.0, 1.0),
                        x=bbox[0],
                        y=bbox[1],
                        width=bbox[2],
                        height=bbox[3],
                    )
                    all_detections.append(motion_det)

                    detections_to_save.append({
                        "camera_id": state.camera_id,
                        "class_name": "motion",
                        "confidence": motion_det.confidence,
                        "bbox_x": motion_det.x,
                        "bbox_y": motion_det.y,
                        "bbox_width": motion_det.width,
                        "bbox_height": motion_det.height,
                        "model_name": "motion_detector",
                        "detected_at": detection_time,
                        "snapshot_path": None,  # Will be set later
                    })
                    self.debounce.mark_notified(state.camera_id, "motion")

        # Run YOLO inference
        results = self.detector.detect(
            frame, self.model_name, confidence_threshold=self.confidence
        )

        logger.debug(
            f"Camera {state.camera_id}: {len(results)} raw detections"
        )

        # Filter by class
        filtered = [
            r for r in results if r.class_name.lower() in self.class_filter
        ]

        logger.debug(
            f"Camera {state.camera_id}: {len(filtered)} after class filter"
        )

        for det in filtered:
            if self.debounce.should_notify(state.camera_id, det.class_name):
                all_detections.append(det)

                detections_to_save.append({
                    "camera_id": state.camera_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "bbox_x": det.x,
                    "bbox_y": det.y,
                    "bbox_width": det.width,
                    "bbox_height": det.height,
                    "model_name": self.model_name,
                    "detected_at": detection_time,
                    "snapshot_path": None,  # Will be set later
                })
                self.debounce.mark_notified(state.camera_id, det.class_name)

        # Save snapshot and update paths if we have detections
        if detections_to_save:
            snapshot_path = self.snapshot_service.save_snapshot(
                frame, state.camera_id, all_detections, detection_time
            )
            logger.info(
                f"Detection on {state.camera_name}: "
                f"{', '.join(d.class_name for d in all_detections)}"
            )

            # Update snapshot path for all detections
            for det in detections_to_save:
                det["snapshot_path"] = str(snapshot_path)

            await self._save_detections(detections_to_save, state.camera_name)

    async def _extract_frame(self, segment_path: Path) -> Optional[np.ndarray]:
        """Extract a single frame from an HLS .ts segment using FFmpeg."""
        try:
            # Extract the first frame from the segment (more reliable than
            # selecting a specific frame number which may not exist at low fps)
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-i", str(segment_path),
                "-frames:v", "1",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "pipe:1",
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=10.0
            )

            if proc.returncode != 0:
                logger.debug(f"FFmpeg failed: {stderr.decode()[:200]}")
                return None

            if not stdout:
                logger.debug("FFmpeg returned no data")
                return None

            # Get frame dimensions from segment
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                str(segment_path),
            ]

            probe_proc = await asyncio.create_subprocess_exec(
                *probe_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            probe_out, probe_err = await probe_proc.communicate()
            if not probe_out:
                logger.debug(f"FFprobe returned no data: {probe_err.decode()[:200]}")
                return None

            # Parse dimensions - take first valid line (handles multiple stream output)
            lines = [l.strip() for l in probe_out.decode().strip().split("\n") if l.strip()]
            if not lines:
                logger.debug("FFprobe returned empty output")
                return None

            dims = lines[0].split(",")
            if len(dims) != 2:
                logger.debug(f"FFprobe unexpected format: {lines[0][:100]}")
                return None

            width, height = int(dims[0]), int(dims[1])
            expected_size = width * height * 3

            if len(stdout) != expected_size:
                return None

            frame = np.frombuffer(stdout, dtype=np.uint8).reshape((height, width, 3))
            return frame

        except asyncio.TimeoutError:
            logger.warning(f"Timeout extracting frame from {segment_path}")
            return None
        except Exception as e:
            logger.error(f"Error extracting frame: {e}")
            return None

    async def _save_detections(
        self, detections: list[dict], camera_name: str
    ) -> None:
        """Save detections to database and notify listeners."""
        if not self._pool or not detections:
            return

        async with self._pool.acquire() as conn:
            # Insert detections
            await conn.executemany(
                """
                INSERT INTO detections (
                    recording_id, camera_id, class_name, confidence,
                    timestamp_ms, frame_number, bbox_x, bbox_y,
                    bbox_width, bbox_height, model_name, detected_at,
                    snapshot_path
                ) VALUES (
                    NULL, $1, $2, $3, 0, 0, $4, $5, $6, $7, $8, $9, $10
                )
                """,
                [
                    (
                        d["camera_id"],
                        d["class_name"],
                        d["confidence"],
                        d["bbox_x"],
                        d["bbox_y"],
                        d["bbox_width"],
                        d["bbox_height"],
                        d["model_name"],
                        d["detected_at"],
                        d["snapshot_path"],
                    )
                    for d in detections
                ],
            )

            # Notify listeners for real-time alerts
            for d in detections:
                payload = json.dumps({
                    "camera_id": d["camera_id"],
                    "camera_name": camera_name,
                    "class_name": d["class_name"],
                    "confidence": d["confidence"],
                    "snapshot_path": d["snapshot_path"],
                    "detected_at": d["detected_at"].isoformat(),
                })
                await conn.execute(
                    "SELECT pg_notify('live_detection', $1)", payload
                )


def main() -> None:
    """Main entry point for live detection worker."""
    parser = argparse.ArgumentParser(
        description="Live detection worker for real-time object detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target frames per second (default: from LIVE_DETECTION_FPS or 1.0)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=None,
        help="Seconds between same-class notifications (default: 30.0)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Detection confidence threshold (default: 0.6)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Override environment from args if provided
    if args.fps is not None:
        os.environ["LIVE_DETECTION_FPS"] = str(args.fps)
    if args.cooldown is not None:
        os.environ["LIVE_DETECTION_COOLDOWN"] = str(args.cooldown)
    if args.confidence is not None:
        os.environ["LIVE_DETECTION_CONFIDENCE"] = str(args.confidence)

    # Load settings
    settings = get_settings()

    # Check if enabled
    if os.getenv("LIVE_DETECTION_ENABLED", "true").lower() == "false":
        logger.info("Live detection is disabled (LIVE_DETECTION_ENABLED=false)")
        return

    # Validate database URL
    if not settings.database_url:
        logger.error("No database URL provided. Set DATABASE_URL")
        sys.exit(1)

    # Generate worker ID
    hostname = socket.gethostname()[:16]
    pid = os.getpid()
    worker_id = f"live-{hostname}-{pid}"

    logger.info(f"Starting live detection worker: {worker_id}")

    # Create worker
    worker = LiveDetectionWorker(
        worker_id=worker_id,
        database_url=settings.database_url,
        storage_root=Path(settings.storage_root),
        settings=settings,
    )

    # Handle shutdown signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    shutdown_event = asyncio.Event()

    def signal_handler(sig: int, frame: object) -> None:
        logger.info(f"Received signal {sig}, shutting down...")
        loop.call_soon_threadsafe(shutdown_event.set)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    async def run_with_shutdown() -> None:
        worker_task = asyncio.create_task(worker.start())
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        done, pending = await asyncio.wait(
            [worker_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

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

    logger.info("Live detection worker stopped")


if __name__ == "__main__":
    main()
