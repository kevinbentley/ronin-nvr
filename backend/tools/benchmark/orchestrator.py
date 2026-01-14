"""Orchestrator for running detectors on video frames."""

import logging
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

from .config import BenchmarkConfig
from .detectors import BaseDetector, DetectorFactory
from .models import CandidateEvent, Detection, DetectionMethod, VideoInfo

logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """Extracts frames from video files at specified intervals."""

    def __init__(self, sample_fps: float = 1.0, max_frames: int = 300):
        """Initialize frame extractor.

        Args:
            sample_fps: Target frames per second to extract
            max_frames: Maximum frames to extract per video
        """
        self.sample_fps = sample_fps
        self.max_frames = max_frames

    def extract_frames(
        self,
        video_path: Path,
    ) -> Generator[tuple[int, float, np.ndarray], None, None]:
        """Extract frames from video at specified sample rate.

        Args:
            video_path: Path to video file

        Yields:
            Tuples of (frame_number, timestamp_seconds, frame)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if video_fps <= 0:
                video_fps = 30.0  # Default assumption

            # Calculate frame skip interval
            frame_interval = max(1, int(video_fps / self.sample_fps))

            frames_yielded = 0
            frame_number = 0

            while frames_yielded < self.max_frames:
                # Seek to next sample frame
                target_frame = frame_number * frame_interval
                if target_frame >= total_frames:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()

                if not ret:
                    break

                timestamp = target_frame / video_fps
                yield target_frame, timestamp, frame

                frames_yielded += 1
                frame_number += 1

        finally:
            cap.release()


class DetectionOrchestrator:
    """Orchestrates running all detectors on video frames."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize orchestrator.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.detectors: dict[DetectionMethod, BaseDetector] = {}
        self.frame_extractor = VideoFrameExtractor(
            sample_fps=config.sample_fps,
            max_frames=config.max_frames_per_video,
        )
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all detectors."""
        if self._initialized:
            return

        logger.info("Initializing detectors...")
        self.detectors = DetectorFactory.create_all(self.config)

        if not self.detectors:
            raise RuntimeError("No detectors could be initialized")

        logger.info(f"Initialized {len(self.detectors)} detectors: "
                   f"{[m.value for m in self.detectors.keys()]}")
        self._initialized = True

    def process_video(
        self,
        video: VideoInfo,
    ) -> list[CandidateEvent]:
        """Process a single video with all detectors.

        Args:
            video: Video information

        Returns:
            List of candidate events detected
        """
        self.initialize()

        # Reset all detectors for new video
        for detector in self.detectors.values():
            detector.reset()

        logger.info(f"Processing: {video.path.name}")

        # Track detections by frame for event consolidation
        frame_detections: dict[int, list[Detection]] = {}
        previous_frame: np.ndarray | None = None

        frames_processed = 0
        for frame_number, timestamp, frame in self.frame_extractor.extract_frames(video.path):
            # Run all detectors on this frame
            all_detections: list[Detection] = []

            for method, detector in self.detectors.items():
                try:
                    detections = detector.detect(
                        frame=frame,
                        frame_number=frame_number,
                        timestamp_seconds=timestamp,
                        previous_frame=previous_frame,
                    )
                    all_detections.extend(detections)
                except Exception as e:
                    logger.warning(f"Detector {method.value} failed on frame {frame_number}: {e}")

            if all_detections:
                frame_detections[frame_number] = all_detections
                # Log detection details
                for det in all_detections:
                    logger.debug(
                        f"  Frame {frame_number} ({timestamp:.1f}s): "
                        f"{det.method.value} -> {det.event_type.value} "
                        f"(conf={det.confidence:.2f})"
                    )

            previous_frame = frame.copy()
            frames_processed += 1

            # Progress logging every 50 frames
            if frames_processed % 50 == 0:
                logger.debug(f"  Progress: {frames_processed} frames processed")

        logger.info(f"  Processed {frames_processed} frames, "
                   f"found detections in {len(frame_detections)} frames")

        # Convert frame detections to candidate events
        events = self._consolidate_events(video, frame_detections)

        return events

    def _consolidate_events(
        self,
        video: VideoInfo,
        frame_detections: dict[int, list[Detection]],
    ) -> list[CandidateEvent]:
        """Consolidate frame-level detections into candidate events.

        Applies cooldown to avoid creating too many events for continuous activity.

        Args:
            video: Video information
            frame_detections: Dictionary mapping frame numbers to detections

        Returns:
            List of consolidated candidate events
        """
        if not frame_detections:
            return []

        events: list[CandidateEvent] = []
        sorted_frames = sorted(frame_detections.keys())

        last_event_time: float = -float("inf")
        cooldown = self.config.event_cooldown_seconds

        for frame_number in sorted_frames:
            detections = frame_detections[frame_number]

            if len(detections) < self.config.min_detections_for_event:
                continue

            # Get timestamp from first detection
            timestamp = detections[0].timestamp_seconds

            # Apply cooldown
            if timestamp - last_event_time < cooldown:
                continue

            # Create candidate event
            event = CandidateEvent(
                video=video,
                frame_number=frame_number,
                timestamp_seconds=timestamp,
                detections=detections,
            )
            events.append(event)
            last_event_time = timestamp

        logger.info(f"  Consolidated to {len(events)} candidate events")
        return events

    def extract_event_frame(
        self,
        event: CandidateEvent,
        output_dir: Path,
    ) -> Path | None:
        """Extract and save the frame for a candidate event.

        Args:
            event: Candidate event
            output_dir: Directory to save frame images

        Returns:
            Path to saved frame image, or None if extraction failed
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(event.video.path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {event.video.path}")
            return None

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, event.frame_number)
            ret, frame = cap.read()

            if not ret:
                logger.error(f"Could not read frame {event.frame_number}")
                return None

            # Generate unique filename
            video_stem = event.video.path.stem
            frame_path = output_dir / f"{video_stem}_frame{event.frame_number}.jpg"

            # Draw bounding boxes for detections
            annotated = self._annotate_frame(frame, event.detections)

            cv2.imwrite(str(frame_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            event.frame_path = frame_path

            return frame_path

        finally:
            cap.release()

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """Annotate frame with detection bounding boxes.

        Args:
            frame: Frame to annotate
            detections: List of detections to draw

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Color map for different methods
        colors = {
            DetectionMethod.YOLOV8N: (0, 255, 0),     # Green
            DetectionMethod.YOLO11L: (0, 255, 255),   # Yellow
            DetectionMethod.MOG2: (255, 0, 0),        # Blue
            DetectionMethod.FRAME_DIFF: (255, 0, 255), # Magenta
            DetectionMethod.EDGE_DETECTION: (0, 165, 255),  # Orange
            DetectionMethod.CORRUPTION: (0, 0, 255),  # Red
        }

        for det in detections:
            color = colors.get(det.method, (128, 128, 128))

            if det.bbox:
                x, y, w, h = det.bbox
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

                # Label
                label = f"{det.method.value}: {det.event_type.value} ({det.confidence:.2f})"
                cv2.putText(
                    annotated, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

        return annotated

    def get_processing_stats(self) -> dict[DetectionMethod, dict]:
        """Get processing statistics for all detectors.

        Returns:
            Dictionary of stats per detector method
        """
        stats = {}
        for method, detector in self.detectors.items():
            fps = (
                detector.frames_processed / detector.processing_time
                if detector.processing_time > 0
                else 0
            )
            stats[method] = {
                "frames_processed": detector.frames_processed,
                "processing_time_seconds": detector.processing_time,
                "fps": fps,
            }
        return stats
