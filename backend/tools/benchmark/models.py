"""Data models for the detection benchmark framework."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class DetectionMethod(Enum):
    """Available detection methods."""

    YOLOV8N = "yolov8n"
    YOLO11L = "yolo11l"
    MOG2 = "mog2"
    FRAME_DIFF = "frame_diff"
    EDGE_DETECTION = "edge_detection"
    CORRUPTION = "corruption"


class EventType(Enum):
    """Types of detected events."""

    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    CORRUPT_IMAGE = "corrupt_image"
    MOTION = "motion"  # Generic motion for non-ML methods
    UNKNOWN = "unknown"


class VLMLabel(Enum):
    """Ground truth labels from VLM."""

    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    UNCERTAIN = "uncertain"
    ERROR = "error"


@dataclass
class VideoInfo:
    """Information about a video file."""

    path: Path
    camera_id: str
    date_str: str  # YYYY-MM-DD
    time_str: str  # HH-MM-SS
    timestamp_utc: datetime
    file_size_mb: float
    duration_seconds: float | None = None
    frame_count: int | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None

    @property
    def filename(self) -> str:
        """Return the filename without path."""
        return self.path.name

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": str(self.path),
            "camera_id": self.camera_id,
            "date_str": self.date_str,
            "time_str": self.time_str,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "file_size_mb": self.file_size_mb,
            "duration_seconds": self.duration_seconds,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class Detection:
    """A single detection from a detection method."""

    method: DetectionMethod
    event_type: EventType
    frame_number: int
    timestamp_seconds: float
    confidence: float
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "method": self.method.value,
            "event_type": self.event_type.value,
            "frame_number": self.frame_number,
            "timestamp_seconds": self.timestamp_seconds,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "metadata": self.metadata,
        }


@dataclass
class CandidateEvent:
    """A candidate event to be verified by VLM."""

    video: VideoInfo
    frame_number: int
    timestamp_seconds: float
    detections: list[Detection]
    frame_path: Path | None = None  # Path to extracted frame image
    vlm_label: VLMLabel | None = None
    vlm_response: str | None = None
    vlm_detected_objects: list[str] = field(default_factory=list)
    manually_verified: bool = False
    manual_label: VLMLabel | None = None

    @property
    def methods_that_detected(self) -> set[DetectionMethod]:
        """Return set of methods that detected this event."""
        return {d.method for d in self.detections}

    @property
    def event_types_detected(self) -> set[EventType]:
        """Return set of event types detected."""
        return {d.event_type for d in self.detections}

    @property
    def ground_truth_label(self) -> VLMLabel | None:
        """Return the final ground truth label (manual overrides VLM)."""
        if self.manually_verified and self.manual_label is not None:
            return self.manual_label
        return self.vlm_label

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_path": str(self.video.path),
            "frame_number": self.frame_number,
            "timestamp_seconds": self.timestamp_seconds,
            "detections": [d.to_dict() for d in self.detections],
            "frame_path": str(self.frame_path) if self.frame_path else None,
            "vlm_label": self.vlm_label.value if self.vlm_label else None,
            "vlm_response": self.vlm_response,
            "vlm_detected_objects": self.vlm_detected_objects,
            "manually_verified": self.manually_verified,
            "manual_label": self.manual_label.value if self.manual_label else None,
        }


@dataclass
class MethodMetrics:
    """Performance metrics for a single detection method."""

    method: DetectionMethod
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_detections: int = 0
    total_frames_processed: int = 0
    processing_time_seconds: float = 0.0

    @property
    def precision(self) -> float:
        """Calculate precision (TP / (TP + FP))."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Calculate recall (TP / (TP + FN))."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """Calculate F1 score (harmonic mean of precision and recall)."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def fps(self) -> float:
        """Calculate frames per second processing rate."""
        if self.processing_time_seconds == 0:
            return 0.0
        return self.total_frames_processed / self.processing_time_seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "method": self.method.value,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_detections": self.total_detections,
            "total_frames_processed": self.total_frames_processed,
            "processing_time_seconds": self.processing_time_seconds,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "fps": self.fps,
        }


@dataclass
class BenchmarkResult:
    """Complete results from a benchmark run."""

    run_id: str
    start_time: datetime
    end_time: datetime | None = None
    videos_processed: list[VideoInfo] = field(default_factory=list)
    candidate_events: list[CandidateEvent] = field(default_factory=list)
    method_metrics: dict[DetectionMethod, MethodMetrics] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def total_videos(self) -> int:
        """Return total number of videos processed."""
        return len(self.videos_processed)

    @property
    def total_events(self) -> int:
        """Return total number of candidate events."""
        return len(self.candidate_events)

    @property
    def duration_seconds(self) -> float | None:
        """Return total benchmark duration in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "total_videos": self.total_videos,
            "total_events": self.total_events,
            "videos_processed": [v.to_dict() for v in self.videos_processed],
            "candidate_events": [e.to_dict() for e in self.candidate_events],
            "method_metrics": {
                k.value: v.to_dict() for k, v in self.method_metrics.items()
            },
            "config_snapshot": self.config_snapshot,
        }
