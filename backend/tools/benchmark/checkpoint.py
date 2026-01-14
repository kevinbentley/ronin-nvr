"""Checkpoint management for resumable benchmark execution."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .models import (
    BenchmarkResult,
    CandidateEvent,
    Detection,
    DetectionMethod,
    EventType,
    MethodMetrics,
    VideoInfo,
    VLMLabel,
)

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving and loading benchmark checkpoints for resumable execution."""

    def __init__(self, config: BenchmarkConfig, run_id: str):
        """Initialize checkpoint manager.

        Args:
            config: Benchmark configuration
            run_id: Unique identifier for this benchmark run
        """
        self.config = config
        self.run_id = run_id
        self.checkpoint_path = config.output_dir / f"checkpoint_{run_id}.json"
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: BenchmarkResult, processed_paths: set[str]) -> None:
        """Save current benchmark state to checkpoint file.

        Args:
            result: Current benchmark result state
            processed_paths: Set of video paths that have been processed
        """
        checkpoint_data = {
            "version": "1.0",
            "run_id": self.run_id,
            "saved_at": datetime.now().isoformat(),
            "processed_paths": list(processed_paths),
            "result": result.to_dict(),
            "config": self.config.to_dict(),
        }

        # Write to temp file first, then rename for atomicity
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        temp_path.rename(self.checkpoint_path)
        logger.info(f"Checkpoint saved: {len(processed_paths)} videos processed")

    def load(self) -> tuple[BenchmarkResult, set[str]] | None:
        """Load benchmark state from checkpoint file.

        Returns:
            Tuple of (BenchmarkResult, processed_paths) if checkpoint exists,
            None otherwise
        """
        if not self.checkpoint_path.exists():
            logger.info("No checkpoint found, starting fresh")
            return None

        try:
            with open(self.checkpoint_path) as f:
                data = json.load(f)

            # Verify run_id matches
            if data.get("run_id") != self.run_id:
                logger.warning(
                    f"Checkpoint run_id mismatch: {data.get('run_id')} != {self.run_id}"
                )
                return None

            processed_paths = set(data.get("processed_paths", []))
            result = self._deserialize_result(data.get("result", {}))

            logger.info(
                f"Checkpoint loaded: {len(processed_paths)} videos already processed"
            )
            return result, processed_paths

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def _deserialize_result(self, data: dict[str, Any]) -> BenchmarkResult:
        """Deserialize BenchmarkResult from dictionary."""
        # Deserialize videos
        videos = [
            self._deserialize_video(v) for v in data.get("videos_processed", [])
        ]

        # Deserialize candidate events
        events = [
            self._deserialize_event(e, videos)
            for e in data.get("candidate_events", [])
        ]

        # Deserialize method metrics
        metrics = {}
        for method_str, metric_data in data.get("method_metrics", {}).items():
            method = DetectionMethod(method_str)
            metrics[method] = MethodMetrics(
                method=method,
                true_positives=metric_data.get("true_positives", 0),
                false_positives=metric_data.get("false_positives", 0),
                false_negatives=metric_data.get("false_negatives", 0),
                total_detections=metric_data.get("total_detections", 0),
                total_frames_processed=metric_data.get("total_frames_processed", 0),
                processing_time_seconds=metric_data.get("processing_time_seconds", 0.0),
            )

        return BenchmarkResult(
            run_id=data.get("run_id", self.run_id),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=(
                datetime.fromisoformat(data["end_time"])
                if data.get("end_time")
                else None
            ),
            videos_processed=videos,
            candidate_events=events,
            method_metrics=metrics,
            config_snapshot=data.get("config_snapshot", {}),
        )

    def _deserialize_video(self, data: dict[str, Any]) -> VideoInfo:
        """Deserialize VideoInfo from dictionary."""
        return VideoInfo(
            path=Path(data["path"]),
            camera_id=data["camera_id"],
            date_str=data["date_str"],
            time_str=data["time_str"],
            timestamp_utc=datetime.fromisoformat(data["timestamp_utc"]),
            file_size_mb=data["file_size_mb"],
            duration_seconds=data.get("duration_seconds"),
            frame_count=data.get("frame_count"),
            fps=data.get("fps"),
            width=data.get("width"),
            height=data.get("height"),
        )

    def _deserialize_event(
        self, data: dict[str, Any], videos: list[VideoInfo]
    ) -> CandidateEvent:
        """Deserialize CandidateEvent from dictionary."""
        # Find the matching video
        video_path = data["video_path"]
        video = next(
            (v for v in videos if str(v.path) == video_path),
            None,
        )

        # If video not found, create a minimal VideoInfo
        if video is None:
            video = VideoInfo(
                path=Path(video_path),
                camera_id="unknown",
                date_str="unknown",
                time_str="unknown",
                timestamp_utc=datetime.now(),
                file_size_mb=0.0,
            )

        # Deserialize detections
        detections = [
            Detection(
                method=DetectionMethod(d["method"]),
                event_type=EventType(d["event_type"]),
                frame_number=d["frame_number"],
                timestamp_seconds=d["timestamp_seconds"],
                confidence=d["confidence"],
                bbox=tuple(d["bbox"]) if d.get("bbox") else None,
                metadata=d.get("metadata", {}),
            )
            for d in data.get("detections", [])
        ]

        return CandidateEvent(
            video=video,
            frame_number=data["frame_number"],
            timestamp_seconds=data["timestamp_seconds"],
            detections=detections,
            frame_path=Path(data["frame_path"]) if data.get("frame_path") else None,
            vlm_label=VLMLabel(data["vlm_label"]) if data.get("vlm_label") else None,
            vlm_response=data.get("vlm_response"),
            vlm_detected_objects=data.get("vlm_detected_objects", []),
            manually_verified=data.get("manually_verified", False),
            manual_label=(
                VLMLabel(data["manual_label"]) if data.get("manual_label") else None
            ),
        )

    def delete(self) -> None:
        """Delete checkpoint file after successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info(f"Checkpoint deleted: {self.checkpoint_path}")

    def exists(self) -> bool:
        """Check if a checkpoint file exists."""
        return self.checkpoint_path.exists()

    @classmethod
    def find_checkpoints(cls, output_dir: Path) -> list[Path]:
        """Find all checkpoint files in the output directory.

        Args:
            output_dir: Directory to search for checkpoints

        Returns:
            List of checkpoint file paths
        """
        if not output_dir.exists():
            return []
        return sorted(output_dir.glob("checkpoint_*.json"))

    @classmethod
    def get_checkpoint_info(cls, checkpoint_path: Path) -> dict[str, Any] | None:
        """Get basic info from a checkpoint file without fully loading it.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary with run_id, saved_at, and videos_processed count
        """
        try:
            with open(checkpoint_path) as f:
                data = json.load(f)

            return {
                "run_id": data.get("run_id"),
                "saved_at": data.get("saved_at"),
                "videos_processed": len(data.get("processed_paths", [])),
                "path": str(checkpoint_path),
            }
        except (json.JSONDecodeError, OSError):
            return None
