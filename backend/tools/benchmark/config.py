"""Configuration for the detection benchmark framework."""

from dataclasses import dataclass, field
from pathlib import Path

from .models import DetectionMethod


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Storage paths
    storage_root: Path = field(default_factory=lambda: Path("/opt3/ronin/storage"))
    output_dir: Path = field(
        default_factory=lambda: Path("/home/kbentley/dev/ronin-nvr/backend/benchmark_results")
    )
    frame_cache_dir: Path = field(
        default_factory=lambda: Path("/tmp/benchmark_frames")
    )

    # Video selection
    target_total_videos: int = 50
    min_file_size_mb: float = 1.0  # Skip very small/corrupt files
    max_file_size_mb: float = 500.0  # Skip unusually large files

    # Daytime filtering (UTC hours) - Idaho MST = UTC-7 in January
    # Local 8AM-6PM = UTC 15:00-01:00 (next day)
    daytime_utc_ranges: list[tuple[int, int]] = field(
        default_factory=lambda: [(15, 23), (0, 1)]
    )

    # Frame sampling
    sample_fps: float = 1.0  # Sample 1 frame per second
    max_frames_per_video: int = 300  # Cap at 5 minutes worth at 1fps

    # Detection thresholds
    yolo_confidence: float = 0.25
    mog2_history: int = 500
    mog2_var_threshold: float = 16.0
    mog2_detect_shadows: bool = False
    mog2_min_area_percent: float = 0.05  # Minimum motion area as % of frame
    frame_diff_threshold: int = 30
    frame_diff_min_area_percent: float = 0.05
    edge_change_threshold: float = 0.1  # 10% edge pixel change
    corruption_streak_threshold: int = 50  # Horizontal streak detection

    # Model paths
    yolov8n_path: Path = field(
        default_factory=lambda: Path("/opt3/ronin/ml_models/yolov8n_dynamic.onnx")
    )
    yolo11l_path: Path = field(
        default_factory=lambda: Path("/opt3/ronin/ml_models/yolo11l_dynamic.onnx")
    )

    # VLM configuration
    vlm_endpoint: str = "http://192.168.1.125:9001/v1/chat/completions"
    vlm_model: str = "default"  # Model name for API
    vlm_timeout: float = 30.0
    vlm_max_retries: int = 3
    vlm_retry_delay: float = 2.0

    # Event detection
    event_cooldown_seconds: float = 5.0  # Minimum time between events
    min_detections_for_event: int = 1  # Minimum detections to trigger event

    # Methods to run
    enabled_methods: list[DetectionMethod] = field(
        default_factory=lambda: [
            DetectionMethod.YOLOV8N,
            DetectionMethod.YOLO11L,
            DetectionMethod.MOG2,
            DetectionMethod.FRAME_DIFF,
            DetectionMethod.EDGE_DETECTION,
            DetectionMethod.CORRUPTION,
        ]
    )

    # Checkpoint configuration
    checkpoint_interval: int = 10  # Save checkpoint every N videos
    checkpoint_file: Path | None = None  # Set automatically based on run_id

    # Logging
    verbose: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Convert string paths to Path objects if needed."""
        if isinstance(self.storage_root, str):
            self.storage_root = Path(self.storage_root)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.frame_cache_dir, str):
            self.frame_cache_dir = Path(self.frame_cache_dir)
        if isinstance(self.yolov8n_path, str):
            self.yolov8n_path = Path(self.yolov8n_path)
        if isinstance(self.yolo11l_path, str):
            self.yolo11l_path = Path(self.yolo11l_path)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "storage_root": str(self.storage_root),
            "output_dir": str(self.output_dir),
            "frame_cache_dir": str(self.frame_cache_dir),
            "target_total_videos": self.target_total_videos,
            "min_file_size_mb": self.min_file_size_mb,
            "max_file_size_mb": self.max_file_size_mb,
            "daytime_utc_ranges": self.daytime_utc_ranges,
            "sample_fps": self.sample_fps,
            "max_frames_per_video": self.max_frames_per_video,
            "yolo_confidence": self.yolo_confidence,
            "mog2_history": self.mog2_history,
            "mog2_var_threshold": self.mog2_var_threshold,
            "mog2_detect_shadows": self.mog2_detect_shadows,
            "mog2_min_area_percent": self.mog2_min_area_percent,
            "frame_diff_threshold": self.frame_diff_threshold,
            "frame_diff_min_area_percent": self.frame_diff_min_area_percent,
            "edge_change_threshold": self.edge_change_threshold,
            "corruption_streak_threshold": self.corruption_streak_threshold,
            "yolov8n_path": str(self.yolov8n_path),
            "yolo11l_path": str(self.yolo11l_path),
            "vlm_endpoint": self.vlm_endpoint,
            "vlm_model": self.vlm_model,
            "vlm_timeout": self.vlm_timeout,
            "vlm_max_retries": self.vlm_max_retries,
            "vlm_retry_delay": self.vlm_retry_delay,
            "event_cooldown_seconds": self.event_cooldown_seconds,
            "min_detections_for_event": self.min_detections_for_event,
            "enabled_methods": [m.value for m in self.enabled_methods],
            "checkpoint_interval": self.checkpoint_interval,
            "verbose": self.verbose,
            "log_level": self.log_level,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkConfig":
        """Create config from dictionary."""
        # Convert method strings back to enums
        if "enabled_methods" in data:
            data["enabled_methods"] = [
                DetectionMethod(m) for m in data["enabled_methods"]
            ]
        return cls(**data)


# Default configuration instance
DEFAULT_CONFIG = BenchmarkConfig()
