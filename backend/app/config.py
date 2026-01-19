"""Application configuration using pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str = "postgresql+asyncpg://localhost:5432/ronin_nvr"

    # Storage
    storage_root: Path = Path("./storage")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Application
    app_name: str = "RoninNVR"
    api_prefix: str = "/api"

    # Recording defaults
    segment_duration_minutes: int = 15

    # Retention settings
    retention_days: Optional[int] = 30
    retention_max_gb: Optional[float] = None
    retention_check_interval_minutes: int = 60

    # JWT Authentication
    jwt_secret_key: str = "CHANGE_ME_IN_PRODUCTION"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60 * 24  # 24 hours

    # Encryption for camera credentials
    encryption_key: str = ""  # Fernet key, generate with Fernet.generate_key()

    # Default admin user (created on first startup if no users exist)
    default_admin_password: Optional[str] = None  # If not set, generates random

    # CORS (comma-separated origins for production)
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # ML Processing
    ml_enabled: bool = True
    ml_workers: int = 4
    ml_max_queue_size: int = 100

    # ML Processing defaults
    ml_default_fps: float = 2.0  # Frames per second to analyze
    ml_batch_size: int = 8
    ml_auto_process: bool = True  # Auto-process new recordings

    # ML Model settings
    ml_models_directory: Optional[Path] = None  # Derived from storage_root if not set
    ml_default_model: str = "yolov8l"
    ml_confidence_threshold: float = 0.5
    ml_nms_threshold: float = 0.45

    @model_validator(mode="after")
    def set_ml_models_directory(self) -> "Settings":
        """Set ml_models_directory from storage_root if not explicitly set."""
        if self.ml_models_directory is None:
            object.__setattr__(
                self, "ml_models_directory", self.storage_root / ".ml" / "models"
            )
        return self

    # Class filter - only save detections for these classes (empty = all classes)
    # Common classes: person, car, truck, bus, motorcycle, bicycle, dog, cat
    ml_class_filter: str = "person,car,truck,bus,motorcycle,bicycle,dog,cat"

    # ML Detection retention
    ml_detection_retention_days: Optional[int] = 90

    # Motion Detection settings
    motion_detection_enabled: bool = True
    motion_threshold: float = 0.5  # Percent of frame for motion trigger (0-100)
    motion_min_contour_area: int = 500  # Minimum pixel area to consider
    motion_history: int = 500  # Frames for background model history
    motion_var_threshold: float = 16.0  # Foreground/background threshold
    motion_detect_shadows: bool = True  # Detect and handle shadows
    motion_learning_rate: float = -1  # Background learning rate (-1 = auto)

    # Transcoding settings (for transcode_worker.py)
    transcode_enabled: bool = True
    transcode_crf: int = 28  # H.265 CRF (18-32, lower = better quality, larger)
    transcode_preset: str = "medium"  # FFmpeg preset (ultrafast to veryslow)
    transcode_min_age_minutes: int = 20  # Wait before transcoding new files
    transcode_check_interval: int = 300  # Seconds between checks in continuous mode

    # Live Detection settings (for live_detection_worker.py)
    live_detection_enabled: bool = True
    live_detection_fps: float = 3.0  # Frames per second per camera (extracted from segments)
    live_detection_cooldown: float = 30.0  # Seconds between same-class notifications
    live_detection_confidence: float = 0.6  # Higher threshold for real-time alerts
    live_detection_classes: str = "person,car,truck"  # Classes that trigger alerts
    live_detection_scale_height: int = 720  # Scale frames to this height for ML processing

    # Motion Gate settings (for live detection)
    motion_gate_enabled: bool = True  # Use motion detection as gate before YOLO
    motion_gate_threshold: float = 25.0  # Pixel difference threshold (0-255)
    motion_gate_min_percent: float = 0.1  # Minimum % of frame that must change
    motion_gate_min_area: int = 500  # Minimum contour area in pixels
    motion_gate_stale_seconds: float = 30.0  # Reset previous frame if older than this

    # NextGen Motion Detection Pipeline (GPU-accelerated)
    # Disable with --legacy flag or NEXTGEN_ENABLED=false
    nextgen_enabled: bool = True  # Use GPU-accelerated pipeline (default)
    nextgen_model_path: str = "/opt3/ronin/ml_models/yolov8n_dynamic.onnx"

    # JSON config file for detection pipeline (overrides env vars if present)
    # Set to empty string to disable JSON config loading
    nextgen_config_file: str = ""  # e.g., "/app/config/detection_config.json"

    # NextGen Motion Detection (GPU MOG2)
    nextgen_motion_history: int = 500  # Frames for background model
    nextgen_motion_var_threshold: float = 16.0  # Foreground threshold
    nextgen_motion_min_percent: float = 0.3  # Min % of frame for motion trigger

    # NextGen Object Detection
    nextgen_detection_confidence: float = 0.65  # Default threshold for vehicles
    nextgen_detection_nms_threshold: float = 0.45
    # Per-class thresholds (JSON string, parsed at runtime)
    # Lower threshold for people since they're harder to detect at night
    nextgen_class_thresholds: str = '{"person": 0.30, "dog": 0.45, "cat": 0.45}'

    # NextGen Tracking (ByteTrack)
    nextgen_track_high_thresh: float = 0.5
    nextgen_track_low_thresh: float = 0.1
    nextgen_track_match_thresh: float = 0.7
    nextgen_track_buffer: int = 90  # Frames to keep lost tracks (~30s at 3fps)
    nextgen_track_min_hits: int = 2  # Min detections to confirm track (lowered for 1 FPS)
    nextgen_track_min_displacement: float = 0.0  # Disabled - FSM handles movement check

    # NextGen FSM (Object State Machine)
    nextgen_fsm_validation_frames: int = 2  # Frames to confirm arrival (lowered for 1 FPS)
    nextgen_fsm_velocity_threshold: float = 0.002  # Normalized units/frame
    nextgen_fsm_displacement_threshold: float = 0.005  # Min displacement to be "moved" (0.5%)
    nextgen_fsm_stationary_seconds: float = 10.0  # Time to mark stationary
    nextgen_fsm_parked_seconds: float = 300.0  # Time to mark parked (5 min)
    nextgen_fsm_lost_seconds: float = 30.0  # Time without detection before departure
    nextgen_fsm_delayed_arrival_threshold: float = 60.0  # Max parked time for delayed arrival
    nextgen_fsm_loitering_seconds: float = 60.0  # Time stationary to trigger loitering alert

    # NextGen Periodic Detection (bypasses motion gate)
    # Run detection every N frames regardless of motion to catch small/distant objects
    # At 3 FPS, 30 frames = 10 seconds. Set to 0 to disable.
    nextgen_periodic_detection_interval: int = 30

    # NextGen Detection Active Window (bypasses motion gate)
    # After any detection, keep running YOLO for N seconds regardless of motion.
    # This handles the case where someone enters frame, stands still, then moves again.
    nextgen_detection_active_seconds: float = 10.0

    # VLLM Activity Characterization settings
    vllm_enabled: bool = False  # Enable VLLM activity characterization
    vllm_endpoint: str = "http://192.168.1.125:9001"  # VLLM server endpoint
    vllm_timeout: int = 60  # Request timeout in seconds
    vllm_frame_count: int = 4  # Number of frames to capture for analysis
    vllm_frame_interval_ms: int = 1000  # Milliseconds between frames
    vllm_poll_interval: float = 5.0  # Seconds between polling for new detections
    vllm_max_age_seconds: float = 300.0  # Max age of detection to process (5 minutes)
    vllm_save_mosaics: bool = True  # Save mosaic images for debugging


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
