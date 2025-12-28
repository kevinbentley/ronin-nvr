"""Application configuration using pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

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
    ml_models_directory: Path = Path("./storage/.ml/models")
    ml_default_model: str = "yolov8l"
    ml_confidence_threshold: float = 0.5
    ml_nms_threshold: float = 0.45

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


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
