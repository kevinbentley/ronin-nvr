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


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
