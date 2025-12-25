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
    retention_days: Optional[int] = 30
    retention_max_gb: Optional[float] = None


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
