"""Configuration for the LLM-powered system watchdog."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WatchdogConfig:
    """Configuration for the watchdog daemon."""

    # LLM endpoint (OpenAI-compatible API)
    llm_endpoint: str = "http://192.168.1.125:9001/v1"
    llm_model: str = "default"
    llm_timeout: float = 30.0
    llm_max_tokens: int = 1024

    # Check interval
    check_interval: int = 60  # seconds

    # Memory thresholds
    memory_warning_percent: float = 80.0
    memory_critical_percent: float = 95.0

    # Process limits
    max_ffmpeg_processes: int = 10
    ffmpeg_memory_warning_mb: float = 2000.0  # Per-process

    # Docker configuration
    docker_compose_path: str = "/home/kbentley/dev/ronin-nvr"
    transcode_container_name: str = "ronin-nvr-transcode-worker"

    # Safety settings
    dry_run: bool = False
    max_actions_per_10min: int = 3
    action_cooldown_seconds: float = 300.0  # 5 minutes
    protected_processes: list[str] = field(
        default_factory=lambda: [
            "postgres",
            "systemd",
            "sshd",
            "init",
            "dockerd",
            "containerd",
        ]
    )

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def from_env(cls) -> "WatchdogConfig":
        """Load configuration from environment variables."""
        return cls(
            llm_endpoint=os.getenv(
                "WATCHDOG_LLM_ENDPOINT", "http://192.168.1.125:9001/v1"
            ),
            llm_model=os.getenv("WATCHDOG_LLM_MODEL", "default"),
            llm_timeout=float(os.getenv("WATCHDOG_LLM_TIMEOUT", "30.0")),
            check_interval=int(os.getenv("WATCHDOG_CHECK_INTERVAL", "60")),
            memory_warning_percent=float(
                os.getenv("WATCHDOG_MEMORY_WARNING", "80.0")
            ),
            memory_critical_percent=float(
                os.getenv("WATCHDOG_MEMORY_CRITICAL", "95.0")
            ),
            max_ffmpeg_processes=int(
                os.getenv("WATCHDOG_MAX_FFMPEG", "10")
            ),
            docker_compose_path=os.getenv(
                "WATCHDOG_COMPOSE_PATH", "/home/kbentley/dev/ronin-nvr"
            ),
            dry_run=os.getenv("WATCHDOG_DRY_RUN", "false").lower() == "true",
            log_level=os.getenv("WATCHDOG_LOG_LEVEL", "INFO"),
            log_file=os.getenv("WATCHDOG_LOG_FILE"),
        )
