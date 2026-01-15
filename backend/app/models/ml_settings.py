"""ML Settings model for persistent configuration."""

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, String, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

# Default per-class thresholds (lower = more sensitive)
DEFAULT_CLASS_THRESHOLDS = {
    "person": 0.30,
    "car": 0.65,
    "truck": 0.65,
    "dog": 0.35,
    "cat": 0.35,
}


class MLSettings(Base):
    """Singleton table storing ML configuration.

    This table always has exactly one row (id=1) which stores the current
    ML settings. Settings are loaded by workers periodically to allow
    runtime configuration changes without restarts.
    """

    __tablename__ = "ml_settings"

    id: Mapped[int] = mapped_column(primary_key=True, default=1)

    # Live detection settings
    live_detection_enabled: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )
    live_detection_fps: Mapped[float] = mapped_column(
        Float, default=3.0, nullable=False
    )
    live_detection_cooldown: Mapped[float] = mapped_column(
        Float, default=30.0, nullable=False
    )
    live_detection_confidence: Mapped[float] = mapped_column(
        Float, default=0.6, nullable=False
    )
    live_detection_classes: Mapped[str] = mapped_column(
        String, default="person,car,truck", nullable=False
    )

    # Per-class confidence thresholds (JSON-encoded dict)
    class_thresholds: Mapped[Optional[str]] = mapped_column(
        Text, default=None, nullable=True
    )

    # Legacy columns (kept for backwards compatibility, not used in UI)
    historical_confidence: Mapped[float] = mapped_column(
        Float, default=0.5, nullable=False
    )
    historical_classes: Mapped[str] = mapped_column(
        String, default="person,car,truck,bus,motorcycle,bicycle,dog,cat", nullable=False
    )

    # Timestamps
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def get_class_thresholds(self) -> dict[str, float]:
        """Get class thresholds as dict, using defaults if not set."""
        if self.class_thresholds:
            try:
                return json.loads(self.class_thresholds)
            except json.JSONDecodeError:
                pass
        return DEFAULT_CLASS_THRESHOLDS.copy()

    def set_class_thresholds(self, thresholds: dict[str, float]) -> None:
        """Set class thresholds from dict."""
        self.class_thresholds = json.dumps(thresholds)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "live_detection_enabled": self.live_detection_enabled,
            "live_detection_fps": self.live_detection_fps,
            "live_detection_cooldown": self.live_detection_cooldown,
            "live_detection_confidence": self.live_detection_confidence,
            "live_detection_classes": self.live_detection_classes.split(","),
            "class_thresholds": self.get_class_thresholds(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
