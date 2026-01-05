"""ML Settings model for persistent configuration."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


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
        Float, default=1.0, nullable=False
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

    # Historical processing settings
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

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "live_detection_enabled": self.live_detection_enabled,
            "live_detection_fps": self.live_detection_fps,
            "live_detection_cooldown": self.live_detection_cooldown,
            "live_detection_confidence": self.live_detection_confidence,
            "live_detection_classes": self.live_detection_classes.split(","),
            "historical_confidence": self.historical_confidence,
            "historical_classes": self.historical_classes.split(","),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
