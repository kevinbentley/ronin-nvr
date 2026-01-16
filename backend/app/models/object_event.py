"""Object event model for FSM state transitions."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ObjectEvent(Base):
    """Stores FSM events like ARRIVAL, DEPARTURE, PARKED, LOITERING.

    These events represent object lifecycle transitions detected by the
    ML pipeline's finite state machine.
    """

    __tablename__ = "object_events"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Event details
    event_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # State transition
    old_state: Mapped[str | None] = mapped_column(String(32), nullable=True)
    new_state: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Detection details
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Snapshot
    snapshot_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Camera reference
    camera_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    camera = relationship("Camera")

    # Timestamps
    event_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "class_name": self.class_name,
            "track_id": self.track_id,
            "old_state": self.old_state,
            "new_state": self.new_state,
            "confidence": self.confidence,
            "duration_seconds": self.duration_seconds,
            "snapshot_path": self.snapshot_path,
            "camera_id": self.camera_id,
            "camera_name": self.camera.name if self.camera else None,
            "event_time": self.event_time.isoformat() if self.event_time else None,
        }
