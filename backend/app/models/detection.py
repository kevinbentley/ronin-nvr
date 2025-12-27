"""Detection model for ML inference results."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.camera import Camera
    from app.models.recording import Recording


class Detection(Base):
    """ML detection result from video analysis."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Links to recording and camera
    recording_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("recordings.id", ondelete="CASCADE"), nullable=False
    )
    camera_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False
    )

    # Detection details
    class_name: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Timestamp within the recording
    timestamp_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Bounding box (normalized 0-1)
    bbox_x: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_width: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_height: Mapped[float] = mapped_column(Float, nullable=False)

    # Model info
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Additional data (for custom attributes)
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    # Relationships
    recording: Mapped["Recording"] = relationship("Recording", back_populates="detections")
    camera: Mapped["Camera"] = relationship("Camera", back_populates="detections")

    # Indexes for common queries
    __table_args__ = (
        Index("ix_detections_camera_id", "camera_id"),
        Index("ix_detections_recording_id", "recording_id"),
        Index("ix_detections_class_name", "class_name"),
        Index("ix_detections_created_at", "created_at"),
        Index("ix_detections_confidence", "confidence"),
    )

    def __repr__(self) -> str:
        return (
            f"<Detection(id={self.id}, class={self.class_name}, "
            f"confidence={self.confidence:.2f}, frame={self.frame_number})>"
        )
