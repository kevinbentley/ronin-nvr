"""Detection model for ML inference results.

This model supports both historical detections (from completed recordings) and
live detections (from real-time stream analysis):

- Historical: Has recording_id, timestamp_ms relative to recording start
- Live: No recording_id, uses detected_at for absolute timestamp

Both types are stored in the same table for unified timeline and retention.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import (
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.camera import Camera
    from app.models.recording import Recording


class Detection(Base):
    """ML detection result from video analysis.

    Supports both historical (file-based) and live (stream-based) detections:

    Historical detections:
        - recording_id: set to the recording being analyzed
        - timestamp_ms: position within the recording
        - detected_at: computed as recording_start + timestamp_ms

    Live detections:
        - recording_id: NULL (correlate by timestamp later if needed)
        - timestamp_ms: 0 (not applicable)
        - detected_at: actual UTC time of detection
        - snapshot_path: JPG with bounding boxes for preview/LLM
    """

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Links to recording and camera
    # recording_id is nullable to support live detections (no recording yet)
    recording_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("recordings.id", ondelete="CASCADE"), nullable=True
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
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    # Actual detection time (for live: now, for historical: recording_start + timestamp_ms)
    # This enables unified timeline queries across both detection types
    detected_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # Snapshot path for event thumbnails (JPG with bounding boxes drawn)
    # Used for previews and future Vision LLM integration
    snapshot_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Mosaic path for VLLM 2x2 time sequence grid
    # Stored in .vllm_debug/{camera_id}/{date}/{time}-{id}.jpg
    mosaic_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Vision LLM scene description (future feature)
    # e.g., "A delivery driver placing a package on the front porch"
    llm_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Event source for unified timeline (distinguishes ML from camera events)
    # Values: "ml", "onvif_motion", "onvif_analytics"
    event_source: Mapped[str] = mapped_column(String(20), default="ml", nullable=False)

    # Relationships
    recording: Mapped[Optional["Recording"]] = relationship(
        "Recording", back_populates="detections"
    )
    camera: Mapped["Camera"] = relationship("Camera", back_populates="detections")

    # Indexes for common queries
    __table_args__ = (
        Index("ix_detections_camera_id", "camera_id"),
        Index("ix_detections_recording_id", "recording_id"),
        Index("ix_detections_class_name", "class_name"),
        Index("ix_detections_created_at", "created_at"),
        Index("ix_detections_confidence", "confidence"),
        # Indexes for live detection timeline queries (added by migration)
        Index("ix_detections_detected_at", "detected_at"),
        Index("ix_detections_camera_detected_at", "camera_id", "detected_at"),
        # Index for event source filtering (added by migration)
        Index("ix_detections_event_source", "event_source"),
    )

    @property
    def is_live_detection(self) -> bool:
        """Check if this is a live detection (no associated recording)."""
        return self.recording_id is None

    def __repr__(self) -> str:
        source = "live" if self.is_live_detection else f"rec={self.recording_id}"
        return (
            f"<Detection(id={self.id}, class={self.class_name}, "
            f"confidence={self.confidence:.2f}, {source})>"
        )
