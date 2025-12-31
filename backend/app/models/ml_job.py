"""ML processing job model."""

from datetime import datetime
from enum import Enum
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
    from app.models.recording import Recording


class JobStatus(str, Enum):
    """Job processing status."""

    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MLJob(Base):
    """ML processing job for a recording."""

    __tablename__ = "ml_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Link to recording
    recording_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("recordings.id", ondelete="CASCADE"), nullable=False
    )

    # Job details
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default=JobStatus.PENDING.value)
    priority: Mapped[int] = mapped_column(Integer, default=0)

    # Progress tracking
    progress_percent: Mapped[float] = mapped_column(Float, default=0.0)
    frames_processed: Mapped[int] = mapped_column(Integer, default=0)
    total_frames: Mapped[int] = mapped_column(Integer, default=0)
    detections_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timing (using TIMESTAMP WITH TIME ZONE for proper UTC handling)
    started_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    processing_time_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    # Worker tracking (for standalone worker processes)
    worker_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    last_heartbeat: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    # Configuration used for this job
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    recording: Mapped["Recording"] = relationship("Recording", back_populates="ml_jobs")

    # Indexes
    __table_args__ = (
        Index("ix_ml_jobs_status", "status"),
        Index("ix_ml_jobs_recording_id", "recording_id"),
        Index("ix_ml_jobs_created_at", "created_at"),
        # Composite index for efficient job claiming: ORDER BY priority DESC, created_at ASC
        Index("ix_ml_jobs_pending_queue", "status", "priority", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<MLJob(id={self.id}, recording_id={self.recording_id}, "
            f"status={self.status}, progress={self.progress_percent:.1f}%)>"
        )
