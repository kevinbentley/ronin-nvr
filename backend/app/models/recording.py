"""Recording model for storing video segment metadata."""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.camera import Camera
    from app.models.detection import Detection
    from app.models.ml_job import MLJob


class RecordingStatus(str, Enum):
    """Recording segment status."""

    RECORDING = "recording"
    COMPLETED = "completed"
    ERROR = "error"


class Recording(Base):
    """Video recording segment metadata."""

    __tablename__ = "recordings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False
    )

    # File info
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    # Time range
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(
        String(20), default=RecordingStatus.RECORDING.value
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    codec: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    resolution: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    # Relationships
    camera: Mapped["Camera"] = relationship("Camera", back_populates="recordings")
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="recording", cascade="all, delete-orphan"
    )
    ml_jobs: Mapped[list["MLJob"]] = relationship(
        "MLJob", back_populates="recording", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Recording(id={self.id}, camera_id={self.camera_id}, "
            f"start={self.start_time}, status={self.status})>"
        )
