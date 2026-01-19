"""Camera model for storing IP camera configurations."""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.detection import Detection
    from app.models.recording import Recording


class CameraStatus(str, Enum):
    """Camera connection status."""

    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    UNKNOWN = "unknown"


class Camera(Base):
    """IP Camera configuration and status."""

    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    # Connection settings
    host: Mapped[str] = mapped_column(String(255), nullable=False)
    port: Mapped[int] = mapped_column(Integer, default=554)
    path: Mapped[str] = mapped_column(String(512), default="/cam/realmonitor")
    username: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    password: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Stream settings
    transport: Mapped[str] = mapped_column(String(10), default="tcp")  # tcp or udp

    # Status
    status: Mapped[str] = mapped_column(
        String(20), default=CameraStatus.UNKNOWN.value
    )
    last_seen: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Recording settings
    recording_enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Scene description for VLLM activity characterization
    # e.g., "White house with driveway, black trashcan on left, planter on porch"
    scene_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ONVIF protocol settings
    onvif_port: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    onvif_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    onvif_profile_token: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )
    onvif_device_info: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    onvif_events_enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps (using TIMESTAMP WITH TIME ZONE for proper UTC handling)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    recordings: Mapped[list["Recording"]] = relationship(
        "Recording", back_populates="camera", cascade="all, delete-orphan"
    )
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="camera", cascade="all, delete-orphan"
    )

    @property
    def rtsp_url(self) -> str:
        """Build RTSP URL from components."""
        auth = ""
        if self.username:
            auth = f"{self.username}"
            if self.password:
                auth += f":{self.password}"
            auth += "@"
        return f"rtsp://{auth}{self.host}:{self.port}{self.path}"

    def __repr__(self) -> str:
        return f"<Camera(id={self.id}, name={self.name}, status={self.status})>"
