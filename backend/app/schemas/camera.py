"""Pydantic schemas for Camera CRUD operations."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CameraBase(BaseModel):
    """Base schema for camera data."""

    name: str = Field(..., min_length=1, max_length=255, description="Camera name")
    host: str = Field(..., min_length=1, max_length=255, description="IP or hostname")
    port: int = Field(default=554, ge=1, le=65535, description="RTSP port")
    path: str = Field(default="/cam/realmonitor", max_length=512, description="RTSP path")
    username: Optional[str] = Field(default=None, max_length=255)
    password: Optional[str] = Field(default=None, max_length=255)
    transport: str = Field(default="tcp", description="Transport protocol: tcp or udp")
    recording_enabled: bool = Field(default=True)

    @field_validator("transport")
    @classmethod
    def validate_transport(cls, v: str) -> str:
        """Validate transport is tcp or udp."""
        v = v.lower()
        if v not in ("tcp", "udp"):
            raise ValueError("Transport must be 'tcp' or 'udp'")
        return v

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Ensure path starts with /."""
        if v and not v.startswith("/"):
            v = "/" + v
        return v


class CameraCreate(CameraBase):
    """Schema for creating a new camera."""

    pass


class CameraUpdate(BaseModel):
    """Schema for updating a camera (all fields optional)."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    host: Optional[str] = Field(default=None, min_length=1, max_length=255)
    port: Optional[int] = Field(default=None, ge=1, le=65535)
    path: Optional[str] = Field(default=None, max_length=512)
    username: Optional[str] = Field(default=None, max_length=255)
    password: Optional[str] = Field(default=None, max_length=255)
    transport: Optional[str] = Field(default=None)
    recording_enabled: Optional[bool] = Field(default=None)

    @field_validator("transport")
    @classmethod
    def validate_transport(cls, v: Optional[str]) -> Optional[str]:
        """Validate transport is tcp or udp."""
        if v is None:
            return v
        v = v.lower()
        if v not in ("tcp", "udp"):
            raise ValueError("Transport must be 'tcp' or 'udp'")
        return v

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Optional[str]) -> Optional[str]:
        """Ensure path starts with /."""
        if v and not v.startswith("/"):
            v = "/" + v
        return v


class CameraResponse(CameraBase):
    """Schema for camera response (includes id and timestamps)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    last_seen: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    # Hide password in response
    password: Optional[str] = Field(default=None, exclude=True)

    @property
    def rtsp_url(self) -> str:
        """Build RTSP URL (without password for security)."""
        auth = ""
        if self.username:
            auth = f"{self.username}@"
        return f"rtsp://{auth}{self.host}:{self.port}{self.path}"


class CameraListResponse(BaseModel):
    """Schema for list of cameras."""

    cameras: list[CameraResponse]
    total: int


class CameraTestResult(BaseModel):
    """Schema for camera connection test result."""

    success: bool
    message: str
    codec: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None
    duration_ms: Optional[int] = None
