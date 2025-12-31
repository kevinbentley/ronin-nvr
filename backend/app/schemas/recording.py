"""Pydantic schemas for recording operations."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.schemas.base import UTCBaseModel


class RecordingStatusResponse(BaseModel):
    """Schema for recording status response."""

    camera_id: int
    camera_name: str
    state: str
    error_message: Optional[str] = None
    start_time: Optional[str] = None
    output_directory: str
    reconnect_attempts: int


class RecordingActionResponse(BaseModel):
    """Schema for recording action response."""

    success: bool
    message: str
    status: Optional[RecordingStatusResponse] = None


class RecordingSegmentResponse(UTCBaseModel):
    """Schema for a recording segment."""

    id: int
    camera_id: int
    file_path: str
    file_size: Optional[int] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    status: str
    codec: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None
    created_at: datetime


class RecordingListResponse(BaseModel):
    """Schema for list of recordings."""

    recordings: list[RecordingSegmentResponse]
    total: int
