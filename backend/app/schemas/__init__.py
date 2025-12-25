"""Pydantic schemas for request/response validation."""

from app.schemas.camera import (
    CameraCreate,
    CameraListResponse,
    CameraResponse,
    CameraTestResult,
    CameraUpdate,
)
from app.schemas.recording import (
    RecordingActionResponse,
    RecordingListResponse,
    RecordingSegmentResponse,
    RecordingStatusResponse,
)

__all__ = [
    "CameraCreate",
    "CameraUpdate",
    "CameraResponse",
    "CameraListResponse",
    "CameraTestResult",
    "RecordingStatusResponse",
    "RecordingActionResponse",
    "RecordingSegmentResponse",
    "RecordingListResponse",
]
