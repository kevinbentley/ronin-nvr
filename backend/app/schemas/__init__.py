"""Pydantic schemas for request/response validation."""

from app.schemas.auth import (
    LoginRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
)
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
    "LoginRequest",
    "TokenResponse",
    "UserCreate",
    "UserResponse",
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
