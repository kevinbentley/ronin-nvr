"""Pydantic schemas for ML API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.base import UTCBaseModel


# === Job Schemas ===

class CreateJobRequest(BaseModel):
    """Request to create an ML processing job."""

    recording_id: int = Field(..., description="ID of recording to process")
    model_name: Optional[str] = Field(None, description="Model to use (default from settings)")
    priority: int = Field(0, description="Job priority (higher = more urgent)")


class JobResponse(UTCBaseModel):
    """Response for an ML job."""

    id: int
    recording_id: int
    model_name: str
    status: str
    priority: int
    progress_percent: float
    frames_processed: int
    total_frames: int
    detections_count: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    processing_time_seconds: Optional[float]
    error_message: Optional[str]
    created_at: datetime


class JobListResponse(BaseModel):
    """Response for job list."""

    jobs: list[JobResponse]
    total: int


# === Detection Schemas ===

class DetectionResponse(UTCBaseModel):
    """Response for a detection."""

    id: int
    recording_id: Optional[int] = None  # Nullable for live detections
    camera_id: int
    class_name: str
    confidence: float
    timestamp_ms: int
    frame_number: int
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    model_name: str
    model_version: Optional[str] = None
    created_at: datetime
    detected_at: Optional[datetime] = None  # Actual detection time for live detections


class DetectionListResponse(BaseModel):
    """Response for detection list."""

    detections: list[DetectionResponse]
    total: int


class DetectionSummaryItem(BaseModel):
    """Summary item for detection statistics."""

    label: str
    count: int
    avg_confidence: float


class DetectionSummaryResponse(BaseModel):
    """Response for detection summary."""

    total_detections: int
    items: list[DetectionSummaryItem]
    time_range: Optional[dict] = None


class TimelineEvent(BaseModel):
    """A single event for timeline display."""

    timestamp_ms: int  # Milliseconds from start of day
    class_name: str
    confidence: float
    recording_id: Optional[int] = None  # None for live detections
    count: int = 1  # Number of detections at this time
    event_source: str = "ml"  # "ml", "onvif_motion", "onvif_analytics"


class TimelineEventsResponse(BaseModel):
    """Response for timeline events."""

    events: list[TimelineEvent]
    total: int
    class_counts: dict[str, int]  # Summary by class


# === Model Schemas ===

class ModelResponse(UTCBaseModel):
    """Response for an ML model."""

    id: int
    name: str
    display_name: str
    version: str
    file_path: str
    model_type: str
    class_names: list[str]
    input_size: list[int]
    default_confidence_threshold: float
    default_nms_threshold: float
    is_enabled: bool
    is_default: bool
    description: Optional[str]
    created_at: datetime
    updated_at: datetime


class ModelListResponse(BaseModel):
    """Response for model list."""

    models: list[ModelResponse]


class ModelConfigRequest(BaseModel):
    """Request to update model configuration."""

    display_name: Optional[str] = None
    description: Optional[str] = None
    default_confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    default_nms_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_enabled: Optional[bool] = None
    is_default: Optional[bool] = None


# === Status Schemas ===

class WorkerStatusResponse(BaseModel):
    """Status of a single worker."""

    id: int
    running: bool
    current_job: Optional[int]


class QueueStatusResponse(BaseModel):
    """Status of the job queue."""

    pending: int
    active: int
    max_size: int
    active_jobs: list[int]


class MLStatusResponse(BaseModel):
    """Overall ML system status."""

    running: bool
    workers: int
    worker_status: list[WorkerStatusResponse]
    queue: QueueStatusResponse
    models_loaded: list[str]


# === Camera ML Settings ===

class CameraMLSettingsResponse(BaseModel):
    """ML settings for a camera."""

    camera_id: int
    ml_enabled: bool
    model_name: Optional[str]
    confidence_threshold: Optional[float]
    classes_filter: Optional[list[str]]


class CameraMLSettingsRequest(BaseModel):
    """Request to update camera ML settings."""

    ml_enabled: Optional[bool] = None
    model_name: Optional[str] = None
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    classes_filter: Optional[list[str]] = None


# === Global ML Settings ===


class MLSettingsResponse(BaseModel):
    """Response for global ML settings."""

    live_detection_enabled: bool
    live_detection_fps: float
    live_detection_cooldown: float
    live_detection_confidence: float
    live_detection_classes: list[str]
    class_thresholds: dict[str, float]
    updated_at: Optional[datetime]


class MLSettingsUpdateRequest(BaseModel):
    """Request to update global ML settings."""

    live_detection_enabled: Optional[bool] = None
    live_detection_fps: Optional[float] = Field(None, ge=0.1, le=10.0)
    live_detection_cooldown: Optional[float] = Field(None, ge=1.0, le=300.0)
    live_detection_confidence: Optional[float] = Field(None, ge=0.1, le=1.0)
    live_detection_classes: Optional[list[str]] = None
    class_thresholds: Optional[dict[str, float]] = None


# === Object Events ===


class ObjectEventResponse(UTCBaseModel):
    """Response for a single object event."""

    id: int
    event_type: str
    class_name: str
    track_id: int
    old_state: Optional[str]
    new_state: Optional[str]
    confidence: float
    duration_seconds: float
    snapshot_url: Optional[str]
    camera_id: int
    camera_name: Optional[str]
    event_time: datetime


class ObjectEventListResponse(BaseModel):
    """Paginated list of object events."""

    events: list[ObjectEventResponse]
    total: int
    offset: int
    limit: int
