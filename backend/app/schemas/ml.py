"""Pydantic schemas for ML API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# === Job Schemas ===

class CreateJobRequest(BaseModel):
    """Request to create an ML processing job."""

    recording_id: int = Field(..., description="ID of recording to process")
    model_name: Optional[str] = Field(None, description="Model to use (default from settings)")
    priority: int = Field(0, description="Job priority (higher = more urgent)")


class JobResponse(BaseModel):
    """Response for an ML job."""

    model_config = ConfigDict(from_attributes=True)

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

class DetectionResponse(BaseModel):
    """Response for a detection."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    recording_id: int
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
    model_version: Optional[str]
    created_at: datetime


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


# === Model Schemas ===

class ModelResponse(BaseModel):
    """Response for an ML model."""

    model_config = ConfigDict(from_attributes=True)

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
