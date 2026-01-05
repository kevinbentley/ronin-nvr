"""SQLAlchemy models for RoninNVR."""

from app.models.camera import Camera
from app.models.detection import Detection
from app.models.ml_job import MLJob, JobStatus
from app.models.ml_model import MLModel
from app.models.ml_settings import MLSettings
from app.models.recording import Recording
from app.models.user import User

__all__ = [
    "Camera",
    "Detection",
    "JobStatus",
    "MLJob",
    "MLModel",
    "MLSettings",
    "Recording",
    "User",
]
