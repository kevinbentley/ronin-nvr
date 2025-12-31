"""ML inference services for video analysis."""

from app.services.ml.frame_extractor import FrameExtractor, VideoInfo
from app.services.ml.model_manager import LoadedModel, ModelManager, model_manager
from app.services.ml.detection_service import DetectionResult, DetectionService, detection_service
from app.services.ml.motion_detector import MotionDetector, MotionResult
from app.services.ml.job_service import JobInfo, JobService
from app.services.ml.coordinator import MLCoordinator, ml_coordinator
from app.services.ml.recording_watcher import RecordingWatcher, recording_watcher
from app.services.ml.events import MLEventService, MLEvent, EventType, ml_event_service

__all__ = [
    "DetectionResult",
    "DetectionService",
    "EventType",
    "FrameExtractor",
    "JobInfo",
    "JobService",
    "LoadedModel",
    "MLCoordinator",
    "MLEvent",
    "MLEventService",
    "ModelManager",
    "MotionDetector",
    "MotionResult",
    "RecordingWatcher",
    "VideoInfo",
    "detection_service",
    "ml_coordinator",
    "ml_event_service",
    "model_manager",
    "recording_watcher",
]
