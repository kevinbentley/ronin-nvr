"""GPU orchestrator for multi-camera, multi-GPU processing.

This module provides centralized management of GPU resources for processing
multiple camera streams. Key features:

- Camera-to-GPU assignment (static sharding)
- Per-GPU pipeline management
- Load balancing across GPUs
- Resource monitoring

Architecture:
```
GPU 0                          GPU 1
┌────────────────────┐        ┌────────────────────┐
│ Cameras 1-8        │        │ Cameras 9-16       │
│ ┌───────────────┐  │        │ ┌───────────────┐  │
│ │ GPU MOG2      │  │        │ │ GPU MOG2      │  │
│ │ TensorRT Det  │  │        │ │ TensorRT Det  │  │
│ │ ByteTrack     │  │        │ │ ByteTrack     │  │
│ │ FSM           │  │        │ │ FSM           │  │
│ └───────────────┘  │        │ └───────────────┘  │
└────────────────────┘        └────────────────────┘
```

For single-GPU systems, all cameras are processed on GPU 0.
"""

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.services.ml.gpu_motion import GPUMotionGate, GPUMotionResult
from app.services.ml.tensorrt_inference import TensorRTDetector, TensorRTDetection
from app.services.ml.tracker import ByteTracker, Detection, TrackedObject
from app.services.ml.object_fsm import ObjectStateMachine, ObjectEvent, EventType

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from processing a frame through the full pipeline."""

    camera_id: int
    frame_number: int
    timestamp: float

    # Motion detection
    motion_detected: bool
    motion_percent: float

    # Object detection
    detections: list[TensorRTDetection] = field(default_factory=list)

    # Tracking
    tracks: list[TrackedObject] = field(default_factory=list)

    # Events
    events: list[ObjectEvent] = field(default_factory=list)

    # Timing
    motion_time_ms: float = 0.0
    detection_time_ms: float = 0.0
    tracking_time_ms: float = 0.0
    fsm_time_ms: float = 0.0

    @property
    def total_time_ms(self) -> float:
        """Total processing time."""
        return (
            self.motion_time_ms +
            self.detection_time_ms +
            self.tracking_time_ms +
            self.fsm_time_ms
        )

    @property
    def has_activity(self) -> bool:
        """Whether there's any activity worth reporting."""
        return len(self.tracks) > 0 or len(self.events) > 0


@dataclass
class GPUPipelineConfig:
    """Configuration for a GPU pipeline."""

    device_id: int = 0

    # Model paths (use dynamic batch model for efficient multi-camera processing)
    model_path: str = "/opt3/ronin/ml_models/yolov8n_dynamic.onnx"

    # Motion detection
    motion_history: int = 500
    motion_var_threshold: float = 16.0
    motion_min_percent: float = 0.1

    # Detection
    detection_confidence: float = 0.5  # Default threshold for all classes
    detection_nms_threshold: float = 0.45
    # Per-class thresholds override detection_confidence for specific classes
    # e.g., {"person": 0.45, "car": 0.65} - lower threshold for people
    class_thresholds: dict = field(default_factory=dict)

    # Tracking
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    track_match_thresh: float = 0.7
    track_buffer: int = 90  # Frames to keep lost tracks (~30s at 3fps)
    track_min_hits: int = 3
    track_min_displacement: float = 0.0  # Minimum movement to confirm track (0=disabled)

    # FSM
    fsm_validation_frames: int = 5
    fsm_velocity_threshold: float = 0.002
    fsm_stationary_seconds: float = 10.0
    fsm_parked_seconds: float = 300.0
    fsm_lost_seconds: float = 30.0  # Time without detection before departure

    # Periodic detection (bypasses motion gate)
    # Run detection every N frames regardless of motion to catch small/distant objects
    # that don't trigger motion gate. Set to 0 to disable.
    periodic_detection_interval: int = 30  # ~10 seconds at 3 FPS


class GPUPipeline:
    """Complete detection pipeline for a single GPU.

    Manages all processing components for cameras assigned to this GPU:
    - Motion detection (GPU MOG2)
    - Object detection (TensorRT/ONNX)
    - Multi-object tracking (ByteTrack)
    - State machine (FSM)
    """

    def __init__(
        self,
        config: GPUPipelineConfig,
        camera_ids: list[int],
    ):
        """Initialize GPU pipeline.

        Args:
            config: Pipeline configuration
            camera_ids: List of camera IDs assigned to this GPU
        """
        self.config = config
        self.camera_ids = set(camera_ids)
        self.device_id = config.device_id

        # Set CUDA device before creating any GPU resources
        # This ensures MOG2, streams, and GpuMats are created on the correct GPU
        cv2.cuda.setDevice(config.device_id)

        # Initialize components
        self._motion_gate = GPUMotionGate(
            history=config.motion_history,
            var_threshold=config.motion_var_threshold,
            min_motion_percent=config.motion_min_percent,
        )

        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._detector = TensorRTDetector(
            model_path=model_path,
            confidence_threshold=config.detection_confidence,
            nms_threshold=config.detection_nms_threshold,
            class_thresholds=config.class_thresholds,
            device_id=config.device_id,
            warmup_iterations=5,
        )

        # Per-camera trackers and FSMs
        self._trackers: dict[int, ByteTracker] = {}
        self._fsms: dict[int, ObjectStateMachine] = {}

        for cam_id in camera_ids:
            self._trackers[cam_id] = ByteTracker(
                track_high_thresh=config.track_high_thresh,
                track_low_thresh=config.track_low_thresh,
                match_thresh=config.track_match_thresh,
                track_buffer=config.track_buffer,
                min_hits=config.track_min_hits,
                min_displacement=config.track_min_displacement,
            )
            self._fsms[cam_id] = ObjectStateMachine(
                validation_frames=config.fsm_validation_frames,
                velocity_threshold=config.fsm_velocity_threshold,
                stationary_seconds=config.fsm_stationary_seconds,
                parked_seconds=config.fsm_parked_seconds,
                lost_seconds=config.fsm_lost_seconds,
            )

        self._frame_counts: dict[int, int] = {cid: 0 for cid in camera_ids}
        self._lock = threading.Lock()

        logger.info(
            f"GPUPipeline initialized: device={config.device_id}, "
            f"cameras={sorted(camera_ids)}"
        )

    def add_camera(self, camera_id: int) -> None:
        """Add a new camera to this pipeline."""
        with self._lock:
            if camera_id not in self.camera_ids:
                self.camera_ids.add(camera_id)
                self._trackers[camera_id] = ByteTracker(
                    track_high_thresh=self.config.track_high_thresh,
                    track_low_thresh=self.config.track_low_thresh,
                    match_thresh=self.config.track_match_thresh,
                    track_buffer=self.config.track_buffer,
                    min_hits=self.config.track_min_hits,
                    min_displacement=self.config.track_min_displacement,
                )
                self._fsms[camera_id] = ObjectStateMachine(
                    validation_frames=self.config.fsm_validation_frames,
                    velocity_threshold=self.config.fsm_velocity_threshold,
                    stationary_seconds=self.config.fsm_stationary_seconds,
                    parked_seconds=self.config.fsm_parked_seconds,
                    lost_seconds=self.config.fsm_lost_seconds,
                )
                self._frame_counts[camera_id] = 0

    def remove_camera(self, camera_id: int) -> None:
        """Remove a camera from this pipeline."""
        with self._lock:
            self.camera_ids.discard(camera_id)
            self._trackers.pop(camera_id, None)
            self._fsms.pop(camera_id, None)
            self._frame_counts.pop(camera_id, None)
            self._motion_gate.reset_camera(camera_id)

    def process(
        self,
        camera_id: int,
        frame: np.ndarray,
        timestamp: float,
        skip_detection: bool = False,
    ) -> PipelineResult:
        """Process a frame through the full pipeline.

        Args:
            camera_id: Camera identifier
            frame: BGR frame as numpy array
            timestamp: Frame timestamp
            skip_detection: Skip object detection (just motion check)

        Returns:
            PipelineResult with all outputs
        """
        import time

        if camera_id not in self.camera_ids:
            self.add_camera(camera_id)

        self._frame_counts[camera_id] += 1
        frame_number = self._frame_counts[camera_id]

        result = PipelineResult(
            camera_id=camera_id,
            frame_number=frame_number,
            timestamp=timestamp,
            motion_detected=False,
            motion_percent=0.0,
        )

        # Step 1: Motion detection
        t0 = time.perf_counter()
        motion_result = self._motion_gate.check(camera_id, frame)
        result.motion_time_ms = (time.perf_counter() - t0) * 1000
        result.motion_detected = motion_result.motion_detected
        result.motion_percent = motion_result.motion_percent

        # Check if this is a periodic detection frame (bypasses motion gate)
        periodic_interval = self.config.periodic_detection_interval
        is_periodic_frame = (
            periodic_interval > 0 and
            frame_number % periodic_interval == 0
        )

        # Run detection if motion detected OR periodic frame
        should_detect = motion_result.motion_detected or is_periodic_frame

        # Skip detection if requested or no trigger
        if skip_detection or not should_detect:
            # Still run FSM update with empty tracks to handle departures
            t0 = time.perf_counter()
            fsm = self._fsms.get(camera_id)
            if fsm:
                result.events = fsm.update([])
            result.fsm_time_ms = (time.perf_counter() - t0) * 1000
            return result

        # Step 2: Object detection
        t0 = time.perf_counter()
        detections = self._detector.detect(frame)
        result.detection_time_ms = (time.perf_counter() - t0) * 1000
        result.detections = detections

        if detections:
            det_summary = ", ".join(f"{d.class_name}:{d.confidence:.2f}" for d in detections[:5])
            logger.debug(f"Camera {camera_id}: YOLO found {len(detections)} objects: {det_summary}")
        elif motion_result.motion_detected:
            logger.debug(f"Camera {camera_id}: YOLO ran but found nothing (motion={result.motion_percent:.1f}%)")

        # Step 3: Tracking
        t0 = time.perf_counter()
        tracker = self._trackers.get(camera_id)
        if tracker:
            # Convert to tracker format
            track_dets = [
                Detection(
                    x=d.x, y=d.y, width=d.width, height=d.height,
                    confidence=d.confidence, class_id=d.class_id,
                    class_name=d.class_name
                )
                for d in detections
            ]
            result.tracks = tracker.update(track_dets)
        result.tracking_time_ms = (time.perf_counter() - t0) * 1000

        # Step 4: FSM
        t0 = time.perf_counter()
        fsm = self._fsms.get(camera_id)
        if fsm:
            result.events = fsm.update(result.tracks)
        result.fsm_time_ms = (time.perf_counter() - t0) * 1000

        return result

    def process_batch(
        self,
        frames: dict[int, np.ndarray],
        timestamp: float,
        skip_detection: bool = False,
    ) -> dict[int, PipelineResult]:
        """Process frames from multiple cameras using batched inference.

        This method is significantly more efficient than calling process()
        multiple times, as it batches the GPU inference across all cameras
        with motion.

        Args:
            frames: Dict mapping camera_id to BGR frame
            timestamp: Frame timestamp (shared across all frames)
            skip_detection: Skip object detection (just motion check)

        Returns:
            Dict mapping camera_id to PipelineResult
        """
        import time

        # Ensure we're on the correct GPU for this pipeline
        cv2.cuda.setDevice(self.device_id)

        results: dict[int, PipelineResult] = {}
        motion_frames: dict[int, np.ndarray] = {}

        # Step 1: Motion detection for all cameras (sequential, ~3ms each)
        t0_motion = time.perf_counter()

        for camera_id, frame in frames.items():
            if camera_id not in self.camera_ids:
                self.add_camera(camera_id)

            self._frame_counts[camera_id] += 1
            frame_number = self._frame_counts[camera_id]

            result = PipelineResult(
                camera_id=camera_id,
                frame_number=frame_number,
                timestamp=timestamp,
                motion_detected=False,
                motion_percent=0.0,
            )

            # Check motion
            motion_result = self._motion_gate.check(camera_id, frame)
            result.motion_detected = motion_result.motion_detected
            result.motion_percent = motion_result.motion_percent

            # Check if this is a periodic detection frame
            periodic_interval = self.config.periodic_detection_interval
            is_periodic_frame = (
                periodic_interval > 0 and
                frame_number % periodic_interval == 0
            )

            # Include frame for detection if motion OR periodic
            should_detect = motion_result.motion_detected or is_periodic_frame
            if should_detect and not skip_detection:
                motion_frames[camera_id] = frame

            results[camera_id] = result

        motion_time_ms = (time.perf_counter() - t0_motion) * 1000
        per_camera_motion_ms = motion_time_ms / len(frames) if frames else 0

        # Distribute motion time across results
        for result in results.values():
            result.motion_time_ms = per_camera_motion_ms

        # Step 2: Batched detection for cameras with motion
        detection_results: dict[int, list] = {}

        if motion_frames:
            t0_det = time.perf_counter()

            # Prepare batch
            camera_ids_with_motion = list(motion_frames.keys())
            frames_list = [motion_frames[cid] for cid in camera_ids_with_motion]

            # Batched inference
            batch_detections = self._detector.detect_batch(frames_list)

            detection_time_ms = (time.perf_counter() - t0_det) * 1000
            per_camera_det_ms = detection_time_ms / len(frames_list)

            # Map results back to camera IDs
            for camera_id, detections in zip(camera_ids_with_motion, batch_detections):
                detection_results[camera_id] = detections
                results[camera_id].detections = detections
                results[camera_id].detection_time_ms = per_camera_det_ms

        # Step 3: Tracking and FSM for all cameras (sequential)
        t0_track = time.perf_counter()

        for camera_id, result in results.items():
            # Get detections for this camera (empty list if no motion)
            detections = detection_results.get(camera_id, [])

            # Convert to tracker format
            track_dets = [
                Detection(
                    x=d.x, y=d.y, width=d.width, height=d.height,
                    confidence=d.confidence, class_id=d.class_id,
                    class_name=d.class_name
                )
                for d in detections
            ]

            # Update tracker
            tracker = self._trackers.get(camera_id)
            if tracker:
                result.tracks = tracker.update(track_dets)

            # Update FSM
            fsm = self._fsms.get(camera_id)
            if fsm:
                result.events = fsm.update(result.tracks)

        tracking_time_ms = (time.perf_counter() - t0_track) * 1000
        per_camera_track_ms = tracking_time_ms / len(frames) if frames else 0

        for result in results.values():
            result.tracking_time_ms = per_camera_track_ms / 2
            result.fsm_time_ms = per_camera_track_ms / 2

        return results

    def reset_camera(self, camera_id: int) -> None:
        """Reset all state for a camera."""
        with self._lock:
            self._motion_gate.reset_camera(camera_id)
            if camera_id in self._trackers:
                self._trackers[camera_id].reset()
            if camera_id in self._fsms:
                self._fsms[camera_id].reset()
            self._frame_counts[camera_id] = 0

    @property
    def stats(self) -> dict:
        """Get pipeline statistics."""
        tracker_stats = {}
        fsm_stats = {}

        for cam_id in self.camera_ids:
            if cam_id in self._trackers:
                tracker_stats[cam_id] = self._trackers[cam_id].stats
            if cam_id in self._fsms:
                fsm_stats[cam_id] = self._fsms[cam_id].stats

        return {
            "device_id": self.device_id,
            "camera_count": len(self.camera_ids),
            "cameras": sorted(self.camera_ids),
            "frame_counts": self._frame_counts.copy(),
            "tracker_stats": tracker_stats,
            "fsm_stats": fsm_stats,
        }


class GPUOrchestrator:
    """Orchestrates multiple GPU pipelines for multi-camera processing.

    Manages camera-to-GPU assignment and provides a unified interface
    for processing frames from any camera.

    Example:
        >>> orchestrator = GPUOrchestrator(device_ids=[0, 1])
        >>> orchestrator.assign_camera(camera_id=1, device_id=0)
        >>> result = orchestrator.process(camera_id=1, frame=frame, timestamp=time.time())
    """

    def __init__(
        self,
        device_ids: Optional[list[int]] = None,
        config: Optional[GPUPipelineConfig] = None,
        cameras_per_gpu: int = 8,
    ):
        """Initialize orchestrator.

        Args:
            device_ids: List of GPU device IDs to use (default: [0])
            config: Pipeline configuration (default: GPUPipelineConfig())
            cameras_per_gpu: Default cameras per GPU for auto-assignment
        """
        self.device_ids = device_ids or [0]
        self.config = config or GPUPipelineConfig()
        self.cameras_per_gpu = cameras_per_gpu

        # Camera to GPU mapping
        self._camera_gpu_map: dict[int, int] = {}

        # GPU pipelines
        self._pipelines: dict[int, GPUPipeline] = {}

        # Initialize pipelines
        for device_id in self.device_ids:
            device_config = GPUPipelineConfig(
                device_id=device_id,
                model_path=self.config.model_path,
                # Motion settings
                motion_history=self.config.motion_history,
                motion_var_threshold=self.config.motion_var_threshold,
                motion_min_percent=self.config.motion_min_percent,
                # Detection settings
                detection_confidence=self.config.detection_confidence,
                detection_nms_threshold=self.config.detection_nms_threshold,
                class_thresholds=self.config.class_thresholds,
                # Tracking settings
                track_high_thresh=self.config.track_high_thresh,
                track_low_thresh=self.config.track_low_thresh,
                track_match_thresh=self.config.track_match_thresh,
                track_buffer=self.config.track_buffer,
                track_min_hits=self.config.track_min_hits,
                track_min_displacement=self.config.track_min_displacement,
                # FSM settings
                fsm_validation_frames=self.config.fsm_validation_frames,
                fsm_velocity_threshold=self.config.fsm_velocity_threshold,
                fsm_stationary_seconds=self.config.fsm_stationary_seconds,
                fsm_parked_seconds=self.config.fsm_parked_seconds,
                # Periodic detection
                periodic_detection_interval=self.config.periodic_detection_interval,
            )
            self._pipelines[device_id] = GPUPipeline(device_config, [])

        self._lock = threading.Lock()

        logger.info(f"GPUOrchestrator initialized with {len(self.device_ids)} GPUs")

    def assign_camera(self, camera_id: int, device_id: Optional[int] = None) -> int:
        """Assign a camera to a GPU.

        Args:
            camera_id: Camera identifier
            device_id: GPU device ID (None = auto-assign)

        Returns:
            The device ID the camera was assigned to
        """
        with self._lock:
            # Remove from existing assignment
            if camera_id in self._camera_gpu_map:
                old_device = self._camera_gpu_map[camera_id]
                self._pipelines[old_device].remove_camera(camera_id)

            # Auto-assign if no device specified
            if device_id is None:
                device_id = self._find_least_loaded_gpu()

            # Assign to new device
            self._camera_gpu_map[camera_id] = device_id
            self._pipelines[device_id].add_camera(camera_id)

            logger.info(f"Camera {camera_id} assigned to GPU {device_id}")
            return device_id

    def _find_least_loaded_gpu(self) -> int:
        """Find the GPU with fewest cameras."""
        counts = {
            dev: len(pipe.camera_ids)
            for dev, pipe in self._pipelines.items()
        }
        return min(counts, key=counts.get)

    def process(
        self,
        camera_id: int,
        frame: np.ndarray,
        timestamp: float,
        skip_detection: bool = False,
    ) -> PipelineResult:
        """Process a frame from any camera.

        Automatically routes to the correct GPU pipeline.

        Args:
            camera_id: Camera identifier
            frame: BGR frame
            timestamp: Frame timestamp
            skip_detection: Skip object detection

        Returns:
            PipelineResult
        """
        # Auto-assign if not assigned
        if camera_id not in self._camera_gpu_map:
            self.assign_camera(camera_id)

        device_id = self._camera_gpu_map[camera_id]
        pipeline = self._pipelines[device_id]

        return pipeline.process(camera_id, frame, timestamp, skip_detection)

    def process_batch(
        self,
        frames: dict[int, np.ndarray],
        timestamp: float,
        skip_detection: bool = False,
    ) -> dict[int, PipelineResult]:
        """Process frames from multiple cameras using batched inference.

        Groups cameras by GPU and uses batched detection for efficiency.
        This is the preferred method for multi-camera processing.

        Args:
            frames: Dict mapping camera_id to BGR frame
            timestamp: Frame timestamp (shared across all frames)
            skip_detection: Skip object detection

        Returns:
            Dict mapping camera_id to PipelineResult
        """
        # Auto-assign unassigned cameras
        for camera_id in frames:
            if camera_id not in self._camera_gpu_map:
                self.assign_camera(camera_id)

        # Group frames by GPU
        gpu_frames: dict[int, dict[int, np.ndarray]] = {
            dev: {} for dev in self.device_ids
        }

        for camera_id, frame in frames.items():
            device_id = self._camera_gpu_map[camera_id]
            gpu_frames[device_id][camera_id] = frame

        # Process each GPU's batch
        all_results: dict[int, PipelineResult] = {}

        for device_id, device_frames in gpu_frames.items():
            if device_frames:
                pipeline = self._pipelines[device_id]
                results = pipeline.process_batch(device_frames, timestamp, skip_detection)
                all_results.update(results)

        return all_results

    def get_pipeline(self, camera_id: int) -> Optional[GPUPipeline]:
        """Get the pipeline for a camera."""
        if camera_id not in self._camera_gpu_map:
            return None
        device_id = self._camera_gpu_map[camera_id]
        return self._pipelines.get(device_id)

    def reset_camera(self, camera_id: int) -> None:
        """Reset all state for a camera."""
        if camera_id in self._camera_gpu_map:
            device_id = self._camera_gpu_map[camera_id]
            self._pipelines[device_id].reset_camera(camera_id)

    @property
    def stats(self) -> dict:
        """Get orchestrator statistics."""
        return {
            "device_ids": self.device_ids,
            "total_cameras": len(self._camera_gpu_map),
            "camera_assignments": self._camera_gpu_map.copy(),
            "pipelines": {
                dev: pipe.stats
                for dev, pipe in self._pipelines.items()
            },
        }


# Global orchestrator instance
_orchestrator: Optional[GPUOrchestrator] = None


def get_orchestrator() -> GPUOrchestrator:
    """Get or create the global GPU orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        # Auto-detect GPUs
        try:
            num_gpus = cv2.cuda.getCudaEnabledDeviceCount()
            device_ids = list(range(num_gpus)) if num_gpus > 0 else [0]
        except Exception:
            device_ids = [0]

        _orchestrator = GPUOrchestrator(device_ids=device_ids)

    return _orchestrator
