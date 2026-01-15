#!/usr/bin/env python3
"""Analyze video for motion detection and object detection.

This script processes a video file in two phases:
1. Motion Detection: Scans entire video with MOG2 and logs motion timestamps
2. Object Detection: Starting from a specified timestamp, logs all detected objects

Results are saved to a JSON file for analysis.

Usage:
    python analyze_motion_and_objects.py /path/to/video.mp4 --start-time 13:50 --output results.json
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use ultralytics for faster GPU inference
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# Fallback to ONNX runtime
try:
    from app.services.ml.tensorrt_inference import TensorRTDetector, COCO_CLASSES
    TENSORRT_AVAILABLE = True
except ImportError:
    TensorRTDetector = None
    COCO_CLASSES = None
    TENSORRT_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class MotionEvent:
    """A motion detection event."""
    frame_number: int
    timestamp_sec: float
    timestamp_str: str
    motion_percent: float
    contour_count: int


@dataclass
class Detection:
    """A single object detection."""
    class_name: str
    class_id: int
    confidence: float
    x: float
    y: float
    width: float
    height: float


@dataclass
class FrameDetections:
    """Detections for a single frame."""
    frame_number: int
    timestamp_sec: float
    timestamp_str: str
    detections: list[Detection] = field(default_factory=list)


@dataclass
class AnalysisResults:
    """Complete analysis results."""
    video_path: str
    video_fps: float
    video_duration_sec: float
    video_frame_count: int
    analysis_timestamp: str
    motion_detection_params: dict
    object_detection_start_sec: float
    motion_events: list[MotionEvent] = field(default_factory=list)
    frame_detections: list[FrameDetections] = field(default_factory=list)


def parse_time_string(time_str: str) -> float:
    """Parse time string (MM:SS or HH:MM:SS) to seconds."""
    parts = time_str.split(":")
    if len(parts) == 2:
        minutes, seconds = map(float, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = map(float, parts)
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid time format: {time_str}. Use MM:SS or HH:MM:SS")


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.mmm."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


class MOG2MotionDetector:
    """CPU-based MOG2 motion detector for video analysis."""

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        min_motion_percent: float = 0.1,
        min_contour_area: int = 500,
        erosion_kernel_size: int = 3,
        dilation_kernel_size: int = 5,
    ):
        self.min_motion_percent = min_motion_percent
        self.min_contour_area = min_contour_area
        self.shadow_threshold = 127

        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

        self._erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size)
        )
        self._dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size)
        )

        self.params = {
            "history": history,
            "var_threshold": var_threshold,
            "detect_shadows": detect_shadows,
            "min_motion_percent": min_motion_percent,
            "min_contour_area": min_contour_area,
            "erosion_kernel_size": erosion_kernel_size,
            "dilation_kernel_size": dilation_kernel_size,
        }

    def process_frame(self, frame: np.ndarray) -> tuple[bool, float, int]:
        """Process a frame and return (motion_detected, motion_percent, contour_count)."""
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply MOG2
        mask = self._mog2.apply(gray)

        # Morphological filtering
        mask = cv2.erode(mask, self._erosion_kernel)
        mask = cv2.dilate(mask, self._dilation_kernel)

        # Remove shadows
        mask = np.where(mask > self.shadow_threshold, 255, 0).astype(np.uint8)

        # Calculate motion percentage
        total_pixels = mask.shape[0] * mask.shape[1]
        motion_pixels = cv2.countNonZero(mask)
        motion_percent = (motion_pixels / total_pixels) * 100

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]

        motion_detected = (
            motion_percent >= self.min_motion_percent and len(valid_contours) > 0
        )

        return motion_detected, motion_percent, len(valid_contours)


def run_motion_analysis(
    video_path: Path,
    results: AnalysisResults,
    start_time_sec: float = 0.0,
    sample_rate: int = 1,
) -> None:
    """Run motion detection from specified timestamp."""
    logger.info(f"Starting motion analysis from {format_timestamp(start_time_sec)}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Seek to start time
    start_frame = int(start_time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    logger.info(f"Seeking to frame {start_frame}")

    detector = MOG2MotionDetector()
    results.motion_detection_params = detector.params

    frame_number = start_frame
    motion_count = 0
    frames_processed = 0
    last_log = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame for speed
        if frames_processed % sample_rate == 0:
            motion_detected, motion_percent, contour_count = detector.process_frame(frame)

            timestamp_sec = frame_number / fps
            # Log every frame's motion status (even if not detected)
            event = MotionEvent(
                frame_number=frame_number,
                timestamp_sec=round(timestamp_sec, 3),
                timestamp_str=format_timestamp(timestamp_sec),
                motion_percent=round(motion_percent, 2),
                contour_count=contour_count,
            )
            # Only record if motion detected
            if motion_detected:
                results.motion_events.append(event)
                motion_count += 1

        frame_number += 1
        frames_processed += 1

        # Progress logging
        if frames_processed - last_log >= 1000:
            progress = (frame_number / total_frames) * 100
            logger.info(
                f"Motion analysis: {progress:.1f}% ({frames_processed} frames), "
                f"motion events: {motion_count}"
            )
            last_log = frames_processed

    cap.release()
    logger.info(f"Motion analysis complete: {motion_count} motion events detected")


def run_object_detection(
    video_path: Path,
    results: AnalysisResults,
    start_time_sec: float,
    model_path: Optional[Path] = None,
    confidence_threshold: float = 0.3,
) -> None:
    """Run object detection from specified timestamp."""
    logger.info(f"Starting object detection from {format_timestamp(start_time_sec)}")

    # Find model - prefer .pt files for ultralytics
    if model_path is None:
        # Look for common model locations
        search_paths = [
            Path("/data/sas1/ronin/ml_models/yolov8l.pt"),
            Path("/data/sas1/ronin/ml_models/yolo11l.pt"),
            Path("./yolov8l.pt"),
            Path("/home/kbentley/dev/ronin-nvr/backend/yolov8l.pt"),
            Path("/data/sas1/ronin/ml_models/yolov8l.onnx"),
            Path("/data/ml_models/yolov8s.onnx"),
        ]
        for p in search_paths:
            if p.exists():
                model_path = p
                break

        if model_path is None:
            # Try to download a default model
            model_path = Path("yolov8l.pt")
            logger.info("No model found, will download yolov8l.pt")

    logger.info(f"Using model: {model_path}")

    # Use ultralytics YOLO if available (GPU accelerated)
    if ULTRALYTICS_AVAILABLE:
        logger.info("Using ultralytics YOLO for GPU-accelerated inference")
        model = YOLO(str(model_path))
        use_ultralytics = True
    elif TENSORRT_AVAILABLE:
        logger.info("Using TensorRT/ONNX Runtime for inference")
        model = TensorRTDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            warmup_iterations=5,
        )
        use_ultralytics = False
    else:
        raise RuntimeError("Neither ultralytics nor TensorRT backend available")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Seek to start time
    start_frame = int(start_time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    logger.info(f"Seeking to frame {start_frame}")

    frame_number = start_frame
    frames_processed = 0
    total_detections = 0
    last_log = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_number / fps
        detections_list = []

        if use_ultralytics:
            # Use ultralytics YOLO
            results_yolo = model(frame, verbose=False, conf=confidence_threshold)
            for r in results_yolo:
                boxes = r.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        box = boxes[i]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = model.names[cls_id]

                        # Convert to normalized coordinates
                        h, w = frame.shape[:2]
                        detections_list.append(Detection(
                            class_name=cls_name,
                            class_id=cls_id,
                            confidence=round(conf, 3),
                            x=round(x1 / w, 4),
                            y=round(y1 / h, 4),
                            width=round((x2 - x1) / w, 4),
                            height=round((y2 - y1) / h, 4),
                        ))
        else:
            # Use TensorRT detector
            dets = model.detect(frame)
            for d in dets:
                detections_list.append(Detection(
                    class_name=d.class_name,
                    class_id=d.class_id,
                    confidence=round(d.confidence, 3),
                    x=round(d.x, 4),
                    y=round(d.y, 4),
                    width=round(d.width, 4),
                    height=round(d.height, 4),
                ))

        if detections_list:
            frame_dets = FrameDetections(
                frame_number=frame_number,
                timestamp_sec=round(timestamp_sec, 3),
                timestamp_str=format_timestamp(timestamp_sec),
                detections=detections_list,
            )
            results.frame_detections.append(frame_dets)
            total_detections += len(detections_list)

        frame_number += 1
        frames_processed += 1

        # Progress logging
        if frames_processed - last_log >= 500:
            progress = (frame_number / total_frames) * 100
            logger.info(
                f"Object detection: {progress:.1f}% ({frames_processed} frames), "
                f"detections: {total_detections}"
            )
            last_log = frames_processed

    cap.release()
    logger.info(
        f"Object detection complete: {frames_processed} frames processed, "
        f"{total_detections} total detections"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze video for motion and object detection"
    )
    parser.add_argument("video_path", type=Path, help="Path to video file")
    parser.add_argument(
        "--start-time", "-s",
        type=str,
        required=True,
        help="Start time for object detection (MM:SS or HH:MM:SS)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON file (default: <video>_analysis.json)"
    )
    parser.add_argument(
        "--model-path", "-m",
        type=Path,
        default=None,
        help="Path to YOLO ONNX model"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3)"
    )
    parser.add_argument(
        "--motion-sample-rate",
        type=int,
        default=1,
        help="Process every Nth frame for motion detection (default: 1)"
    )
    parser.add_argument(
        "--skip-motion",
        action="store_true",
        help="Skip motion detection phase"
    )
    parser.add_argument(
        "--skip-objects",
        action="store_true",
        help="Skip object detection phase"
    )

    args = parser.parse_args()

    if not args.video_path.exists():
        logger.error(f"Video file not found: {args.video_path}")
        sys.exit(1)

    # Parse start time
    start_time_sec = parse_time_string(args.start_time)
    logger.info(f"Object detection will start at {format_timestamp(start_time_sec)}")

    # Output file
    output_path = args.output
    if output_path is None:
        output_path = args.video_path.with_suffix("").with_suffix("_analysis.json")

    # Get video info
    cap = cv2.VideoCapture(str(args.video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()

    # Initialize results
    results = AnalysisResults(
        video_path=str(args.video_path),
        video_fps=fps,
        video_duration_sec=round(duration, 3),
        video_frame_count=frame_count,
        analysis_timestamp=datetime.now().isoformat(),
        motion_detection_params={},
        object_detection_start_sec=start_time_sec,
    )

    logger.info(f"Video: {args.video_path}")
    logger.info(f"  FPS: {fps:.2f}, Duration: {format_timestamp(duration)}, Frames: {frame_count}")

    # Phase 1: Motion detection
    if not args.skip_motion:
        run_motion_analysis(
            args.video_path,
            results,
            start_time_sec=start_time_sec,
            sample_rate=args.motion_sample_rate,
        )
    else:
        logger.info("Skipping motion detection phase")

    # Phase 2: Object detection
    if not args.skip_objects:
        run_object_detection(
            args.video_path,
            results,
            start_time_sec,
            model_path=args.model_path,
            confidence_threshold=args.confidence,
        )
    else:
        logger.info("Skipping object detection phase")

    # Save results
    logger.info(f"Saving results to {output_path}")

    def convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert_to_serializable(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    logger.info("Analysis complete!")
    logger.info(f"  Motion events: {len(results.motion_events)}")
    logger.info(f"  Frames with detections: {len(results.frame_detections)}")


if __name__ == "__main__":
    main()
