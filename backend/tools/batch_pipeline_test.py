#!/usr/bin/env python3
"""Batch test the full detection pipeline on daytime videos.

Walks a directory tree for daytime videos, runs the detection pipeline,
and saves results (JSON + annotated frames) for each video.

Daytime is determined by filename (UTC) adjusted for Boise, Idaho in mid-January:
- Sunrise: ~8:00 AM MST = 15:00 UTC
- Sunset: ~5:30 PM MST = 00:30 UTC (next day)

Usage:
    python batch_pipeline_test.py /opt3/ronin/storage --output ./batch_results
    python batch_pipeline_test.py /opt3/ronin/storage --cameras House_South,Hangar_East
    python batch_pipeline_test.py /opt3/ronin/storage --max-videos 10  # Test mode
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.object_fsm import (
    ObjectStateMachine,
    ObjectState,
    EventType,
)
from app.services.ml.tracker import ByteTracker, Detection as TrackerDetection
from app.services.ml.motion_gate import MotionGate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Class whitelist - only these classes generate alerts
CLASS_WHITELIST = {
    # Vehicles
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'boat',
    # People
    'person',
    # Animals
    'dog', 'cat', 'horse', 'sheep', 'cow', 'bear', 'bird',
}

# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Daytime hours in UTC for Boise, Idaho mid-January
# MST = UTC-7, sunrise ~8 AM MST (15:00 UTC), sunset ~5:30 PM MST (00:30 UTC)
DAYTIME_UTC_HOURS = list(range(15, 24)) + [0]  # 15:00-23:59 and 00:00-00:59 UTC

# Maximum realistic FPS for security cameras (used to detect bad metadata)
# Most security cameras are 15-30 FPS. Anything above 60 is suspicious.
MAX_REALISTIC_FPS = 60.0
DEFAULT_FALLBACK_FPS = 15.0

# Minimum object size (normalized 0-1) to filter tiny false positives
# 0.01 = 1% of frame dimension, e.g., 38x21 pixels on 4K (3840x2160)
# Roof vents, distant birds, etc. are typically smaller than this
MIN_OBJECT_WIDTH = 0.02   # 2% of frame width (~77 pixels on 4K)
MIN_OBJECT_HEIGHT = 0.02  # 2% of frame height (~43 pixels on 4K)


# ============================================================================
# Detection Classes
# ============================================================================

@dataclass
class Detection:
    """Single detection from YOLO."""
    class_id: int
    class_name: str
    confidence: float
    x: float  # center x (normalized 0-1)
    y: float  # center y (normalized 0-1)
    width: float  # width (normalized 0-1)
    height: float  # height (normalized 0-1)


@dataclass
class EventRecord:
    """Record of a notification event."""
    event_type: str  # "arrival" or "departure"
    class_name: str
    track_id: int
    timestamp: float  # seconds into video
    frame_number: int
    confidence: float
    bbox: tuple[float, float, float, float]  # x, y, w, h normalized
    frame_path: Optional[str] = None  # Path to saved frame


class SimpleYOLODetector:
    """YOLO detector using ONNX Runtime with GPU acceleration."""

    def __init__(self, model_path: str, confidence: float = 0.25):
        import onnxruntime as ort

        self.confidence = confidence

        # Check available providers
        available = ort.get_available_providers()
        logger.info(f"ONNX Runtime available providers: {available}")

        # Use CUDA (most reliable GPU option)
        # TensorRT is faster but has library dependency issues
        if 'CUDAExecutionProvider' in available:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger.info("Using CUDA - expect ~10-15ms per frame")
        else:
            logger.warning("CUDA not available! Using CPU only - this will be SLOW")
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)

        # Log which provider is actually being used
        actual_providers = self.session.get_providers()
        logger.info(f"ONNX Runtime using providers: {actual_providers}")

        if 'CUDAExecutionProvider' not in actual_providers:
            logger.warning("WARNING: Running on CPU - expect ~1 second per frame")

        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on frame."""
        resized = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        outputs = self.session.run(None, {self.input_name: blob})
        output = outputs[0]

        if len(output.shape) == 3:
            output = output[0].T

        detections = []
        for row in output:
            x_center, y_center, width, height = row[:4]
            class_scores = row[4:]

            max_idx = np.argmax(class_scores)
            max_score = class_scores[max_idx]

            if max_score >= self.confidence:
                class_name = COCO_CLASSES[max_idx] if max_idx < len(COCO_CLASSES) else f"class_{max_idx}"
                detections.append(Detection(
                    class_id=int(max_idx),
                    class_name=class_name,
                    confidence=float(max_score),
                    x=float(x_center) / 640.0,
                    y=float(y_center) / 640.0,
                    width=float(width) / 640.0,
                    height=float(height) / 640.0,
                ))

        return self._nms(detections)

    def _nms(self, detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
        """NMS - filter overlapping detections per class."""
        by_class = defaultdict(list)
        for d in detections:
            by_class[d.class_name].append(d)

        result = []
        for class_name, dets in by_class.items():
            sorted_dets = sorted(dets, key=lambda x: -x.confidence)
            keep = []

            while sorted_dets:
                best = sorted_dets.pop(0)
                keep.append(best)

                remaining = []
                for d in sorted_dets:
                    iou = self._calculate_iou(best, d)
                    if iou < iou_threshold:
                        remaining.append(d)
                sorted_dets = remaining

            result.extend(keep)

        return result

    def _calculate_iou(self, d1: Detection, d2: Detection) -> float:
        """Calculate IoU between two detections."""
        x1_1 = d1.x - d1.width / 2
        y1_1 = d1.y - d1.height / 2
        x2_1 = d1.x + d1.width / 2
        y2_1 = d1.y + d1.height / 2

        x1_2 = d2.x - d2.width / 2
        y1_2 = d2.y - d2.height / 2
        x2_2 = d2.x + d2.width / 2
        y2_2 = d2.y + d2.height / 2

        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = d1.width * d1.height
        area2 = d2.width * d2.height
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


# ============================================================================
# FFmpeg Frame Extraction
# ============================================================================

def probe_video(video_path: str) -> dict:
    """Probe video for metadata using FFprobe.

    Returns:
        Dict with src_width, src_height, src_fps, duration
    """
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration,r_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]

    try:
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if probe_result.returncode != 0:
            logger.error(f"FFprobe failed: {probe_result.stderr[:200]}")
            return {}

        probe_data = json.loads(probe_result.stdout)

        # Get source dimensions
        stream = probe_data.get("streams", [{}])[0]
        src_width = int(stream.get("width", 1920))
        src_height = int(stream.get("height", 1080))

        # Get duration (try stream first, then format)
        duration = float(stream.get("duration", 0))
        if duration == 0:
            duration = float(probe_data.get("format", {}).get("duration", 0))

        # Parse frame rate
        fps_str = stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            src_fps = float(num) / float(den) if float(den) > 0 else 30.0
        else:
            src_fps = float(fps_str)

        return {
            "src_width": src_width,
            "src_height": src_height,
            "src_fps": src_fps,
            "duration": duration,
        }

    except Exception as e:
        logger.error(f"FFprobe error: {e}")
        return {}


def stream_frames_ffmpeg(
    video_path: str,
    fps: float = 2.0,
    scale_height: int = 720,
):
    """Stream frames from video using FFmpeg at target FPS and resolution.

    Generator that yields frames one at a time to avoid loading all into memory.

    Much faster than OpenCV because FFmpeg:
    - Only decodes frames at target FPS (skips intermediate frames)
    - Scales during decode (no separate resize step)
    - Uses hardware acceleration when available

    Args:
        video_path: Path to video file
        fps: Target frames per second to extract
        scale_height: Target height (width scales proportionally)

    Yields:
        (frame_idx, frame, metadata) tuples
    """
    # First probe the video
    metadata = probe_video(video_path)
    if not metadata:
        return

    src_width = metadata["src_width"]
    src_height = metadata["src_height"]
    duration = metadata["duration"]

    # Calculate scaled dimensions
    scale_ratio = scale_height / src_height
    scaled_width = int(src_width * scale_ratio)
    scaled_width = scaled_width + (scaled_width % 2)  # Round to even
    scaled_height = scale_height

    metadata["scaled_width"] = scaled_width
    metadata["scaled_height"] = scaled_height
    metadata["target_fps"] = fps

    logger.info(f"  FFmpeg streaming: {src_width}x{src_height} -> {scaled_width}x{scaled_height} @ {fps}fps")

    # Start FFmpeg process with pipe output
    extract_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"scale={scaled_width}:{scaled_height},fps={fps}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]

    frame_size = scaled_width * scaled_height * 3
    frame_idx = 0

    try:
        proc = subprocess.Popen(
            extract_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=frame_size * 10,  # Buffer ~10 frames
        )

        while True:
            # Read exactly one frame
            raw_data = proc.stdout.read(frame_size)
            if len(raw_data) < frame_size:
                break

            frame = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                (scaled_height, scaled_width, 3)
            ).copy()  # Copy to make it writable

            yield frame_idx, frame, metadata
            frame_idx += 1

        proc.wait()

        if proc.returncode != 0:
            stderr = proc.stderr.read().decode()[:200]
            logger.warning(f"FFmpeg ended with code {proc.returncode}: {stderr}")

        logger.info(f"  Streamed {frame_idx} frames ({duration:.1f}s video)")
        metadata["num_frames"] = frame_idx

    except Exception as e:
        logger.error(f"FFmpeg streaming error: {e}")
        if 'proc' in locals():
            proc.kill()


# ============================================================================
# Video Discovery
# ============================================================================

def parse_video_time(filename: str) -> Optional[int]:
    """Extract UTC hour from video filename (e.g., '15-30-00.mp4' -> 15)."""
    match = re.match(r'^(\d{2})-\d{2}-\d{2}\.mp4$', filename)
    if match:
        return int(match.group(1))
    return None


def is_daytime_video(filename: str) -> bool:
    """Check if video filename indicates daytime (for Boise, Idaho mid-January)."""
    hour = parse_video_time(filename)
    if hour is None:
        return False
    return hour in DAYTIME_UTC_HOURS


def find_daytime_videos(
    root_dir: str,
    cameras: Optional[list[str]] = None,
    max_videos: Optional[int] = None,
) -> list[dict]:
    """Find all daytime videos in the storage directory.

    Expected structure: {root}/{CameraName}/{date}/{HH-MM-SS}.mp4

    Returns list of dicts with 'path', 'camera', 'date', 'filename'
    """
    videos = []
    root_path = Path(root_dir)

    if not root_path.exists():
        logger.error(f"Directory not found: {root_dir}")
        return videos

    # Walk camera directories
    for camera_dir in sorted(root_path.iterdir()):
        if not camera_dir.is_dir():
            continue

        camera_name = camera_dir.name

        # Filter by camera if specified
        if cameras and camera_name not in cameras:
            continue

        # Walk date directories
        for date_dir in sorted(camera_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            date_str = date_dir.name

            # Find video files
            for video_file in sorted(date_dir.glob("*.mp4")):
                if is_daytime_video(video_file.name):
                    videos.append({
                        'path': str(video_file),
                        'camera': camera_name,
                        'date': date_str,
                        'filename': video_file.name,
                    })

                    if max_videos and len(videos) >= max_videos:
                        return videos

    return videos


# ============================================================================
# Frame Annotation
# ============================================================================

def annotate_frame(
    frame: np.ndarray,
    event_type: str,
    class_name: str,
    track_id: int,
    confidence: float,
    bbox: tuple[float, float, float, float],
    timestamp: float,
    all_detections: list = None,
) -> np.ndarray:
    """Annotate frame with event information and bounding boxes."""
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Colors
    if event_type == "arrival":
        color = (0, 255, 0)  # Green for arrival
        text_color = (0, 200, 0)
    else:
        color = (0, 0, 255)  # Red for departure
        text_color = (0, 0, 200)

    # Draw main event bbox (x, y is top-left corner, not center)
    x, y, bw, bh = bbox
    x1 = int(x * w)
    y1 = int(y * h)
    x2 = int((x + bw) * w)
    y2 = int((y + bh) * h)

    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

    # Draw all other detections in gray
    if all_detections:
        for det in all_detections:
            dx1 = int((det.x - det.width/2) * w)
            dy1 = int((det.y - det.height/2) * h)
            dx2 = int((det.x + det.width/2) * w)
            dy2 = int((det.y + det.height/2) * h)
            cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), (128, 128, 128), 1)
            cv2.putText(annotated, f"{det.class_name} {det.confidence:.0%}",
                       (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

    # Event label
    label = f"{event_type.upper()}: {class_name}"
    cv2.putText(annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Info text at top
    info_text = f"Track {track_id} | Conf: {confidence:.0%} | Time: {timestamp:.1f}s"
    cv2.putText(annotated, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    # Event type banner
    banner_text = f"EVENT: {event_type.upper()}"
    cv2.putText(annotated, banner_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return annotated


# ============================================================================
# Pipeline Processing
# ============================================================================

def process_video(
    video_path: str,
    output_dir: str,
    detector: SimpleYOLODetector,
    confidence: float = 0.25,
    detection_fps: float = 2.0,
    frames_subdir: str = None,
) -> dict:
    """Process a single video through the full pipeline.

    Args:
        video_path: Path to input video
        output_dir: Base output directory
        detector: YOLO detector instance
        confidence: Detection confidence threshold
        detection_fps: Frames per second to process
        frames_subdir: Subdirectory name for frames (default: video filename stem)

    Returns dict with summary and list of events.
    """
    video_name = Path(video_path).stem
    subdir_name = frames_subdir or video_name
    frames_dir = Path(output_dir) / "frames" / subdir_name
    frames_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing: {video_path}")

    # Probe video first to get metadata
    metadata = probe_video(video_path)
    if not metadata:
        logger.error(f"Cannot probe video: {video_path}")
        return {"error": f"Cannot probe video: {video_path}"}

    video_duration = metadata.get("duration", 0)
    video_fps = metadata.get("src_fps", 15.0)
    expected_frames = int(video_duration * detection_fps)

    # Initialize tracker and FSM
    # track_buffer should match lost_seconds: 30s * 2fps = 60 frames
    tracker = ByteTracker(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        match_thresh=0.8,
        track_buffer=60,  # 30 seconds at 2fps
        min_hits=3,
        min_displacement=0.0,
    )

    fsm = ObjectStateMachine(
        validation_frames=5,  # ~2.5 seconds at 2fps to confirm arrival
        velocity_threshold=0.002,
        stationary_seconds=10.0,
        parked_seconds=60.0,
        lost_seconds=30.0,  # Wait 30 seconds before declaring departure
        fps=detection_fps,
    )

    # Initialize motion gate to skip inference on static scenes
    motion_gate = MotionGate(
        threshold=25.0,
        min_area=500,
        min_percent=0.1,
    )

    # Process frames (already at 720p from FFmpeg)
    events = []
    detection_counts = defaultdict(int)
    filtered_counts = defaultdict(int)
    size_filtered_counts = defaultdict(int)
    frames_processed = 0
    frames_skipped_no_motion = 0
    previous_frame = None

    # Keep recent frames for event capture (only store when detections present)
    recent_frames = {}  # frame_idx -> (frame, detections)
    max_recent = 30
    last_frame = None
    last_frame_idx = 0
    total_frames = 0

    # Stream frames from FFmpeg (already at 720p, memory efficient)
    start_time = time.time()
    for frame_idx, frame, meta in stream_frames_ffmpeg(video_path, fps=detection_fps, scale_height=720):
        # Update metadata from stream
        if frame_idx == 0:
            metadata.update(meta)
        total_frames = frame_idx + 1
        # Calculate timestamp for this frame
        timestamp = frame_idx / detection_fps

        # Check for motion before running expensive YOLO inference
        motion_result = motion_gate.check(frame, previous_frame)
        previous_frame = frame  # No copy needed - each frame is independent

        if not motion_result.should_run_inference:
            # No motion - skip YOLO but still update tracker/FSM with empty detections
            frames_skipped_no_motion += 1
            filtered_detections = []
        else:
            # Motion detected - run YOLO
            raw_detections = detector.detect(frame)

            for d in raw_detections:
                detection_counts[d.class_name] += 1

            # Apply class whitelist
            whitelisted = [d for d in raw_detections if d.class_name in CLASS_WHITELIST]

            # Apply minimum size filter to remove tiny false positives
            filtered_detections = []
            for d in whitelisted:
                if d.width >= MIN_OBJECT_WIDTH and d.height >= MIN_OBJECT_HEIGHT:
                    filtered_detections.append(d)
                    filtered_counts[d.class_name] += 1
                else:
                    size_filtered_counts[d.class_name] += 1

        # Only store frames with detections (saves memory)
        if filtered_detections:
            recent_frames[frame_idx] = (frame, filtered_detections)
            if len(recent_frames) > max_recent:
                oldest = min(recent_frames.keys())
                del recent_frames[oldest]

        # Always keep last frame for potential departure events
        last_frame = frame
        last_frame_idx = frame_idx

        # Convert to tracker format
        tracker_dets = [
            TrackerDetection(
                x=d.x - d.width/2,
                y=d.y - d.height/2,
                width=d.width,
                height=d.height,
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
            )
            for d in filtered_detections
        ]

        # Update tracker and FSM
        tracks = tracker.update(tracker_dets)
        fsm_events = fsm.update(tracks)

        # Process events
        for event in fsm_events:
            if event.event_type in (EventType.ARRIVAL, EventType.DEPARTURE):
                # Get bbox from track
                bbox = event.bbox if event.bbox else (0.5, 0.5, 0.1, 0.1)

                # Save annotated frame
                frame_filename = f"{event.event_type.value}_{event.track_id}_{frame_idx}.jpg"
                frame_path = frames_dir / frame_filename

                annotated = annotate_frame(
                    frame,
                    event.event_type.value,
                    event.class_name,
                    event.track_id,
                    event.confidence,
                    bbox,
                    timestamp,
                    filtered_detections,
                )
                cv2.imwrite(str(frame_path), annotated)

                # Record event
                event_record = {
                    "event_type": event.event_type.value,
                    "class_name": event.class_name,
                    "track_id": event.track_id,
                    "timestamp": timestamp,
                    "frame_number": frame_idx,
                    "confidence": event.confidence,
                    "bbox": list(bbox),
                    "frame_path": str(frame_path.relative_to(Path(output_dir))),
                }
                events.append(event_record)

                logger.info(f"  [{timestamp:.1f}s] {event.event_type.value.upper()}: "
                           f"{event.class_name} (track {event.track_id})")

        frames_processed += 1

        # Progress update
        if frames_processed % 100 == 0:
            elapsed = time.time() - start_time
            fps_actual = frames_processed / elapsed if elapsed > 0 else 0
            progress = (frame_idx / expected_frames) * 100 if expected_frames > 0 else 0
            logger.info(f"  Progress: {progress:.1f}% ({frames_processed} frames, {fps_actual:.1f} fps)")

    # Final FSM update for departures
    for _ in range(10):
        time.sleep(0.5)
        fsm_events = fsm.update([])
        for event in fsm_events:
            if event.event_type == EventType.DEPARTURE:
                bbox = event.bbox if event.bbox else (0.5, 0.5, 0.1, 0.1)

                # Use last_frame for departure annotation
                if last_frame is not None:
                    departure_timestamp = last_frame_idx / detection_fps

                    frame_filename = f"{event.event_type.value}_{event.track_id}_{last_frame_idx}.jpg"
                    frame_path = frames_dir / frame_filename

                    annotated = annotate_frame(
                        last_frame,
                        event.event_type.value,
                        event.class_name,
                        event.track_id,
                        event.confidence,
                        bbox,
                        departure_timestamp,
                        [],  # No current detections for departure
                    )
                    cv2.imwrite(str(frame_path), annotated)

                    event_record = {
                        "event_type": event.event_type.value,
                        "class_name": event.class_name,
                        "track_id": event.track_id,
                        "timestamp": departure_timestamp,
                        "frame_number": last_frame_idx,
                        "confidence": event.confidence,
                        "bbox": list(bbox),
                        "frame_path": str(frame_path.relative_to(Path(output_dir))),
                    }
                    events.append(event_record)

                    logger.info(f"  [END] DEPARTURE: {event.class_name} (track {event.track_id})")

    # Summary
    arrivals = [e for e in events if e["event_type"] == "arrival"]
    departures = [e for e in events if e["event_type"] == "departure"]

    # Calculate actual processing speed
    total_elapsed = time.time() - start_time
    actual_fps = frames_processed / total_elapsed if total_elapsed > 0 else 0

    result = {
        "video_path": video_path,
        "video_name": video_name,
        "duration_seconds": video_duration,
        "frames_extracted": total_frames,
        "frames_processed": frames_processed,
        "frames_skipped_no_motion": frames_skipped_no_motion,
        "processing_fps": round(actual_fps, 1),
        "source_fps": video_fps,
        "source_resolution": f"{metadata.get('src_width', 0)}x{metadata.get('src_height', 0)}",
        "processing_resolution": f"{metadata.get('scaled_width', 0)}x{metadata.get('scaled_height', 0)}",
        "detection_fps": detection_fps,
        "confidence_threshold": confidence,
        "raw_detection_counts": dict(detection_counts),
        "filtered_detection_counts": dict(filtered_counts),
        "size_filtered_counts": dict(size_filtered_counts),
        "total_arrivals": len(arrivals),
        "total_departures": len(departures),
        "events": events,
    }

    motion_skip_pct = (frames_skipped_no_motion / frames_processed * 100) if frames_processed > 0 else 0
    logger.info(f"  Complete: {len(arrivals)} arrivals, {len(departures)} departures ({actual_fps:.1f} fps)")
    logger.info(f"  Motion gating: {frames_skipped_no_motion}/{frames_processed} frames skipped ({motion_skip_pct:.0f}%)")
    if size_filtered_counts:
        logger.info(f"  Size filtered: {dict(size_filtered_counts)}")

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch test detection pipeline on daytime videos"
    )
    parser.add_argument(
        "storage_dir",
        help="Root storage directory (e.g., /opt3/ronin/storage)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./batch_results",
        help="Output directory for results (default: ./batch_results)"
    )
    parser.add_argument(
        "--cameras",
        help="Comma-separated list of cameras to process (default: all)"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        help="Maximum number of videos to process (for testing)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Detection FPS (default: 2.0)"
    )
    parser.add_argument(
        "--yolo",
        default="/opt3/ronin/ml_models/yolov8l.onnx",
        help="Path to YOLO model (use static .onnx for TensorRT, not _dynamic)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip already processed videos)"
    )
    args = parser.parse_args()

    # Parse cameras
    cameras = None
    if args.cameras:
        cameras = [c.strip() for c in args.cameras.split(",")]

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find videos
    logger.info(f"Scanning for daytime videos in: {args.storage_dir}")
    videos = find_daytime_videos(
        args.storage_dir,
        cameras=cameras,
        max_videos=args.max_videos,
    )

    if not videos:
        logger.error("No daytime videos found!")
        return 1

    logger.info(f"Found {len(videos)} daytime videos")

    # Check for already processed (if resuming)
    processed_videos = set()
    if args.resume:
        for json_file in (output_dir / "json").glob("*.json"):
            processed_videos.add(json_file.stem)
        logger.info(f"Resuming: {len(processed_videos)} videos already processed")

    # Initialize detector
    logger.info(f"Loading YOLO model: {args.yolo}")
    detector = SimpleYOLODetector(args.yolo, confidence=args.confidence)

    # Create subdirectories
    (output_dir / "json").mkdir(exist_ok=True)
    (output_dir / "frames").mkdir(exist_ok=True)

    # Process videos
    start_time = time.time()
    total_events = 0
    results_summary = []

    for i, video_info in enumerate(videos):
        video_path = video_info['path']
        video_name = Path(video_path).stem
        camera = video_info['camera']
        date = video_info['date']

        # Unique name for this video's output
        output_name = f"{camera}_{date}_{video_name}"
        final_frames_dir = output_dir / "frames" / output_name
        temp_frames_dir = output_dir / "frames" / f"{output_name}.tmp"

        # Skip if already processed (check for final frames directory)
        if final_frames_dir.exists():
            logger.info(f"[{i+1}/{len(videos)}] Skipping (already exists): {output_name}")
            continue

        # Also skip if in processed_videos set (from --resume JSON check)
        if output_name in processed_videos:
            logger.info(f"[{i+1}/{len(videos)}] Skipping (already processed): {output_name}")
            continue

        # Clean up any leftover temp directory from interrupted run
        if temp_frames_dir.exists():
            shutil.rmtree(temp_frames_dir)
            logger.info(f"  Cleaned up incomplete temp directory")

        logger.info(f"[{i+1}/{len(videos)}] Processing: {camera}/{date}/{video_name}.mp4")

        try:
            result = process_video(
                video_path,
                str(output_dir),
                detector,
                confidence=args.confidence,
                detection_fps=args.fps,
                frames_subdir=f"{output_name}.tmp",
            )

            # Add metadata
            result["camera"] = camera
            result["date"] = date
            result["processed_at"] = datetime.now().isoformat()

            # Rename temp directory to final name (atomic completion marker)
            if temp_frames_dir.exists():
                temp_frames_dir.rename(final_frames_dir)

            # Save JSON (only after frames are finalized)
            json_path = output_dir / "json" / f"{output_name}.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)

            total_events += len(result.get("events", []))

            results_summary.append({
                "video": output_name,
                "arrivals": result.get("total_arrivals", 0),
                "departures": result.get("total_departures", 0),
            })

        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            # Clean up temp directory on error
            if temp_frames_dir.exists():
                shutil.rmtree(temp_frames_dir)
                logger.info(f"  Cleaned up temp directory after error")

        # Progress estimate
        elapsed = time.time() - start_time
        if i > 0:
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(videos) - i - 1)
            logger.info(f"  Estimated time remaining: {remaining/60:.1f} minutes")

    # Final summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Videos processed: {len(results_summary)}")
    logger.info(f"Total events: {total_events}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Results saved to: {output_dir}")

    # Save summary
    summary_path = output_dir / "summary.json"
    summary = {
        "processed_at": datetime.now().isoformat(),
        "storage_dir": args.storage_dir,
        "videos_found": len(videos),
        "videos_processed": len(results_summary),
        "total_events": total_events,
        "elapsed_seconds": elapsed,
        "confidence_threshold": args.confidence,
        "detection_fps": args.fps,
        "results": results_summary,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
