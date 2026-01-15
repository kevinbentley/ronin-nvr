#!/usr/bin/env python3
"""Test FSM and tracking on a video file.

This script runs the full detection → tracking → FSM pipeline on a video
and outputs detailed logs for comparison with ground truth detections.

Usage:
    python test_fsm_tracking.py /path/to/video.mp4 --start-time 13:50 --output fsm_results.json
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.tracker import ByteTracker, Detection, TrackedObject
from app.services.ml.object_fsm import (
    ObjectStateMachine,
    ObjectState,
    EventType,
    ObjectEvent,
)

# Use ultralytics for GPU inference
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Enable debug logging for tracker and FSM
logging.getLogger("app.services.ml.tracker").setLevel(logging.DEBUG)
logging.getLogger("app.services.ml.object_fsm").setLevel(logging.DEBUG)


@dataclass
class FrameResult:
    """Result for a single frame."""
    frame_number: int
    timestamp_sec: float
    timestamp_str: str
    detections: list[dict] = field(default_factory=list)
    tracks: list[dict] = field(default_factory=list)
    fsm_events: list[dict] = field(default_factory=list)
    fsm_state: dict = field(default_factory=dict)


@dataclass
class FSMTestResults:
    """Complete FSM test results."""
    video_path: str
    video_fps: float
    start_time_sec: float
    analysis_timestamp: str
    tracker_config: dict = field(default_factory=dict)
    fsm_config: dict = field(default_factory=dict)
    frames: list[FrameResult] = field(default_factory=list)
    all_events: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


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


def event_to_dict(event: ObjectEvent) -> dict:
    """Convert ObjectEvent to serializable dict."""
    return {
        "event_type": event.event_type.value,
        "track_id": event.track_id,
        "class_name": event.class_name,
        "class_id": event.class_id,
        "timestamp": event.timestamp,
        "old_state": event.old_state.value if event.old_state else None,
        "new_state": event.new_state.value if event.new_state else None,
        "bbox": list(event.bbox) if event.bbox else None,
        "confidence": event.confidence,
        "duration_seconds": event.duration_seconds,
    }


def track_to_dict(track: TrackedObject) -> dict:
    """Convert TrackedObject to serializable dict."""
    return {
        "track_id": track.track_id,
        "class_name": track.class_name,
        "class_id": track.class_id,
        "x": round(track.x, 4),
        "y": round(track.y, 4),
        "width": round(track.width, 4),
        "height": round(track.height, 4),
        "confidence": round(track.confidence, 3),
        "velocity_x": round(track.velocity_x, 6),
        "velocity_y": round(track.velocity_y, 6),
        "state": track.state.value,
        "hits": track.hits,
        "age": track.age,
        "time_since_update": track.time_since_update,
        "max_displacement": round(track.max_displacement, 6),
    }


class SimulatedTime:
    """Simulate time based on video timestamps instead of real time.

    This allows the FSM to use video time rather than wall clock time,
    which is essential for testing against recorded video.
    """

    def __init__(self, start_time: float = 0.0):
        self._current_time = start_time
        self._original_time = time.time

    def set_time(self, t: float) -> None:
        """Set the current simulated time."""
        self._current_time = t

    def time(self) -> float:
        """Return simulated time."""
        return self._current_time

    def patch(self) -> None:
        """Patch time.time() with simulated time."""
        import app.services.ml.object_fsm as fsm_module
        fsm_module.time = type('SimTime', (), {'time': self.time})()

    def unpatch(self) -> None:
        """Restore original time.time()."""
        import app.services.ml.object_fsm as fsm_module
        fsm_module.time = time


def run_fsm_test(
    video_path: Path,
    start_time_sec: float,
    model_path: Optional[Path] = None,
    confidence_threshold: float = 0.3,
    fps_override: Optional[float] = None,
) -> FSMTestResults:
    """Run FSM test on video."""

    logger.info(f"Starting FSM test from {format_timestamp(start_time_sec)}")

    # Find model
    if model_path is None:
        search_paths = [
            Path("/data/sas1/ronin/ml_models/yolov8l.pt"),
            Path("./yolov8l.pt"),
            Path("/home/kbentley/dev/ronin-nvr/backend/yolov8l.pt"),
        ]
        for p in search_paths:
            if p.exists():
                model_path = p
                break
        if model_path is None:
            model_path = Path("yolov8l.pt")

    logger.info(f"Using model: {model_path}")

    if not ULTRALYTICS_AVAILABLE:
        raise RuntimeError("ultralytics not available")

    # Initialize YOLO
    yolo = YOLO(str(model_path))

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = fps_override or cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize tracker and FSM with video FPS
    tracker = ByteTracker(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        match_thresh=0.8,
        track_buffer=30,
        min_hits=3,
        min_displacement=0.0,
    )

    fsm = ObjectStateMachine(
        validation_frames=10,
        velocity_threshold=0.002,
        displacement_threshold=0.02,
        stationary_seconds=10.0,
        parked_seconds=300.0,
        lost_seconds=5.0,
        loitering_seconds=60.0,
        fps=fps,
    )

    # Create simulated time and patch
    sim_time = SimulatedTime(start_time_sec)
    sim_time.patch()

    # Seek to start
    start_frame = int(start_time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    logger.info(f"Seeking to frame {start_frame} (FPS: {fps:.2f})")

    # Results
    results = FSMTestResults(
        video_path=str(video_path),
        video_fps=fps,
        start_time_sec=start_time_sec,
        analysis_timestamp=datetime.now().isoformat(),
        tracker_config={
            "track_high_thresh": tracker.track_high_thresh,
            "track_low_thresh": tracker.track_low_thresh,
            "match_thresh": tracker.match_thresh,
            "track_buffer": tracker.track_buffer,
            "min_hits": tracker.min_hits,
            "min_displacement": tracker.min_displacement,
        },
        fsm_config={
            "validation_frames": fsm.validation_frames,
            "velocity_threshold": fsm.velocity_threshold,
            "displacement_threshold": fsm.displacement_threshold,
            "stationary_seconds": fsm.stationary_seconds,
            "parked_seconds": fsm.parked_seconds,
            "lost_seconds": fsm.lost_seconds,
            "fps": fsm.fps,
        },
    )

    frame_number = start_frame
    frames_processed = 0
    total_detections = 0
    total_tracks = 0
    total_events = 0
    last_log = 0

    # Event counters
    event_counts = {et.value: 0 for et in EventType}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_sec = frame_number / fps
            sim_time.set_time(timestamp_sec)

            # Run YOLO detection
            yolo_results = yolo(frame, verbose=False, conf=confidence_threshold)

            detections = []
            detection_dicts = []

            for r in yolo_results:
                boxes = r.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        box = boxes[i]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = yolo.names[cls_id]

                        h, w = frame.shape[:2]

                        det = Detection(
                            x=float(x1 / w),
                            y=float(y1 / h),
                            width=float((x2 - x1) / w),
                            height=float((y2 - y1) / h),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name,
                        )
                        detections.append(det)
                        detection_dicts.append({
                            "class_name": cls_name,
                            "class_id": cls_id,
                            "confidence": round(conf, 3),
                            "x": round(det.x, 4),
                            "y": round(det.y, 4),
                            "width": round(det.width, 4),
                            "height": round(det.height, 4),
                        })

            # Run tracker
            tracks = tracker.update(detections)
            track_dicts = [track_to_dict(t) for t in tracks]

            # Run FSM
            events = fsm.update(tracks)
            event_dicts = [event_to_dict(e) for e in events]

            # Count events
            for e in events:
                event_counts[e.event_type.value] += 1
                results.all_events.append({
                    "frame_number": frame_number,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "timestamp_str": format_timestamp(timestamp_sec),
                    **event_to_dict(e),
                })

            # Get FSM state
            fsm_state = {
                "stats": fsm.stats,
                "active_objects": [
                    {
                        "track_id": lc.track_id,
                        "class_name": lc.class_name,
                        "state": lc.state.value,
                        "age_seconds": round(lc.age_seconds, 2),
                        "time_in_state": round(lc.time_in_state, 2),
                        "has_ever_moved": lc.has_ever_moved,
                        "displacement": round(lc.displacement, 4),
                        "speed": round(lc.speed, 6),
                    }
                    for lc in fsm._objects.values()
                ],
            }

            # Record frame result
            frame_result = FrameResult(
                frame_number=frame_number,
                timestamp_sec=round(timestamp_sec, 3),
                timestamp_str=format_timestamp(timestamp_sec),
                detections=detection_dicts,
                tracks=track_dicts,
                fsm_events=event_dicts,
                fsm_state=fsm_state,
            )
            results.frames.append(frame_result)

            # Update counters
            total_detections += len(detections)
            total_tracks += len(tracks)
            total_events += len(events)
            frame_number += 1
            frames_processed += 1

            # Progress logging
            if frames_processed - last_log >= 500:
                progress = (frame_number / total_frames) * 100
                logger.info(
                    f"FSM test: {progress:.1f}% ({frames_processed} frames), "
                    f"detections: {total_detections}, tracks: {total_tracks}, "
                    f"events: {total_events}"
                )
                last_log = frames_processed

    finally:
        sim_time.unpatch()
        cap.release()

    # Summary
    results.summary = {
        "frames_processed": frames_processed,
        "total_detections": total_detections,
        "total_tracks": total_tracks,
        "total_events": total_events,
        "event_counts": event_counts,
        "tracker_stats": tracker.stats,
        "fsm_stats": fsm.stats,
    }

    logger.info(f"FSM test complete: {frames_processed} frames")
    logger.info(f"  Detections: {total_detections}")
    logger.info(f"  Tracks: {total_tracks}")
    logger.info(f"  Events: {total_events}")
    logger.info(f"  Event breakdown: {event_counts}")

    return results


def convert_to_serializable(obj):
    """Convert to JSON-serializable types."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Test FSM and tracking on a video file"
    )
    parser.add_argument("video_path", type=Path, help="Path to video file")
    parser.add_argument(
        "--start-time", "-s",
        type=str,
        required=True,
        help="Start time (MM:SS or HH:MM:SS)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON file"
    )
    parser.add_argument(
        "--model-path", "-m",
        type=Path,
        default=None,
        help="Path to YOLO model"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.3,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--compare", "-C",
        type=Path,
        default=None,
        help="Compare with ground truth JSON from analyze_motion_and_objects.py"
    )

    args = parser.parse_args()

    if not args.video_path.exists():
        logger.error(f"Video file not found: {args.video_path}")
        sys.exit(1)

    start_time_sec = parse_time_string(args.start_time)

    output_path = args.output
    if output_path is None:
        output_path = args.video_path.with_suffix("").with_suffix("_fsm_test.json")

    # Run FSM test
    results = run_fsm_test(
        args.video_path,
        start_time_sec,
        model_path=args.model_path,
        confidence_threshold=args.confidence,
    )

    # Save results
    logger.info(f"Saving results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    # Compare with ground truth if provided
    if args.compare and args.compare.exists():
        logger.info(f"Comparing with ground truth: {args.compare}")
        with open(args.compare) as f:
            ground_truth = json.load(f)

        gt_frames = {fd["frame_number"]: fd for fd in ground_truth.get("frame_detections", [])}

        # Compare detection counts
        matched_frames = 0
        detection_diff = 0

        for frame_result in results.frames:
            fn = frame_result.frame_number
            if fn in gt_frames:
                matched_frames += 1
                gt_dets = len(gt_frames[fn]["detections"])
                our_dets = len(frame_result.detections)
                detection_diff += abs(gt_dets - our_dets)

        logger.info(f"Comparison summary:")
        logger.info(f"  Matched frames: {matched_frames}")
        logger.info(f"  Avg detection difference: {detection_diff / max(matched_frames, 1):.2f}")

    # Print event summary
    print("\n=== FSM EVENTS ===")
    for event in results.all_events:
        if event["event_type"] in ("arrival", "departure"):
            print(
                f"{event['timestamp_str']}: {event['event_type'].upper()} - "
                f"{event['class_name']} (track {event['track_id']})"
            )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
