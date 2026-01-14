#!/usr/bin/env python3
"""Test the full detection pipeline with FSM on video files.

Simulates the live detection flow:
1. YOLO detection at configurable threshold
2. Class whitelist filtering
3. Simple tracker (by class + position)
4. FSM for arrival/departure events

Usage:
    python test_full_pipeline_fsm.py video.mp4 --confidence 0.25
    python test_full_pipeline_fsm.py video.mp4 --start 300 --duration 60
"""

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.object_fsm import (
    ObjectStateMachine,
    ObjectState,
    EventType,
    ObjectEvent,
)
from app.services.ml.tracker import ByteTracker, Detection as TrackerDetection


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
class TrackedObject:
    """Tracked object for FSM input."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x: float
    y: float
    width: float
    height: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0


class SimpleYOLODetector:
    """YOLO detector using ONNX Runtime."""

    def __init__(self, model_path: str, confidence: float = 0.25):
        import onnxruntime as ort

        self.confidence = confidence
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on frame."""
        # Preprocess
        resized = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        # Run inference
        outputs = self.session.run(None, {self.input_name: blob})
        output = outputs[0]

        # Parse output - shape is (1, 84, 8400) for YOLO
        if len(output.shape) == 3:
            output = output[0].T  # Now (8400, 84)

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

        # NMS - keep top detections per class
        return self._nms(detections)

    def _nms(self, detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
        """NMS - filter overlapping detections per class."""
        by_class = defaultdict(list)
        for d in detections:
            by_class[d.class_name].append(d)

        result = []
        for class_name, dets in by_class.items():
            # Sort by confidence descending
            sorted_dets = sorted(dets, key=lambda x: -x.confidence)
            keep = []

            while sorted_dets:
                best = sorted_dets.pop(0)
                keep.append(best)

                # Remove overlapping detections
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
        # Convert center+size to corners
        x1_1 = d1.x - d1.width / 2
        y1_1 = d1.y - d1.height / 2
        x2_1 = d1.x + d1.width / 2
        y2_1 = d1.y + d1.height / 2

        x1_2 = d2.x - d2.width / 2
        y1_2 = d2.y - d2.height / 2
        x2_2 = d2.x + d2.width / 2
        y2_2 = d2.y + d2.height / 2

        # Intersection
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


class SimpleTracker:
    """Simple tracker that associates detections by class and position."""

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks: dict[int, TrackedObject] = {}
        self.last_positions: dict[int, tuple[float, float]] = {}

    def update(self, detections: list[Detection], dt: float = 1.0) -> list[TrackedObject]:
        """Update tracks with new detections."""
        # Match detections to existing tracks
        matched = set()
        updated_tracks = []

        for det in detections:
            best_track_id = None
            best_iou = 0.0

            for track_id, track in self.tracks.items():
                if track.class_name != det.class_name:
                    continue
                if track_id in matched:
                    continue

                iou = self._calculate_iou(det, track)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                track = self.tracks[best_track_id]
                old_x, old_y = self.last_positions.get(best_track_id, (det.x, det.y))

                # Calculate velocity
                velocity_x = (det.x - old_x) / dt if dt > 0 else 0.0
                velocity_y = (det.y - old_y) / dt if dt > 0 else 0.0

                track.x = det.x
                track.y = det.y
                track.width = det.width
                track.height = det.height
                track.confidence = det.confidence
                track.velocity_x = velocity_x
                track.velocity_y = velocity_y

                self.last_positions[best_track_id] = (det.x, det.y)
                matched.add(best_track_id)
                updated_tracks.append(track)
            else:
                # Create new track
                track = TrackedObject(
                    track_id=self.next_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    x=det.x,
                    y=det.y,
                    width=det.width,
                    height=det.height,
                )
                self.tracks[self.next_id] = track
                self.last_positions[self.next_id] = (det.x, det.y)
                updated_tracks.append(track)
                self.next_id += 1

        # Remove unmatched tracks (will be handled by FSM lost_seconds)
        # Keep them for now, FSM handles departure

        return updated_tracks

    def _calculate_iou(self, det: Detection, track: TrackedObject) -> float:
        """Calculate IoU between detection and track."""
        # Convert center+size to corners
        d_x1 = det.x - det.width / 2
        d_y1 = det.y - det.height / 2
        d_x2 = det.x + det.width / 2
        d_y2 = det.y + det.height / 2

        t_x1 = track.x - track.width / 2
        t_y1 = track.y - track.height / 2
        t_x2 = track.x + track.width / 2
        t_y2 = track.y + track.height / 2

        # Intersection
        inter_x1 = max(d_x1, t_x1)
        inter_y1 = max(d_y1, t_y1)
        inter_x2 = min(d_x2, t_x2)
        inter_y2 = min(d_y2, t_y2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        det_area = det.width * det.height
        track_area = track.width * track.height
        union_area = det_area + track_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


def process_video(
    video_path: str,
    yolo_model: str,
    confidence: float = 0.25,
    start_sec: float = 0,
    duration_sec: float = None,
    detection_fps: float = 1.0,
) -> dict:
    """Process video through full pipeline."""

    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return {"error": "Cannot open video"}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps

    if duration_sec is None:
        duration_sec = video_duration - start_sec

    end_sec = min(start_sec + duration_sec, video_duration)

    print(f"Video: {video_fps:.1f} FPS, {video_duration:.1f}s total")
    print(f"Processing: {start_sec:.1f}s to {end_sec:.1f}s at {detection_fps} detection FPS")
    print(f"Confidence threshold: {confidence}")
    print(f"Class whitelist: {sorted(CLASS_WHITELIST)}")
    print()

    # Initialize components
    print("Loading YOLO model...")
    detector = SimpleYOLODetector(yolo_model, confidence=confidence)

    # Use production ByteTracker instead of simple tracker
    tracker = ByteTracker(
        track_high_thresh=0.5,    # High confidence threshold
        track_low_thresh=0.1,     # Low confidence threshold
        match_thresh=0.8,         # IoU match threshold
        track_buffer=30,          # Keep lost tracks for 30 frames
        min_hits=3,               # Need 3 detections to confirm
        min_displacement=0.0,     # No min displacement for now
    )

    fsm = ObjectStateMachine(
        validation_frames=3,      # Need 3 detections to confirm (matches tracker)
        velocity_threshold=0.002,  # Movement threshold
        stationary_seconds=10.0,   # 10s to become stationary
        parked_seconds=60.0,       # 1 min to become parked (shorter for testing)
        lost_seconds=5.0,          # 5s without detection = departed
        fps=detection_fps,
    )

    # Process frames
    frame_interval = max(1, int(video_fps / detection_fps))
    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)

    print(f"Frame range: {start_frame} to {end_frame}, interval: {frame_interval}")
    print("=" * 70)

    all_events = []
    detection_counts = defaultdict(int)
    filtered_counts = defaultdict(int)
    frames_processed = 0

    frame_num = start_frame
    while frame_num < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_num / video_fps

        # 1. Run YOLO detection
        raw_detections = detector.detect(frame)

        # Count raw detections
        for d in raw_detections:
            detection_counts[d.class_name] += 1

        # 2. Apply class whitelist
        filtered_detections = [d for d in raw_detections if d.class_name in CLASS_WHITELIST]

        for d in filtered_detections:
            filtered_counts[d.class_name] += 1

        # 3. Convert to tracker format and update ByteTracker
        tracker_dets = [
            TrackerDetection(
                x=d.x - d.width/2,  # Convert center to top-left
                y=d.y - d.height/2,
                width=d.width,
                height=d.height,
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
            )
            for d in filtered_detections
        ]
        tracks = tracker.update(tracker_dets)

        # 4. Update FSM
        events = fsm.update(tracks)

        # Log events
        for event in events:
            if event.event_type in (EventType.ARRIVAL, EventType.DEPARTURE):
                event_info = {
                    "frame": frame_num,
                    "timestamp": timestamp,
                    "type": event.event_type.value,
                    "class": event.class_name,
                    "track_id": event.track_id,
                    "confidence": event.confidence,
                }
                all_events.append(event_info)

                print(f"[{timestamp:6.1f}s] {event.event_type.value.upper():10s} "
                      f"{event.class_name} (track {event.track_id}, conf={event.confidence:.2f})")

        frames_processed += 1
        frame_num += frame_interval

        if frames_processed % 30 == 0:
            print(f"  ... processed {frames_processed} frames ({timestamp:.1f}s)")

    # Final FSM update to catch departures
    print("\nFinalizing (checking for departures)...")
    for _ in range(10):
        time.sleep(0.5)
        events = fsm.update([])
        for event in events:
            if event.event_type == EventType.DEPARTURE:
                event_info = {
                    "frame": frame_num,
                    "timestamp": end_sec,
                    "type": event.event_type.value,
                    "class": event.class_name,
                    "track_id": event.track_id,
                    "confidence": event.confidence,
                }
                all_events.append(event_info)
                print(f"[{end_sec:6.1f}s] DEPARTURE   {event.class_name} (track {event.track_id})")

    cap.release()

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nFrames processed: {frames_processed}")
    print(f"Duration analyzed: {end_sec - start_sec:.1f}s")

    print(f"\nRaw detections (before whitelist):")
    for cls, count in sorted(detection_counts.items(), key=lambda x: -x[1])[:15]:
        in_whitelist = "✓" if cls in CLASS_WHITELIST else "✗"
        print(f"  {in_whitelist} {cls}: {count}")

    print(f"\nFiltered detections (after whitelist):")
    for cls, count in sorted(filtered_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")

    arrivals = [e for e in all_events if e["type"] == "arrival"]
    departures = [e for e in all_events if e["type"] == "departure"]

    print(f"\nFSM Events:")
    print(f"  Arrivals: {len(arrivals)}")
    print(f"  Departures: {len(departures)}")

    if arrivals:
        print(f"\n  Arrival details:")
        for e in arrivals:
            print(f"    [{e['timestamp']:.1f}s] {e['class']} (track {e['track_id']})")

    if departures:
        print(f"\n  Departure details:")
        for e in departures:
            print(f"    [{e['timestamp']:.1f}s] {e['class']} (track {e['track_id']})")

    # FSM stats
    print(f"\nFSM final state:")
    stats = fsm.stats
    print(f"  Active objects: {stats['total_objects']}")
    print(f"  Recently departed: {stats['departed_recently']}")
    print(f"  By state: {stats['by_state']}")

    return {
        "frames_processed": frames_processed,
        "raw_detection_counts": dict(detection_counts),
        "filtered_detection_counts": dict(filtered_counts),
        "events": all_events,
        "arrivals": len(arrivals),
        "departures": len(departures),
    }


def main():
    parser = argparse.ArgumentParser(description="Test full detection pipeline with FSM")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds")
    parser.add_argument("--fps", type=float, default=1.0, help="Detection FPS")
    parser.add_argument("--yolo", default="/opt3/ronin/ml_models/yolo11l_dynamic.onnx", help="YOLO model path")
    args = parser.parse_args()

    results = process_video(
        args.video,
        args.yolo,
        confidence=args.confidence,
        start_sec=args.start,
        duration_sec=args.duration,
        detection_fps=args.fps,
    )

    print("\n" + "=" * 70)
    if results.get("arrivals", 0) == 0 and results.get("departures", 0) == 0:
        print("RESULT: No events generated (good if nothing happened in video)")
    else:
        print(f"RESULT: {results.get('arrivals', 0)} arrivals, {results.get('departures', 0)} departures")


if __name__ == "__main__":
    main()
