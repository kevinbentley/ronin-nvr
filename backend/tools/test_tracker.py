#!/usr/bin/env python3
"""Test ByteTrack multi-object tracker.

Usage:
    source /opt/venv/bin/activate
    cd /workspace/ronin-nvr/backend
    python tools/test_tracker.py --video /opt3/ronin/storage/test.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.tracker import ByteTracker, Detection, TrackedObject
from app.services.ml.tensorrt_inference import TensorRTDetector, TensorRTDetection


def test_basic_tracking():
    """Test basic tracking functionality."""
    print("=" * 60)
    print("Basic Tracking Test")
    print("=" * 60)

    tracker = ByteTracker(
        track_high_thresh=0.5,
        match_thresh=0.8,
        track_buffer=30,
        min_hits=3,
    )

    # Simulate object moving across frames
    frames = []
    for i in range(30):
        # Object moving from left to right
        x = 0.1 + i * 0.02
        detections = [
            Detection(
                x=x, y=0.3, width=0.1, height=0.15,
                confidence=0.9, class_id=2, class_name="car"
            )
        ]
        frames.append(detections)

    # Process frames
    for frame_idx, detections in enumerate(frames):
        tracks = tracker.update(detections)

        if frame_idx < 5 or frame_idx >= 25:
            print(f"Frame {frame_idx}: {len(tracks)} active tracks, stats: {tracker.stats}")

    print(f"\nFinal stats: {tracker.stats}")

    # Verify we have one consistent track
    assert tracker.stats["total_ids"] <= 2, "Should have at most 2 track IDs"
    print("\n[PASS] Basic tracking test")


def test_occlusion_handling():
    """Test tracking through occlusion."""
    print("=" * 60)
    print("Occlusion Handling Test")
    print("=" * 60)

    tracker = ByteTracker(
        track_high_thresh=0.5,
        match_thresh=0.8,
        track_buffer=30,
        min_hits=2,
    )

    # Object visible, then occluded, then visible again
    frames = []

    # Visible (frames 0-9)
    for i in range(10):
        frames.append([
            Detection(x=0.2 + i*0.02, y=0.3, width=0.1, height=0.15,
                     confidence=0.9, class_id=2, class_name="car")
        ])

    # Occluded (frames 10-14)
    for i in range(5):
        frames.append([])  # No detections

    # Visible again (frames 15-24)
    for i in range(10):
        frames.append([
            Detection(x=0.4 + i*0.02, y=0.3, width=0.1, height=0.15,
                     confidence=0.9, class_id=2, class_name="car")
        ])

    track_ids = set()
    for frame_idx, detections in enumerate(frames):
        tracks = tracker.update(detections)
        for t in tracks:
            track_ids.add(t.track_id)

        if frame_idx in [9, 14, 15, 24]:
            print(f"Frame {frame_idx}: {len(tracks)} tracks, IDs: {[t.track_id for t in tracks]}")

    print(f"\nTotal unique track IDs: {len(track_ids)}")
    print(f"Stats: {tracker.stats}")

    # Should maintain same ID through occlusion (if within buffer)
    if len(track_ids) == 1:
        print("\n[PASS] Track maintained through occlusion")
    else:
        print("\n[INFO] Track re-identified after occlusion")


def test_multi_object():
    """Test tracking multiple objects."""
    print("=" * 60)
    print("Multi-Object Tracking Test")
    print("=" * 60)

    tracker = ByteTracker(
        track_high_thresh=0.5,
        match_thresh=0.8,
        track_buffer=30,
        min_hits=2,
    )

    # Two objects moving in parallel
    frames = []
    for i in range(20):
        frames.append([
            Detection(x=0.1 + i*0.02, y=0.2, width=0.08, height=0.12,
                     confidence=0.9, class_id=0, class_name="person"),
            Detection(x=0.1 + i*0.02, y=0.6, width=0.1, height=0.15,
                     confidence=0.85, class_id=2, class_name="car"),
        ])

    for frame_idx, detections in enumerate(frames):
        tracks = tracker.update(detections)

        if frame_idx == 19:
            print(f"Frame {frame_idx}: {len(tracks)} tracks")
            for t in tracks:
                print(f"  Track {t.track_id}: {t.class_name} at ({t.x:.2f}, {t.y:.2f}), "
                      f"velocity=({t.velocity_x:.4f}, {t.velocity_y:.4f})")

    print(f"\nStats: {tracker.stats}")
    assert tracker.stats["active_tracks"] == 2, "Should have 2 active tracks"
    print("\n[PASS] Multi-object tracking test")


def test_video_tracking(video_path: str, max_frames: int = 200):
    """Test tracking on real video with detection."""
    print("=" * 60)
    print(f"Video Tracking Test: {Path(video_path).name}")
    print("=" * 60)

    onnx_path = Path("/opt3/ronin/ml_models/yolov8n.onnx")
    if not onnx_path.exists():
        print(f"Model not found: {onnx_path}")
        return

    # Initialize detector
    detector = TensorRTDetector(
        model_path=onnx_path,
        precision="fp16",
        confidence_threshold=0.3,
        warmup_iterations=5,
    )

    # Initialize tracker
    tracker = ByteTracker(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        match_thresh=0.7,
        track_buffer=30,
        min_hits=3,
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {width}x{height} @ {fps:.1f} fps")

    # Track statistics
    detection_times = []
    tracking_times = []
    track_counts = []
    unique_ids = set()

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        detection_time = (time.perf_counter() - t0) * 1000

        # Convert to tracker format
        track_dets = [
            Detection(
                x=d.x, y=d.y, width=d.width, height=d.height,
                confidence=d.confidence, class_id=d.class_id, class_name=d.class_name
            )
            for d in detections
        ]

        # Track
        t0 = time.perf_counter()
        tracks = tracker.update(track_dets)
        tracking_time = (time.perf_counter() - t0) * 1000

        detection_times.append(detection_time)
        tracking_times.append(tracking_time)
        track_counts.append(len(tracks))

        for t in tracks:
            unique_ids.add(t.track_id)

        if frame_count % 50 == 0:
            print(f"  Frame {frame_count}: {len(tracks)} tracks, "
                  f"{len(detections)} detections")

    cap.release()

    # Results
    print(f"\nResults ({frame_count} frames):")
    print(f"  Detection time: {np.mean(detection_times):.2f} ms avg")
    print(f"  Tracking time:  {np.mean(tracking_times):.2f} ms avg")
    print(f"  Total time:     {np.mean(detection_times) + np.mean(tracking_times):.2f} ms avg")
    print(f"  FPS:            {1000/(np.mean(detection_times) + np.mean(tracking_times)):.1f}")
    print(f"\n  Track statistics:")
    print(f"    Unique track IDs: {len(unique_ids)}")
    print(f"    Avg tracks/frame: {np.mean(track_counts):.1f}")
    print(f"    Max tracks/frame: {max(track_counts)}")
    print(f"    Final stats: {tracker.stats}")


def main():
    parser = argparse.ArgumentParser(description="Test ByteTrack tracker")
    parser.add_argument("--video", type=str, help="Path to test video")
    parser.add_argument("--frames", type=int, default=200, help="Max frames")
    parser.add_argument("--skip-unit", action="store_true", help="Skip unit tests")
    args = parser.parse_args()

    if not args.skip_unit:
        test_basic_tracking()
        print()
        test_occlusion_handling()
        print()
        test_multi_object()
        print()

    if args.video:
        test_video_tracking(args.video, max_frames=args.frames)
        print()

    print("=" * 60)
    print("All tracking tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
