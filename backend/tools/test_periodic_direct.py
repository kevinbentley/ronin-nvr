#!/usr/bin/env python3
"""Direct comparison - same frames with and without periodic detection."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.gpu_orchestrator import GPUOrchestrator, GPUPipelineConfig


def test_config(name: str, periodic_interval: int, video_path: str) -> dict:
    """Test a specific configuration."""
    config = GPUPipelineConfig(
        model_path="/opt3/ronin/ml_models/yolov8n_dynamic.onnx",
        detection_confidence=0.65,
        class_thresholds={"person": 0.45, "dog": 0.45, "cat": 0.45},
        motion_min_percent=0.3,
        track_min_hits=1,
        track_min_displacement=0.0,
        periodic_detection_interval=periodic_interval,
    )

    orchestrator = GPUOrchestrator(device_ids=[0], config=config)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Sample at 2 FPS (matching debug_find_people.py)
    frame_skip = int(fps / 2)
    frame_idx = 0
    person_count = 0
    truck_count = 0
    yolo_runs = 0
    frames_sampled = 0

    # Process first 50 seconds (100 FPS * 50s = 5000 frames, ~100 samples at 2 FPS)
    while frames_sampled < 100:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        frames_sampled += 1
        timestamp = frame_idx / fps

        # Scale to 720p
        h, w = frame.shape[:2]
        if h > 720:
            scale = 720 / h
            frame = cv2.resize(frame, (int(w * scale), 720))

        result = orchestrator.process(
            camera_id=1,
            frame=frame,
            timestamp=timestamp,
        )

        if result.detections:
            yolo_runs += 1

        for det in result.detections:
            if det.class_name == "person":
                person_count += 1
            elif det.class_name == "truck":
                truck_count += 1

        frame_idx += 1

    cap.release()

    return {
        "name": name,
        "frames": frames_sampled,
        "yolo_runs": yolo_runs,
        "persons": person_count,
        "trucks": truck_count,
    }


def main():
    video_path = "/opt3/ronin/storage/Hangar_East/2025-12-30/00-21-28.mp4"

    print("Comparing motion-only vs periodic detection...")
    print("Testing first 50 seconds of video at 2 FPS sampling")
    print()

    # Test 1: Motion-only (periodic=0)
    print("Testing motion-only (periodic=0)...")
    result1 = test_config("Motion only", 0, video_path)

    # Test 2: Periodic every 5 frames
    print("Testing periodic every 5 frames...")
    result2 = test_config("Periodic=5", 5, video_path)

    # Test 3: Periodic every 3 frames (more frequent)
    print("Testing periodic every 3 frames...")
    result3 = test_config("Periodic=3", 3, video_path)

    print()
    print("=" * 60)
    print(f"{'Config':<20} {'Frames':>8} {'YOLO runs':>10} {'Persons':>8} {'Trucks':>8}")
    print("=" * 60)

    for r in [result1, result2, result3]:
        print(f"{r['name']:<20} {r['frames']:>8} {r['yolo_runs']:>10} {r['persons']:>8} {r['trucks']:>8}")

    print()

    if result2['persons'] > result1['persons'] or result3['persons'] > result1['persons']:
        print("SUCCESS: Periodic detection found more people than motion-only!")
    else:
        print("Note: Periodic detection didn't find more people in this video segment.")
        print("This video may have insufficient person movement to trigger motion,")
        print("or persons may not be present in sampled frames.")


if __name__ == "__main__":
    main()
