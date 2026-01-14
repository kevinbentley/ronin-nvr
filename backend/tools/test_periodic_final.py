#!/usr/bin/env python3
"""Final test - verify full pipeline with periodic detection on multiple videos."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.gpu_orchestrator import GPUOrchestrator, GPUPipelineConfig


def test_video(video_path: str, config: GPUPipelineConfig, max_seconds: int = 60) -> dict:
    """Test a video with given configuration."""
    orchestrator = GPUOrchestrator(device_ids=[0], config=config)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Failed to open {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample at 3 FPS (production setting)
    frame_skip = int(fps / 3)
    max_samples = int(max_seconds * 3)

    frame_idx = 0
    results = {
        "video": Path(video_path).name,
        "fps": fps,
        "frames_sampled": 0,
        "motion_frames": 0,
        "detection_frames": 0,
        "persons": 0,
        "trucks": 0,
        "cars": 0,
        "other": 0,
    }

    while results["frames_sampled"] < max_samples:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        results["frames_sampled"] += 1
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

        if result.motion_detected:
            results["motion_frames"] += 1

        if result.detections:
            results["detection_frames"] += 1

        for det in result.detections:
            if det.class_name == "person":
                results["persons"] += 1
            elif det.class_name == "truck":
                results["trucks"] += 1
            elif det.class_name == "car":
                results["cars"] += 1
            else:
                results["other"] += 1

        frame_idx += 1

    cap.release()
    return results


def main():
    # Test videos - daytime, nighttime, with/without people
    videos = [
        # Nighttime with people and trucks
        "/opt3/ronin/storage/Hangar_East/2025-12-30/00-21-28.mp4",
        # Daytime
        "/opt3/ronin/storage/Hangar_East/2025-12-20/12-51-10.mp4",
        # Another nighttime
        "/opt3/ronin/storage/Hangar_East/2025-12-29/00-06-12.mp4",
    ]

    # Production-like config with periodic detection
    config = GPUPipelineConfig(
        model_path="/opt3/ronin/ml_models/yolov8n_dynamic.onnx",
        detection_confidence=0.65,
        class_thresholds={"person": 0.45, "dog": 0.45, "cat": 0.45},
        motion_min_percent=0.3,
        track_min_hits=1,
        track_min_displacement=0.0,
        periodic_detection_interval=30,  # Every 30 frames (~10 seconds at 3 FPS)
    )

    print("=" * 70)
    print("NextGen Pipeline - Production Config with Periodic Detection")
    print("=" * 70)
    print()
    print("Config:")
    print(f"  Detection confidence: {config.detection_confidence}")
    print(f"  Person threshold: {config.class_thresholds.get('person', config.detection_confidence)}")
    print(f"  Motion min percent: {config.motion_min_percent}%")
    print(f"  Periodic interval: {config.periodic_detection_interval} frames")
    print()
    print("Testing 60 seconds from each video at 3 FPS...")
    print()

    all_results = []
    for video in videos:
        if not Path(video).exists():
            print(f"Skipping {video} (not found)")
            continue

        print(f"Testing: {Path(video).name}...")
        result = test_video(video, config, max_seconds=60)
        all_results.append(result)

    print()
    print("=" * 70)
    print(f"{'Video':<30} {'Samples':>8} {'Motion':>8} {'Person':>8} {'Truck':>8}")
    print("=" * 70)

    total_persons = 0
    total_trucks = 0

    for r in all_results:
        if "error" in r:
            print(f"{r.get('video', 'unknown'):<30} ERROR: {r['error']}")
            continue

        motion_pct = r['motion_frames'] / r['frames_sampled'] * 100 if r['frames_sampled'] > 0 else 0
        print(f"{r['video']:<30} {r['frames_sampled']:>8} {motion_pct:>7.1f}% {r['persons']:>8} {r['trucks']:>8}")
        total_persons += r['persons']
        total_trucks += r['trucks']

    print("=" * 70)
    print(f"{'TOTAL':<30} {'':<8} {'':<8} {total_persons:>8} {total_trucks:>8}")
    print()

    if total_persons > 0:
        print("SUCCESS: People detected with periodic detection enabled!")
    else:
        print("Note: No people detected in test videos.")

    print()
    print("False positive check:")
    print("  - Review 'other' detections count for unexpected classes")
    print("  - Check for consistent person/truck counts across runs")


if __name__ == "__main__":
    main()
