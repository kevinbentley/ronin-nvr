#!/usr/bin/env python3
"""Debug - check if motion gate is filtering people."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.gpu_orchestrator import GPUOrchestrator, GPUPipelineConfig


def main():
    video_path = "/opt3/ronin/storage/Hangar_East/2025-12-30/00-21-28.mp4"

    # Lower motion threshold to catch everything
    config = GPUPipelineConfig(
        model_path="/opt3/ronin/ml_models/yolov8n_dynamic.onnx",
        detection_confidence=0.65,
        class_thresholds={"person": 0.45},
        motion_min_percent=0.05,  # Very low - should catch everything
        track_min_hits=1,
        track_min_displacement=0.0,
    )

    orchestrator = GPUOrchestrator(device_ids=[0], config=config)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video FPS: {fps}")

    # Sample at 3 FPS like the test
    frame_skip = int(fps / 3)
    frame_idx = 0
    motion_count = 0
    person_count = 0
    frames_sampled = 0

    print("Processing first 300 frames (same as test)...")
    print("With motion_min_percent=0.05 (very low)")
    print()

    while frames_sampled < 300:
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

        if result.motion_detected:
            motion_count += 1

        for det in result.detections:
            if det.class_name == "person":
                person_count += 1
                print(f"  Frame {frame_idx} ({timestamp:.1f}s): person @ {det.confidence:.2f}, motion={result.motion_percent:.1f}%")

        frame_idx += 1

    cap.release()

    print()
    print(f"Frames sampled: {frames_sampled}")
    print(f"Motion detected in: {motion_count} frames ({motion_count/frames_sampled*100:.1f}%)")
    print(f"Person detections: {person_count}")


if __name__ == "__main__":
    main()
