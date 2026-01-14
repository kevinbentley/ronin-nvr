#!/usr/bin/env python3
"""Debug per-class thresholds on 00-21-28 video."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.gpu_orchestrator import GPUOrchestrator, GPUPipelineConfig


def main():
    video_path = "/opt3/ronin/storage/Hangar_East/2025-12-30/00-21-28.mp4"

    # Test with per-class thresholds
    print("Testing with per-class thresholds:")
    print("  - Default (vehicles): 0.65")
    print("  - Person: 0.45")
    print()

    config = GPUPipelineConfig(
        model_path="/opt3/ronin/ml_models/yolov8n_dynamic.onnx",
        detection_confidence=0.65,  # Default for vehicles
        class_thresholds={"person": 0.45, "dog": 0.45, "cat": 0.45},
        track_min_hits=1,  # Show all tracks
        track_min_displacement=0.0,
    )

    orchestrator = GPUOrchestrator(device_ids=[0], config=config)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Sample at 3 FPS (same as test)
    frame_skip = int(fps / 3)
    frame_idx = 0
    detections_by_class = {}
    frames_with_person = 0

    while frame_idx < 9000:  # ~300 frames at 3fps from 30fps video
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # Scale to 720p
        h, w = frame.shape[:2]
        if h > 720:
            scale = 720 / h
            frame = cv2.resize(frame, (int(w * scale), 720))

        result = orchestrator.process(
            camera_id=1,
            frame=frame,
            timestamp=frame_idx / fps,
        )

        # Count all raw detections
        for det in result.detections:
            cls = det.class_name
            conf = det.confidence
            detections_by_class.setdefault(cls, []).append(conf)
            if cls == "person":
                frames_with_person += 1
                print(f"  Frame {frame_idx}: person detected @ {conf:.2f}")

        frame_idx += 1

    cap.release()

    print(f"\nDetections by class:")
    for cls, confs in sorted(detections_by_class.items()):
        avg_conf = sum(confs) / len(confs)
        print(f"  - {cls}: {len(confs)} detections (avg conf: {avg_conf:.2f}, max: {max(confs):.2f}, min: {min(confs):.2f})")

    if not detections_by_class:
        print("  (no detections)")

    print(f"\nFrames with person detection: {frames_with_person}")


if __name__ == "__main__":
    main()
