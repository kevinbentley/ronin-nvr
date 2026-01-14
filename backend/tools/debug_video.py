#!/usr/bin/env python3
"""Debug detection on a specific video to see all detections."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.gpu_orchestrator import GPUOrchestrator, GPUPipelineConfig


def main():
    video_path = "/opt3/ronin/storage/Hangar_East/2025-12-30/00-21-28.mp4"

    # Test with LOWER confidence to see what we're missing
    print("Testing with confidence=0.65 (current setting)...")
    test_video(video_path, confidence=0.65)

    print("\n" + "="*60)
    print("Testing with confidence=0.4 (lower threshold)...")
    test_video(video_path, confidence=0.4)

    print("\n" + "="*60)
    print("Testing with confidence=0.25 (very low threshold)...")
    test_video(video_path, confidence=0.25)


def test_video(video_path: str, confidence: float):
    config = GPUPipelineConfig(
        model_path="/opt3/ronin/ml_models/yolov8n_dynamic.onnx",
        detection_confidence=confidence,
        track_min_hits=1,  # Show all detections, not just confirmed
        track_min_displacement=0.0,  # Disable displacement filter
    )

    orchestrator = GPUOrchestrator(device_ids=[0], config=config)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Sample at 1 FPS
    frame_skip = int(fps)
    frame_idx = 0
    detections_by_class = {}

    while frame_idx < 3000:  # ~100 seconds
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

        # Count all raw detections (before tracking filter)
        for det in result.detections:
            cls = det.class_name
            conf = det.confidence
            detections_by_class.setdefault(cls, []).append(conf)

        frame_idx += 1

    cap.release()

    print(f"  Confidence threshold: {confidence}")
    print(f"  Detections by class:")
    for cls, confs in sorted(detections_by_class.items()):
        avg_conf = sum(confs) / len(confs)
        print(f"    - {cls}: {len(confs)} detections (avg conf: {avg_conf:.2f}, max: {max(confs):.2f})")

    if not detections_by_class:
        print("    (no detections)")


if __name__ == "__main__":
    main()
