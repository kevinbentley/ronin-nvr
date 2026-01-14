#!/usr/bin/env python3
"""Debug - find where people appear in the video."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.tensorrt_inference import TensorRTDetector


def main():
    video_path = "/opt3/ronin/storage/Hangar_East/2025-12-30/00-21-28.mp4"

    # Create detector with LOW threshold to find all possible person detections
    detector = TensorRTDetector(
        model_path="/opt3/ronin/ml_models/yolov8n_dynamic.onnx",
        confidence_threshold=0.25,  # Very low to catch all
        warmup_iterations=3,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video: {video_path}")
    print(f"FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.1f}s")
    print()

    # Sample every 0.5 seconds to find people
    frame_skip = int(fps / 2)  # 2 samples per second
    frame_idx = 0
    person_frames = []

    print("Scanning for people (threshold=0.25)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps

        # Scale to 720p
        h, w = frame.shape[:2]
        if h > 720:
            scale = 720 / h
            frame = cv2.resize(frame, (int(w * scale), 720))

        detections = detector.detect(frame)

        # Look for people
        for det in detections:
            if det.class_name == "person":
                person_frames.append({
                    "frame": frame_idx,
                    "time": timestamp,
                    "confidence": det.confidence,
                })
                print(f"  Frame {frame_idx} ({timestamp:.1f}s): person @ {det.confidence:.2f}")

        frame_idx += 1

    cap.release()

    print()
    if person_frames:
        print(f"Found {len(person_frames)} person detections")
        min_time = min(p["time"] for p in person_frames)
        max_time = max(p["time"] for p in person_frames)
        min_conf = min(p["confidence"] for p in person_frames)
        max_conf = max(p["confidence"] for p in person_frames)
        print(f"Time range: {min_time:.1f}s - {max_time:.1f}s")
        print(f"Confidence range: {min_conf:.2f} - {max_conf:.2f}")

        # Check if these would pass 0.45 threshold
        above_45 = [p for p in person_frames if p["confidence"] >= 0.45]
        print(f"Detections above 0.45: {len(above_45)}")
    else:
        print("No person detections found in the video!")


if __name__ == "__main__":
    main()
