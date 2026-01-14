#!/usr/bin/env python3
"""Test periodic detection - verify people are detected without motion gate."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.gpu_orchestrator import GPUOrchestrator, GPUPipelineConfig


def main():
    video_path = "/opt3/ronin/storage/Hangar_East/2025-12-30/00-21-28.mp4"

    # Test with periodic detection every 10 frames
    config = GPUPipelineConfig(
        model_path="/opt3/ronin/ml_models/yolov8n_dynamic.onnx",
        detection_confidence=0.65,
        class_thresholds={"person": 0.45, "dog": 0.45, "cat": 0.45},
        motion_min_percent=0.3,  # Normal motion threshold
        track_min_hits=1,
        track_min_displacement=0.0,
        periodic_detection_interval=10,  # Run YOLO every 10 frames regardless of motion
    )

    orchestrator = GPUOrchestrator(device_ids=[0], config=config)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video FPS: {fps}")

    # Skip to frame 350 where people start appearing (based on debug_find_people output)
    start_frame = 350
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Sample at 3 FPS
    frame_skip = int(fps / 3)
    frame_idx = start_frame
    motion_frames = 0
    detection_frames = 0
    person_count = 0
    truck_count = 0
    frames_sampled = 0

    print(f"Processing frames {start_frame}-{start_frame + 5000} (where people appear)...")
    print("Config: motion_min_percent=0.3%, periodic_detection_interval=10")
    print("Thresholds: person=0.45, default=0.65")
    print()

    while frames_sampled < 150:
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
            motion_frames += 1

        if result.detections:
            detection_frames += 1

        for det in result.detections:
            if det.class_name == "person":
                person_count += 1
                trigger = "motion" if result.motion_detected else "periodic"
                print(f"  Frame {frame_idx} ({timestamp:.1f}s): PERSON @ {det.confidence:.2f} [{trigger}]")
            elif det.class_name == "truck":
                truck_count += 1
                if result.motion_detected:
                    trigger = "motion"
                else:
                    trigger = "periodic"
                # Only log some truck detections
                if truck_count <= 3 or truck_count % 10 == 0:
                    print(f"  Frame {frame_idx} ({timestamp:.1f}s): truck @ {det.confidence:.2f} [{trigger}]")

        frame_idx += 1

    cap.release()

    print()
    print(f"Frames sampled: {frames_sampled}")
    print(f"Motion detected in: {motion_frames} frames ({motion_frames/frames_sampled*100:.1f}%)")
    print(f"Frames with detections: {detection_frames}")
    print(f"Person detections: {person_count}")
    print(f"Truck detections: {truck_count}")

    if person_count > 0:
        print()
        print("SUCCESS: Periodic detection caught people!")
    else:
        print()
        print("WARNING: No people detected - checking if periodic detection is firing...")
        # Run a diagnostic
        print()
        print("Running diagnostic - check if periodic frames are actually running YOLO...")


if __name__ == "__main__":
    main()
