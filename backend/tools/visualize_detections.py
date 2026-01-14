#!/usr/bin/env python3
"""Visualize detections - save annotated frames and create video montage."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.gpu_orchestrator import GPUOrchestrator, GPUPipelineConfig


def get_class_color(class_name: str) -> tuple:
    """Get BGR color for class."""
    colors = {
        "person": (0, 255, 0),    # Green
        "car": (255, 0, 0),       # Blue
        "truck": (255, 128, 0),   # Orange-blue
        "bus": (255, 0, 128),     # Purple
        "motorcycle": (0, 165, 255),  # Orange
        "dog": (0, 255, 255),     # Yellow
        "cat": (0, 255, 255),     # Yellow
    }
    return colors.get(class_name, (255, 255, 255))


def draw_detections(frame: np.ndarray, detections: list, tracks: list = None) -> np.ndarray:
    """Draw detection boxes on frame."""
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Draw raw detections (thin boxes)
    for det in detections:
        x1 = int(det.x * w)
        y1 = int(det.y * h)
        x2 = int((det.x + det.width) * w)
        y2 = int((det.y + det.height) * h)

        color = get_class_color(det.class_name)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

        label = f"{det.class_name} {det.confidence:.0%}"
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw tracks (thick boxes with ID)
    if tracks:
        for track in tracks:
            x1 = int(track.x * w)
            y1 = int(track.y * h)
            x2 = int((track.x + track.width) * w)
            y2 = int((track.y + track.height) * h)

            color = get_class_color(track.class_name)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"#{track.track_id} {track.class_name} {track.confidence:.0%}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return annotated


def main():
    video_path = "/opt3/ronin/storage/Hangar_East/2025-12-30/00-21-28.mp4"
    output_dir = Path("/workspace/ronin-nvr/backend/tools/detection_output")
    output_dir.mkdir(exist_ok=True)

    # Production config with periodic detection
    config = GPUPipelineConfig(
        model_path="/opt3/ronin/ml_models/yolov8n_dynamic.onnx",
        detection_confidence=0.65,
        class_thresholds={"person": 0.45, "dog": 0.45, "cat": 0.45},
        motion_min_percent=0.3,
        track_min_hits=1,
        track_min_displacement=0.0,
        periodic_detection_interval=10,  # More frequent for visualization
    )

    orchestrator = GPUOrchestrator(device_ids=[0], config=config)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video FPS: {fps}")

    # Sample at 3 FPS
    frame_skip = int(fps / 3)
    frame_idx = 0
    frames_sampled = 0
    frames_with_detections = []

    print(f"Processing video and saving frames with detections...")
    print(f"Output directory: {output_dir}")
    print()

    # Process first 90 seconds (where people appear based on earlier scan)
    max_samples = 270  # 90 seconds * 3 FPS

    while frames_sampled < max_samples:
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

        # Save frames with detections
        if result.detections:
            annotated = draw_detections(frame, result.detections, result.tracks)

            # Add info overlay
            info = f"Frame {frame_idx} | {timestamp:.1f}s | Motion: {result.motion_percent:.1f}%"
            cv2.putText(annotated, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Count detections by class
            det_counts = {}
            for det in result.detections:
                det_counts[det.class_name] = det_counts.get(det.class_name, 0) + 1

            det_str = " | ".join(f"{k}: {v}" for k, v in det_counts.items())
            cv2.putText(annotated, det_str, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Save frame
            filename = f"frame_{frame_idx:06d}_{timestamp:.1f}s.jpg"
            cv2.imwrite(str(output_dir / filename), annotated)
            frames_with_detections.append((frame_idx, timestamp, det_counts))

            # Print progress
            print(f"  Saved: {filename} - {det_str}")

        frame_idx += 1

    cap.release()

    print()
    print(f"=" * 60)
    print(f"Processed {frames_sampled} frames")
    print(f"Saved {len(frames_with_detections)} frames with detections")
    print(f"Output: {output_dir}")
    print()

    # Create summary montage if we have enough frames
    if len(frames_with_detections) >= 4:
        print("Creating montage of detection frames...")
        create_montage(output_dir, frames_with_detections[:16])

    # List person detections specifically
    person_frames = [(f, t, c) for f, t, c in frames_with_detections if 'person' in c]
    if person_frames:
        print()
        print(f"Frames with PERSON detections ({len(person_frames)}):")
        for frame_idx, timestamp, counts in person_frames[:10]:
            print(f"  Frame {frame_idx} ({timestamp:.1f}s): {counts.get('person', 0)} person(s)")
        if len(person_frames) > 10:
            print(f"  ... and {len(person_frames) - 10} more")


def create_montage(output_dir: Path, frame_info: list):
    """Create a 4x4 montage of detection frames."""
    images = []
    for frame_idx, timestamp, _ in frame_info:
        filename = f"frame_{frame_idx:06d}_{timestamp:.1f}s.jpg"
        img_path = output_dir / filename
        if img_path.exists():
            img = cv2.imread(str(img_path))
            # Resize for montage
            img = cv2.resize(img, (320, 180))
            images.append(img)

    if len(images) < 4:
        return

    # Pad to 16 images
    while len(images) < 16:
        images.append(np.zeros((180, 320, 3), dtype=np.uint8))

    # Create 4x4 grid
    rows = []
    for i in range(4):
        row = np.hstack(images[i*4:(i+1)*4])
        rows.append(row)
    montage = np.vstack(rows)

    montage_path = output_dir / "montage.jpg"
    cv2.imwrite(str(montage_path), montage)
    print(f"Montage saved: {montage_path}")


if __name__ == "__main__":
    main()
