#!/usr/bin/env python3
"""End-to-end test of the nextgen detection pipeline.

This script tests the complete pipeline:
1. Video decoding (NVDEC or CPU)
2. GPU motion detection (MOG2)
3. Object detection (TensorRT/ONNX)
4. Multi-object tracking (ByteTrack)
5. Finite State Machine (arrival/departure events)

Usage:
    source /opt/venv/bin/activate
    cd /workspace/ronin-nvr/backend
    python tools/test_full_pipeline.py --video /opt3/ronin/storage/test.mp4

    # Process multiple cameras (simulated)
    python tools/test_full_pipeline.py --video /opt3/ronin/storage/test.mp4 --cameras 4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.gpu_orchestrator import (
    GPUOrchestrator, GPUPipelineConfig, PipelineResult
)
from app.services.ml.object_fsm import EventType
from app.services.ml.nvdec_extractor import NVDECExtractor


def test_single_camera(video_path: str, max_frames: int = 300):
    """Test full pipeline on a single camera/video."""
    print("=" * 60)
    print("Single Camera Full Pipeline Test")
    print("=" * 60)

    # Check model exists
    model_path = Path("/opt3/ronin/ml_models/yolov8n.onnx")
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    # Configure pipeline
    config = GPUPipelineConfig(
        device_id=0,
        model_path=str(model_path),
        motion_min_percent=0.05,  # Lower threshold for testing
        detection_confidence=0.4,
        track_high_thresh=0.5,
        track_buffer=30,
        fsm_validation_frames=5,
        fsm_stationary_seconds=5.0,
        fsm_parked_seconds=30.0,
    )

    # Create orchestrator
    orchestrator = GPUOrchestrator(device_ids=[0], config=config)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
    print(f"Processing up to {max_frames} frames...")
    print()

    # Metrics
    processing_times = []
    motion_frames = 0
    detection_frames = 0
    all_events = []
    track_ids_seen = set()

    camera_id = 1
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = time.time()

        # Scale to 720p for processing
        if height > 720:
            scale = 720 / height
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # Process through pipeline
        t0 = time.perf_counter()
        result = orchestrator.process(
            camera_id=camera_id,
            frame=frame,
            timestamp=timestamp,
        )
        processing_time = (time.perf_counter() - t0) * 1000
        processing_times.append(processing_time)

        # Collect metrics
        if result.motion_detected:
            motion_frames += 1

        if result.detections:
            detection_frames += 1

        for track in result.tracks:
            track_ids_seen.add(track.track_id)

        for event in result.events:
            all_events.append(event)
            if event.event_type == EventType.ARRIVAL:
                print(f"[Frame {frame_count}] ARRIVAL: {event.class_name} (track {event.track_id})")
            elif event.event_type == EventType.DEPARTURE:
                print(f"[Frame {frame_count}] DEPARTURE: {event.class_name} (track {event.track_id}, "
                      f"duration={event.duration_seconds:.1f}s)")

        # Progress
        if frame_count % 100 == 0:
            print(f"  Frame {frame_count}: motion={motion_frames}, "
                  f"detections={detection_frames}, tracks={len(track_ids_seen)}")

    cap.release()

    # Results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Frames processed: {frame_count}")
    print()
    print("Timing:")
    print(f"  Mean time:  {np.mean(processing_times):.2f} ms")
    print(f"  Std dev:    {np.std(processing_times):.2f} ms")
    print(f"  P50:        {np.percentile(processing_times, 50):.2f} ms")
    print(f"  P95:        {np.percentile(processing_times, 95):.2f} ms")
    print(f"  FPS:        {1000/np.mean(processing_times):.1f}")
    print()
    print("Activity:")
    print(f"  Motion frames:    {motion_frames} ({100*motion_frames/frame_count:.1f}%)")
    print(f"  Detection frames: {detection_frames} ({100*detection_frames/frame_count:.1f}%)")
    print(f"  Unique tracks:    {len(track_ids_seen)}")
    print()
    print("Events:")
    arrivals = [e for e in all_events if e.event_type == EventType.ARRIVAL]
    departures = [e for e in all_events if e.event_type == EventType.DEPARTURE]
    print(f"  Arrivals:    {len(arrivals)}")
    print(f"  Departures:  {len(departures)}")
    print()
    print("Pipeline stats:")
    stats = orchestrator.stats
    print(f"  Cameras: {stats['total_cameras']}")
    for dev_id, pipe_stats in stats['pipelines'].items():
        print(f"  GPU {dev_id}: {pipe_stats['camera_count']} cameras")


def test_multi_camera(video_path: str, num_cameras: int = 4, frames_per_camera: int = 100):
    """Simulate processing frames from multiple cameras."""
    print("=" * 60)
    print(f"Multi-Camera Pipeline Test ({num_cameras} cameras)")
    print("=" * 60)

    model_path = Path("/opt3/ronin/ml_models/yolov8n.onnx")
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    config = GPUPipelineConfig(
        device_id=0,
        model_path=str(model_path),
        motion_min_percent=0.05,
        detection_confidence=0.4,
    )

    orchestrator = GPUOrchestrator(device_ids=[0], config=config)

    # Load video frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    # Read frames to process
    frames = []
    for _ in range(frames_per_camera):
        ret, frame = cap.read()
        if not ret:
            break
        # Scale to 720p
        height = frame.shape[0]
        if height > 720:
            scale = 720 / height
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        frames.append(frame)
    cap.release()

    print(f"Loaded {len(frames)} frames from video")
    print(f"Simulating {num_cameras} cameras with {len(frames)} frames each")
    print()

    # Process all cameras
    total_time = 0
    total_frames = 0
    events_per_camera = {cam: [] for cam in range(1, num_cameras + 1)}

    for frame_idx, frame in enumerate(frames):
        for camera_id in range(1, num_cameras + 1):
            timestamp = time.time()

            t0 = time.perf_counter()
            result = orchestrator.process(
                camera_id=camera_id,
                frame=frame,
                timestamp=timestamp,
            )
            total_time += time.perf_counter() - t0
            total_frames += 1

            for event in result.events:
                events_per_camera[camera_id].append(event)

        if (frame_idx + 1) % 25 == 0:
            elapsed = total_time
            fps = total_frames / elapsed
            print(f"  Frame {frame_idx + 1}/{len(frames)}: {fps:.1f} total FPS, "
                  f"{fps/num_cameras:.1f} per camera")

    # Results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total frames processed: {total_frames}")
    print(f"Total time: {total_time:.2f} s")
    print(f"Overall FPS: {total_frames/total_time:.1f}")
    print(f"Per-camera FPS: {total_frames/total_time/num_cameras:.1f}")
    print()
    print("Events per camera:")
    for camera_id, events in events_per_camera.items():
        arrivals = len([e for e in events if e.event_type == EventType.ARRIVAL])
        departures = len([e for e in events if e.event_type == EventType.DEPARTURE])
        print(f"  Camera {camera_id}: {arrivals} arrivals, {departures} departures")


def test_pipeline_components_timing(video_path: str, frames: int = 100):
    """Test timing of individual pipeline components."""
    print("=" * 60)
    print("Pipeline Component Timing Analysis")
    print("=" * 60)

    model_path = Path("/opt3/ronin/ml_models/yolov8n.onnx")

    config = GPUPipelineConfig(
        device_id=0,
        model_path=str(model_path),
    )

    orchestrator = GPUOrchestrator(device_ids=[0], config=config)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video")
        return

    motion_times = []
    detection_times = []
    tracking_times = []
    fsm_times = []

    for i in range(frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Scale to 720p
        height = frame.shape[0]
        if height > 720:
            scale = 720 / height
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        result = orchestrator.process(camera_id=1, frame=frame, timestamp=time.time())

        motion_times.append(result.motion_time_ms)
        detection_times.append(result.detection_time_ms)
        tracking_times.append(result.tracking_time_ms)
        fsm_times.append(result.fsm_time_ms)

    cap.release()

    print(f"\nComponent timing ({len(motion_times)} frames):\n")
    print(f"{'Component':<20} {'Mean':>10} {'Std':>10} {'P95':>10}")
    print("-" * 52)
    print(f"{'Motion (GPU MOG2)':<20} {np.mean(motion_times):>10.2f} {np.std(motion_times):>10.2f} {np.percentile(motion_times, 95):>10.2f}")
    print(f"{'Detection (YOLO)':<20} {np.mean(detection_times):>10.2f} {np.std(detection_times):>10.2f} {np.percentile(detection_times, 95):>10.2f}")
    print(f"{'Tracking (ByteTrack)':<20} {np.mean(tracking_times):>10.2f} {np.std(tracking_times):>10.2f} {np.percentile(tracking_times, 95):>10.2f}")
    print(f"{'FSM':<20} {np.mean(fsm_times):>10.2f} {np.std(fsm_times):>10.2f} {np.percentile(fsm_times, 95):>10.2f}")
    print("-" * 52)

    total_mean = np.mean(motion_times) + np.mean(detection_times) + np.mean(tracking_times) + np.mean(fsm_times)
    print(f"{'TOTAL':<20} {total_mean:>10.2f}")
    print(f"\nEffective FPS: {1000/total_mean:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Test nextgen detection pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to test video")
    parser.add_argument("--frames", type=int, default=300, help="Max frames to process")
    parser.add_argument("--cameras", type=int, default=0, help="Number of cameras to simulate")
    parser.add_argument("--timing-only", action="store_true", help="Only run timing analysis")
    args = parser.parse_args()

    if args.timing_only:
        test_pipeline_components_timing(args.video, frames=args.frames)
    elif args.cameras > 1:
        test_multi_camera(args.video, num_cameras=args.cameras, frames_per_camera=args.frames)
    else:
        test_single_camera(args.video, max_frames=args.frames)

    print()
    print("=" * 60)
    print("Full pipeline test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
