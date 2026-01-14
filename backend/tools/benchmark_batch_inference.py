#!/usr/bin/env python3
"""Benchmark batch inference vs sequential processing.

Compares performance of:
1. Sequential processing (one camera at a time)
2. Batched processing (all cameras in parallel)

Usage:
    source /opt/venv/bin/activate
    cd /workspace/ronin-nvr/backend
    python tools/benchmark_batch_inference.py --cameras 8
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from app.services.ml.gpu_orchestrator import GPUOrchestrator, GPUPipelineConfig


def generate_test_frames(
    num_cameras: int,
    frame_size: tuple[int, int] = (1280, 720),
    add_motion: bool = False,
) -> dict[int, np.ndarray]:
    """Generate test frames for multiple cameras.

    Args:
        num_cameras: Number of camera frames to generate
        frame_size: Frame dimensions (width, height)
        add_motion: Add synthetic motion to frames

    Returns:
        Dict mapping camera_id to frame
    """
    frames = {}
    width, height = frame_size

    for cam_id in range(1, num_cameras + 1):
        # Create base frame with some variation per camera
        frame = np.random.randint(50, 150, (height, width, 3), dtype=np.uint8)

        if add_motion:
            # Add a moving rectangle to simulate motion
            x = (cam_id * 100 + int(time.time() * 100)) % (width - 100)
            y = (cam_id * 50) % (height - 100)
            cv2.rectangle(frame, (x, y), (x + 100, y + 80), (255, 255, 255), -1)

        frames[cam_id] = frame

    return frames


def load_video_frames(
    video_path: str,
    num_cameras: int,
    start_frame: int = 0,
) -> dict[int, np.ndarray]:
    """Load frames from a video file, simulating multiple cameras.

    Args:
        video_path: Path to video file
        num_cameras: Number of camera frames to extract
        start_frame: Starting frame offset

    Returns:
        Dict mapping camera_id to frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = {}
    for cam_id in range(1, num_cameras + 1):
        ret, frame = cap.read()
        if not ret:
            # Loop back to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        if ret:
            # Scale to 720p
            if frame.shape[0] > 720:
                scale = 720 / frame.shape[0]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            frames[cam_id] = frame

    cap.release()
    return frames


class VideoFrameGenerator:
    """Generate sequential frames from video to simulate real camera feeds."""

    def __init__(self, video_path: str, num_cameras: int):
        """Initialize with video path and number of camera streams to simulate."""
        self.video_path = video_path
        self.num_cameras = num_cameras
        self.caps = []
        self.frame_counts = []

        # Open video for each camera (at different positions)
        total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        spacing = total_frames // (num_cameras + 1)

        for cam_id in range(num_cameras):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            # Start each camera at different point in video
            start_pos = (cam_id + 1) * spacing
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
            self.caps.append(cap)
            self.frame_counts.append(0)

    def get_frames(self) -> dict[int, np.ndarray]:
        """Get next frame from each simulated camera."""
        frames = {}

        for cam_id, cap in enumerate(self.caps, start=1):
            ret, frame = cap.read()
            if not ret:
                # Loop back
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            if ret:
                # Scale to 720p
                if frame.shape[0] > 720:
                    scale = 720 / frame.shape[0]
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                frames[cam_id] = frame
                self.frame_counts[cam_id - 1] += 1

        return frames

    def close(self):
        """Release video captures."""
        for cap in self.caps:
            cap.release()


def benchmark_sequential(
    orchestrator: GPUOrchestrator,
    frames: dict[int, np.ndarray],
    num_iterations: int,
) -> dict:
    """Benchmark sequential processing (one camera at a time).

    Returns:
        Dict with timing statistics
    """
    times = []
    total_events = 0

    for _ in range(num_iterations):
        timestamp = time.time()
        t0 = time.perf_counter()

        for camera_id, frame in frames.items():
            result = orchestrator.process(camera_id, frame, timestamp)
            total_events += len(result.events)

        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    return {
        "mode": "sequential",
        "cameras": len(frames),
        "iterations": num_iterations,
        "total_frames": len(frames) * num_iterations,
        "total_time_sec": sum(times),
        "mean_batch_ms": np.mean(times) * 1000,
        "std_batch_ms": np.std(times) * 1000,
        "min_batch_ms": np.min(times) * 1000,
        "max_batch_ms": np.max(times) * 1000,
        "fps": len(frames) * num_iterations / sum(times),
        "frames_per_batch_per_sec": len(frames) / np.mean(times),
        "events": total_events,
    }


def benchmark_batched(
    orchestrator: GPUOrchestrator,
    frames: dict[int, np.ndarray],
    num_iterations: int,
) -> dict:
    """Benchmark batched processing (all cameras together).

    Returns:
        Dict with timing statistics
    """
    times = []
    total_events = 0

    for _ in range(num_iterations):
        timestamp = time.time()
        t0 = time.perf_counter()

        results = orchestrator.process_batch(frames, timestamp)

        for result in results.values():
            total_events += len(result.events)

        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    return {
        "mode": "batched",
        "cameras": len(frames),
        "iterations": num_iterations,
        "total_frames": len(frames) * num_iterations,
        "total_time_sec": sum(times),
        "mean_batch_ms": np.mean(times) * 1000,
        "std_batch_ms": np.std(times) * 1000,
        "min_batch_ms": np.min(times) * 1000,
        "max_batch_ms": np.max(times) * 1000,
        "fps": len(frames) * num_iterations / sum(times),
        "frames_per_batch_per_sec": len(frames) / np.mean(times),
        "events": total_events,
    }


def benchmark_streaming_sequential(
    orchestrator: GPUOrchestrator,
    video_generator: "VideoFrameGenerator",
    num_iterations: int,
) -> dict:
    """Benchmark sequential processing with streaming video frames.

    This simulates real-world conditions where frames change over time,
    triggering proper motion detection.
    """
    times = []
    total_events = 0
    motion_frames = 0
    detection_frames = 0

    for _ in range(num_iterations):
        frames = video_generator.get_frames()
        timestamp = time.time()
        t0 = time.perf_counter()

        for camera_id, frame in frames.items():
            result = orchestrator.process(camera_id, frame, timestamp)
            total_events += len(result.events)
            if result.motion_detected:
                motion_frames += 1
            if result.detections:
                detection_frames += 1

        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    num_cameras = video_generator.num_cameras
    return {
        "mode": "sequential",
        "cameras": num_cameras,
        "iterations": num_iterations,
        "total_frames": num_cameras * num_iterations,
        "total_time_sec": sum(times),
        "mean_batch_ms": np.mean(times) * 1000,
        "std_batch_ms": np.std(times) * 1000,
        "min_batch_ms": np.min(times) * 1000,
        "max_batch_ms": np.max(times) * 1000,
        "fps": num_cameras * num_iterations / sum(times),
        "frames_per_batch_per_sec": num_cameras / np.mean(times),
        "events": total_events,
        "motion_frames": motion_frames,
        "detection_frames": detection_frames,
    }


def benchmark_streaming_batched(
    orchestrator: GPUOrchestrator,
    video_generator: "VideoFrameGenerator",
    num_iterations: int,
) -> dict:
    """Benchmark batched processing with streaming video frames."""
    times = []
    total_events = 0
    motion_frames = 0
    detection_frames = 0

    for _ in range(num_iterations):
        frames = video_generator.get_frames()
        timestamp = time.time()
        t0 = time.perf_counter()

        results = orchestrator.process_batch(frames, timestamp)

        for result in results.values():
            total_events += len(result.events)
            if result.motion_detected:
                motion_frames += 1
            if result.detections:
                detection_frames += 1

        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    num_cameras = video_generator.num_cameras
    return {
        "mode": "batched",
        "cameras": num_cameras,
        "iterations": num_iterations,
        "total_frames": num_cameras * num_iterations,
        "total_time_sec": sum(times),
        "mean_batch_ms": np.mean(times) * 1000,
        "std_batch_ms": np.std(times) * 1000,
        "min_batch_ms": np.min(times) * 1000,
        "max_batch_ms": np.max(times) * 1000,
        "fps": num_cameras * num_iterations / sum(times),
        "frames_per_batch_per_sec": num_cameras / np.mean(times),
        "events": total_events,
        "motion_frames": motion_frames,
        "detection_frames": detection_frames,
    }


def benchmark_motion_scenarios(
    orchestrator: GPUOrchestrator,
    num_cameras: int,
    num_iterations: int,
    video_path: str = None,
) -> list[dict]:
    """Benchmark with different motion scenarios.

    Returns:
        List of benchmark results for each scenario
    """
    results = []

    # Scenario 1: Streaming video (realistic motion detection)
    if video_path and Path(video_path).exists():
        print(f"\n--- Scenario: Streaming Video ({Path(video_path).name}) ---")
        print(f"    Simulating {num_cameras} cameras with sequential video frames")

        # Create generator for sequential benchmarks
        generator_seq = VideoFrameGenerator(video_path, num_cameras)
        seq_result = benchmark_streaming_sequential(orchestrator, generator_seq, num_iterations)
        generator_seq.close()

        # Reset state
        for cam_id in range(1, num_cameras + 1):
            orchestrator.reset_camera(cam_id)

        # Create new generator for batched benchmarks
        generator_bat = VideoFrameGenerator(video_path, num_cameras)
        bat_result = benchmark_streaming_batched(orchestrator, generator_bat, num_iterations)
        generator_bat.close()

        results.append({"scenario": "streaming_video", "sequential": seq_result, "batched": bat_result})

        print(f"Sequential: {seq_result['fps']:.1f} FPS ({seq_result['mean_batch_ms']:.1f}ms per batch)")
        print(f"  Motion: {seq_result['motion_frames']}/{seq_result['total_frames']} frames")
        print(f"  Detections: {seq_result['detection_frames']} frames with objects")
        print(f"Batched:    {bat_result['fps']:.1f} FPS ({bat_result['mean_batch_ms']:.1f}ms per batch)")
        print(f"  Motion: {bat_result['motion_frames']}/{bat_result['total_frames']} frames")
        print(f"  Detections: {bat_result['detection_frames']} frames with objects")
        speedup = bat_result['fps'] / seq_result['fps'] if seq_result['fps'] > 0 else 0
        print(f"Speedup:    {speedup:.2f}x")

        # Reset state
        for cam_id in range(1, num_cameras + 1):
            orchestrator.reset_camera(cam_id)

    # Scenario 2: Direct detector benchmark (bypasses motion gate entirely)
    print(f"\n--- Scenario: Direct YOLO Inference (bypass entire pipeline) ---")
    print(f"    Measuring pure YOLO detect() throughput")

    if video_path and Path(video_path).exists():
        frames = load_video_frames(video_path, num_cameras, start_frame=5000)
    else:
        frames = generate_test_frames(num_cameras, add_motion=True)

    frame_list = list(frames.values())

    # Get detector directly from pipeline
    pipeline = orchestrator._pipelines[0]
    detector = pipeline._detector

    # Sequential detection (direct detector calls)
    seq_times = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        for frame in frame_list:
            detector.detect(frame)
        seq_times.append(time.perf_counter() - t0)

    seq_fps = len(frame_list) * num_iterations / sum(seq_times)
    seq_batch_ms = np.mean(seq_times) * 1000

    # Try batched detection if model supports dynamic batch
    bat_fps = seq_fps
    bat_batch_ms = seq_batch_ms
    batch_supported = False

    try:
        detector.detect_batch(frame_list)
        batch_supported = True

        bat_times = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            detector.detect_batch(frame_list)
            bat_times.append(time.perf_counter() - t0)

        bat_fps = len(frame_list) * num_iterations / sum(bat_times)
        bat_batch_ms = np.mean(bat_times) * 1000
    except Exception as e:
        print(f"    Note: Batch inference not supported (model has fixed batch size)")
        print(f"    To enable batching, re-export model with: dynamic=True")

    seq_result = {"fps": seq_fps, "mean_batch_ms": seq_batch_ms}
    bat_result = {"fps": bat_fps, "mean_batch_ms": bat_batch_ms}
    results.append({"scenario": "direct_yolo", "sequential": seq_result, "batched": bat_result})

    print(f"Sequential: {seq_fps:.1f} FPS ({seq_batch_ms:.1f}ms per batch of {num_cameras})")
    if batch_supported:
        print(f"Batched:    {bat_fps:.1f} FPS ({bat_batch_ms:.1f}ms per batch of {num_cameras})")
        speedup = bat_fps / seq_fps if seq_fps > 0 else 0
        print(f"Speedup:    {speedup:.2f}x")
    else:
        print(f"Per-frame:  {seq_fps / num_cameras:.1f} FPS per camera")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark batch inference vs sequential processing"
    )
    parser.add_argument("--cameras", type=int, default=8,
                       help="Number of cameras to simulate (default: 8)")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of iterations per benchmark (default: 100)")
    parser.add_argument("--video", type=str, default="/workspace/ronin-nvr/15-01-38.mp4",
                       help="Video file for real-world testing")
    parser.add_argument("--model-path", type=str,
                       default="/opt3/ronin/ml_models/yolov8n.onnx",
                       help="Path to ONNX model")

    args = parser.parse_args()

    print("=" * 70)
    print("Batch Inference Benchmark")
    print("=" * 70)
    print(f"Cameras: {args.cameras}")
    print(f"Iterations: {args.iterations}")
    print(f"Model: {args.model_path}")

    # Initialize orchestrator with tuned config
    print("\nInitializing GPU pipeline...")
    config = GPUPipelineConfig(
        device_id=0,
        model_path=args.model_path,
        motion_min_percent=0.3,
        detection_confidence=0.65,
        track_high_thresh=0.65,
        track_min_hits=5,
        track_min_displacement=0.02,
        fsm_validation_frames=10,
    )

    orchestrator = GPUOrchestrator(device_ids=[0], config=config)

    # Run benchmarks
    results = benchmark_motion_scenarios(
        orchestrator,
        num_cameras=args.cameras,
        num_iterations=args.iterations,
        video_path=args.video,
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Scenario':<20} {'Sequential FPS':<18} {'Batched FPS':<18} {'Speedup':<10}")
    print("-" * 66)

    for r in results:
        scenario = r["scenario"]
        seq_fps = r["sequential"]["fps"]
        bat_fps = r["batched"]["fps"]
        speedup = bat_fps / seq_fps

        print(f"{scenario:<20} {seq_fps:<18.1f} {bat_fps:<18.1f} {speedup:<10.2f}x")

    # Capacity analysis
    print("\n" + "=" * 70)
    print("CAPACITY ANALYSIS")
    print("=" * 70)

    # Use direct_yolo (worst-case, all cameras running YOLO) for conservative estimate
    worst_case = next((r for r in results if r["scenario"] == "direct_yolo"), results[-1])
    batched_fps = worst_case["batched"]["fps"]
    seq_fps = worst_case["sequential"]["fps"]

    print(f"\nWorst-case throughput (forced detection on all cameras):")
    print(f"  Sequential: {seq_fps:.1f} FPS")
    print(f"  Batched:    {batched_fps:.1f} FPS")
    print(f"Cameras in test: {args.cameras}")
    print(f"Per-camera rate (batched): {batched_fps / args.cameras:.1f} FPS per camera")

    # Calculate max cameras at different target FPS
    print("\nMaximum cameras at target per-camera FPS (batched):")
    for target_fps in [10, 15, 20, 30]:
        max_cameras = int(batched_fps / target_fps)
        print(f"  {target_fps} FPS/camera: {max_cameras} cameras")

    print()


if __name__ == "__main__":
    main()
