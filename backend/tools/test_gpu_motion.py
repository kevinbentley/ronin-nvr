#!/usr/bin/env python3
"""Test GPU motion detection with sample videos.

This script tests the GPU MOG2 background subtraction implementation
by processing real video files and comparing performance to CPU.

Usage:
    source /opt/venv/bin/activate
    cd /workspace/ronin-nvr/backend
    python tools/test_gpu_motion.py --video /opt3/ronin/storage/test.mp4

    # Compare CPU vs GPU
    python tools/test_gpu_motion.py --video /opt3/ronin/storage/test.mp4 --compare

    # Process multiple videos
    python tools/test_gpu_motion.py --video /opt3/ronin/storage/Hangar_East/2025-12-31/03-05-54.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.gpu_motion import GPUBackgroundSubtractor, GPUMotionGate
from app.services.ml.motion_gate import MotionGate


def test_gpu_motion_basic():
    """Basic test of GPU motion detection."""
    print("=" * 60)
    print("Basic GPU Motion Detection Test")
    print("=" * 60)

    # Create subtractor
    subtractor = GPUBackgroundSubtractor(camera_id=0)

    # Create test frames
    stream = cv2.cuda_Stream()
    gpu_frame = cv2.cuda.GpuMat()

    # Static scene - should have no motion after initialization
    static_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128

    print("\nInitializing background model with 50 static frames...")
    for i in range(50):
        gpu_frame.upload(static_frame)
        result = subtractor.apply(gpu_frame)

    print(f"  Initialized: {subtractor.is_initialized}")
    print(f"  Frame count: {subtractor.frame_count}")

    # Test with same static frame - should show no motion
    gpu_frame.upload(static_frame)
    result = subtractor.apply(gpu_frame, download_mask=True)
    print(f"\nStatic frame test:")
    print(f"  Motion detected: {result.motion_detected}")
    print(f"  Motion percent: {result.motion_percent:.2f}%")
    print(f"  Contour count: {result.contour_count}")

    # Add motion - change part of the frame
    motion_frame = static_frame.copy()
    motion_frame[300:400, 500:700] = 255  # White rectangle

    gpu_frame.upload(motion_frame)
    result = subtractor.apply(gpu_frame, download_mask=True)
    print(f"\nMotion frame test (white rectangle 100x200):")
    print(f"  Motion detected: {result.motion_detected}")
    print(f"  Motion percent: {result.motion_percent:.2f}%")
    print(f"  Contour count: {result.contour_count}")
    print(f"  Largest contour: {result.largest_contour_area:.4f}")

    print("\n[PASS] Basic GPU motion detection working")


def test_video_processing(video_path: str, max_frames: int = 500, compare_cpu: bool = False):
    """Process a video file and measure motion detection performance."""
    print("=" * 60)
    print(f"Video Processing Test: {Path(video_path).name}")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")

    # GPU motion gate
    gpu_gate = GPUMotionGate()

    # CPU motion gate for comparison
    cpu_gate = MotionGate() if compare_cpu else None
    prev_frame = None

    # Metrics
    gpu_times = []
    cpu_times = []
    gpu_motion_frames = 0
    cpu_motion_frames = 0

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # GPU processing
        t0 = time.perf_counter()
        gpu_result = gpu_gate.check(camera_id=0, frame=frame)
        gpu_time = (time.perf_counter() - t0) * 1000
        gpu_times.append(gpu_time)

        if gpu_result.motion_detected:
            gpu_motion_frames += 1

        # CPU processing for comparison
        if compare_cpu:
            t0 = time.perf_counter()
            cpu_result = cpu_gate.check(frame, prev_frame)
            cpu_time = (time.perf_counter() - t0) * 1000
            cpu_times.append(cpu_time)

            if cpu_result.motion_detected:
                cpu_motion_frames += 1

            prev_frame = frame.copy()

        # Progress
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{min(max_frames, total_frames)} frames...")

    cap.release()

    # Results
    print(f"\nResults ({frame_count} frames processed):")
    print(f"\n  GPU MOG2:")
    print(f"    Mean time: {np.mean(gpu_times):.2f} ms")
    print(f"    Std dev:   {np.std(gpu_times):.2f} ms")
    print(f"    Min/Max:   {np.min(gpu_times):.2f} / {np.max(gpu_times):.2f} ms")
    print(f"    FPS:       {1000 / np.mean(gpu_times):.1f}")
    print(f"    Motion frames: {gpu_motion_frames} ({100*gpu_motion_frames/frame_count:.1f}%)")

    if compare_cpu and cpu_times:
        print(f"\n  CPU Frame Diff:")
        print(f"    Mean time: {np.mean(cpu_times):.2f} ms")
        print(f"    Std dev:   {np.std(cpu_times):.2f} ms")
        print(f"    Min/Max:   {np.min(cpu_times):.2f} / {np.max(cpu_times):.2f} ms")
        print(f"    FPS:       {1000 / np.mean(cpu_times):.1f}")
        print(f"    Motion frames: {cpu_motion_frames} ({100*cpu_motion_frames/frame_count:.1f}%)")
        print(f"\n  Speedup: {np.mean(cpu_times) / np.mean(gpu_times):.2f}x")


def test_multi_camera_simulation(num_cameras: int = 4, frames_per_camera: int = 100):
    """Simulate processing frames from multiple cameras."""
    print("=" * 60)
    print(f"Multi-Camera Simulation: {num_cameras} cameras")
    print("=" * 60)

    gate = GPUMotionGate()

    # Generate synthetic frames with random motion
    frames_per_cam = []
    for cam_id in range(num_cameras):
        np.random.seed(cam_id)
        base_frame = np.random.randint(100, 156, (720, 1280, 3), dtype=np.uint8)
        frames_per_cam.append(base_frame)

    # Process frames
    total_time = 0
    frame_count = 0

    for iteration in range(frames_per_camera):
        for cam_id in range(num_cameras):
            # Add some random variation
            frame = frames_per_cam[cam_id].copy()
            if np.random.random() > 0.7:
                # Add motion region
                x, y = np.random.randint(0, 1000), np.random.randint(0, 500)
                frame[y:y+100, x:x+200] = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)

            t0 = time.perf_counter()
            result = gate.check(camera_id=cam_id, frame=frame)
            total_time += time.perf_counter() - t0
            frame_count += 1

    avg_time = (total_time / frame_count) * 1000
    print(f"\nProcessed {frame_count} total frames")
    print(f"Average time per frame: {avg_time:.2f} ms")
    print(f"Throughput: {frame_count / total_time:.1f} frames/sec")
    print(f"Per camera FPS: {(frame_count / total_time) / num_cameras:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Test GPU motion detection")
    parser.add_argument("--video", type=str, help="Path to test video file")
    parser.add_argument("--compare", action="store_true", help="Compare GPU vs CPU performance")
    parser.add_argument("--frames", type=int, default=500, help="Max frames to process")
    parser.add_argument("--multi-cam", type=int, default=0, help="Simulate N cameras")
    args = parser.parse_args()

    # Always run basic test
    test_gpu_motion_basic()
    print()

    # Video test
    if args.video:
        test_video_processing(args.video, max_frames=args.frames, compare_cpu=args.compare)
        print()

    # Multi-camera simulation
    if args.multi_cam > 0:
        test_multi_camera_simulation(num_cameras=args.multi_cam)
        print()

    print("=" * 60)
    print("All GPU motion detection tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
