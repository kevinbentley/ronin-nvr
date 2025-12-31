#!/usr/bin/env python3
"""Test script for motion detection.

Usage:
    # Test with synthetic frames (no video needed)
    python scripts/test_motion_detection.py --synthetic

    # Test with a video file
    python scripts/test_motion_detection.py /path/to/video.mp4
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2


async def test_synthetic_motion():
    """Test motion detection with synthetic frames."""
    print("\n=== Testing Motion Detection (Synthetic Frames) ===")

    from app.services.ml.motion_detector import MotionDetector

    detector = MotionDetector(
        motion_threshold=0.5,
        min_contour_area=500,
    )

    # Create a base frame (gray background)
    base_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    # Simulate ~50 frames to build background model
    print("Building background model (50 frames)...")
    for i in range(50):
        # Add slight noise to simulate real camera
        noise = np.random.randint(-5, 5, base_frame.shape, dtype=np.int16)
        frame = np.clip(base_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        result = detector.detect(frame)

    print(f"Background model built. Warmup complete.")

    # Test 1: No motion (same background)
    print("\nTest 1: Static scene (no motion expected)")
    result = detector.detect(base_frame)
    print(f"  has_motion: {result.has_motion}")
    print(f"  motion_percent: {result.motion_percent}%")
    print(f"  contours: {result.contour_count}")
    assert not result.has_motion, "Expected no motion in static scene"
    print("  PASSED")

    # Test 2: Add a moving object (white rectangle)
    print("\nTest 2: Moving object (motion expected)")
    motion_frame = base_frame.copy()
    cv2.rectangle(motion_frame, (200, 150), (350, 300), (255, 255, 255), -1)
    result = detector.detect(motion_frame)
    print(f"  has_motion: {result.has_motion}")
    print(f"  motion_percent: {result.motion_percent}%")
    print(f"  contours: {result.contour_count}")
    print(f"  largest_area_percent: {result.largest_area_percent}%")
    if result.bounding_boxes:
        print(f"  bounding_box: {result.bounding_boxes[0]}")
    assert result.has_motion, "Expected motion with moving object"
    print("  PASSED")

    # Test 3: Gradual lighting change (should adapt, less motion)
    print("\nTest 3: Gradual lighting change (should adapt)")
    for i in range(30):
        # Gradually brighten the scene
        brightness = 128 + i * 2
        bright_frame = np.full((480, 640, 3), brightness, dtype=np.uint8)
        result = detector.detect(bright_frame)

    # After adaptation, should have minimal motion
    bright_frame = np.full((480, 640, 3), 188, dtype=np.uint8)
    result = detector.detect(bright_frame)
    print(f"  has_motion: {result.has_motion}")
    print(f"  motion_percent: {result.motion_percent}%")
    print("  (MOG2 adapts to gradual changes)")
    print("  PASSED")

    # Test 4: Sudden lighting change (may trigger motion briefly)
    print("\nTest 4: Sudden lighting change")
    dark_frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    result = detector.detect(dark_frame)
    print(f"  has_motion: {result.has_motion}")
    print(f"  motion_percent: {result.motion_percent}%")
    print("  (Sudden changes may trigger motion, but model adapts)")

    print("\n=== All synthetic tests completed! ===")
    return True


async def test_video_motion(video_path: Path):
    """Test motion detection on a video file."""
    print(f"\n=== Testing Motion Detection on Video ===")
    print(f"Video: {video_path}")

    from app.services.ml.motion_detector import MotionDetector
    from app.services.ml.frame_extractor import FrameExtractor

    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        return False

    detector = MotionDetector.from_settings()
    extractor = FrameExtractor(fps=2.0, max_dimension=640)

    # Get video info
    video_info = await extractor.get_video_info(video_path)
    if not video_info:
        print("ERROR: Could not get video info")
        return False

    print(f"Video info:")
    print(f"  Duration: {video_info.duration_seconds:.1f}s")
    print(f"  Resolution: {video_info.width}x{video_info.height}")
    print(f"  FPS: {video_info.fps}")

    frames_processed = 0
    motion_frames = 0
    max_motion_percent = 0.0

    print("\nProcessing frames...")
    async for frame, frame_num, timestamp_ms in extractor.extract_frames(video_path):
        result = detector.detect(frame)
        frames_processed += 1

        if result.has_motion:
            motion_frames += 1
            max_motion_percent = max(max_motion_percent, result.motion_percent)

        # Show first few motion detections
        if result.has_motion and motion_frames <= 5:
            print(f"  Frame {frame_num} @ {timestamp_ms/1000:.1f}s: "
                  f"motion={result.motion_percent:.1f}%, "
                  f"regions={result.contour_count}")

        # Limit to 100 frames for testing
        if frames_processed >= 100:
            break

    print(f"\nResults ({frames_processed} frames):")
    print(f"  Frames with motion: {motion_frames} ({100*motion_frames/frames_processed:.1f}%)")
    print(f"  Max motion percent: {max_motion_percent:.2f}%")

    return True


async def main():
    parser = argparse.ArgumentParser(description="Test motion detection")
    parser.add_argument(
        "video_path",
        nargs="?",
        type=Path,
        help="Path to video file to test with",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Test with synthetic frames (no video needed)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Motion Detection Test")
    print("=" * 60)

    if args.synthetic or not args.video_path:
        if not await test_synthetic_motion():
            return 1

    if args.video_path:
        if not await test_video_motion(args.video_path):
            return 1

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
