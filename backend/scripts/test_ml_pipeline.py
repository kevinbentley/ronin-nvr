#!/usr/bin/env python3
"""Test script for the ML inference pipeline.

This script tests the ML pipeline components:
1. Frame extraction from a video file
2. Model loading
3. Object detection
4. End-to-end job processing (if database available)

Usage:
    # Test with a video file
    python scripts/test_ml_pipeline.py /path/to/video.mp4

    # Test model loading only
    python scripts/test_ml_pipeline.py --model-only

    # Test with synthetic frame
    python scripts/test_ml_pipeline.py --synthetic
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


async def test_model_loading():
    """Test that the model can be loaded."""
    print("\n=== Testing Model Loading ===")

    from app.services.ml.model_manager import ModelManager
    from app.config import get_settings

    settings = get_settings()
    manager = ModelManager(settings.ml_models_directory)

    # List available models
    models = manager.list_available_models()
    print(f"Available models: {models}")

    if not models:
        print("ERROR: No models found in", settings.ml_models_directory)
        return False

    # Load default model
    model_name = models[0]
    print(f"Loading model: {model_name}")

    model = manager.load_model(model_name)
    if not model:
        print(f"ERROR: Failed to load model {model_name}")
        return False

    print(f"Model loaded successfully:")
    print(f"  - Input shape: {model.input_shape}")
    print(f"  - Input size: {model.input_size}")
    print(f"  - Classes: {len(model.class_names)}")
    print(f"  - Confidence threshold: {model.confidence_threshold}")

    return True


async def test_detection_synthetic():
    """Test detection with a synthetic frame."""
    print("\n=== Testing Detection (Synthetic Frame) ===")

    from app.services.ml.detection_service import DetectionService
    from app.services.ml.model_manager import ModelManager
    from app.config import get_settings

    settings = get_settings()
    manager = ModelManager(settings.ml_models_directory)
    detector = DetectionService(manager)

    # Create a synthetic frame (random noise)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    models = manager.list_available_models()
    if not models:
        print("ERROR: No models available")
        return False

    model_name = models[0]
    print(f"Running detection with model: {model_name}")
    print(f"Frame shape: {frame.shape}")

    results = detector.detect(frame, model_name)

    print(f"Detections found: {len(results)}")
    for r in results[:5]:  # Show first 5
        print(f"  - {r.class_name}: {r.confidence:.2f} at ({r.x:.2f}, {r.y:.2f})")

    print("Detection test passed!")
    return True


async def test_frame_extraction(video_path: Path):
    """Test frame extraction from a video file."""
    print(f"\n=== Testing Frame Extraction ===")
    print(f"Video: {video_path}")

    from app.services.ml.frame_extractor import FrameExtractor

    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        return False

    extractor = FrameExtractor(fps=1.0, max_dimension=640)

    # Get video info
    info = await extractor.get_video_info(video_path)
    if not info:
        print("ERROR: Could not get video info")
        return False

    print(f"Video info:")
    print(f"  - Duration: {info.duration_seconds:.1f}s")
    print(f"  - Resolution: {info.width}x{info.height}")
    print(f"  - FPS: {info.fps}")
    print(f"  - Codec: {info.codec}")
    print(f"  - Expected frames at 1 fps: {int(info.duration_seconds)}")

    # Extract a few frames
    frame_count = 0
    async for frame, frame_num, timestamp_ms in extractor.extract_frames(video_path):
        frame_count += 1
        if frame_count <= 3:
            print(f"  Frame {frame_num}: shape={frame.shape}, timestamp={timestamp_ms:.0f}ms")
        if frame_count >= 5:
            break

    print(f"Extracted {frame_count} frames successfully!")
    return True


async def test_detection_on_video(video_path: Path):
    """Test detection on frames from a video file."""
    print(f"\n=== Testing Detection on Video ===")

    from app.services.ml.frame_extractor import FrameExtractor
    from app.services.ml.detection_service import DetectionService
    from app.services.ml.model_manager import ModelManager
    from app.config import get_settings

    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        return False

    settings = get_settings()
    manager = ModelManager(settings.ml_models_directory)
    detector = DetectionService(manager)
    extractor = FrameExtractor(fps=1.0, max_dimension=640)

    models = manager.list_available_models()
    if not models:
        print("ERROR: No models available")
        return False

    model_name = models[0]
    print(f"Processing with model: {model_name}")

    total_detections = 0
    frames_processed = 0
    detection_counts = {}

    async for frame, frame_num, timestamp_ms in extractor.extract_frames(video_path):
        results = detector.detect(frame, model_name)
        frames_processed += 1

        for r in results:
            total_detections += 1
            detection_counts[r.class_name] = detection_counts.get(r.class_name, 0) + 1

        if frames_processed <= 5 and results:
            print(f"  Frame {frame_num}: {len(results)} detections")
            for r in results[:3]:
                print(f"    - {r.class_name}: {r.confidence:.2f}")

        if frames_processed >= 10:
            break

    print(f"\nResults ({frames_processed} frames processed):")
    print(f"  Total detections: {total_detections}")
    if detection_counts:
        print("  Detection breakdown:")
        for class_name, count in sorted(detection_counts.items(), key=lambda x: -x[1]):
            print(f"    - {class_name}: {count}")

    return True


async def main():
    parser = argparse.ArgumentParser(description="Test ML inference pipeline")
    parser.add_argument(
        "video_path",
        nargs="?",
        type=Path,
        help="Path to video file to test with",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Only test model loading",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Test with synthetic frame (no video needed)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ML Pipeline Test")
    print("=" * 60)

    # Always test model loading
    if not await test_model_loading():
        print("\nERROR: Model loading failed!")
        return 1

    if args.model_only:
        print("\n✅ Model loading test passed!")
        return 0

    if args.synthetic:
        if not await test_detection_synthetic():
            print("\nERROR: Synthetic detection failed!")
            return 1
        print("\n✅ All synthetic tests passed!")
        return 0

    if not args.video_path:
        print("\nNo video path provided. Testing with synthetic frame...")
        if not await test_detection_synthetic():
            print("\nERROR: Synthetic detection failed!")
            return 1
        print("\n✅ All tests passed!")
        return 0

    # Full video test
    if not await test_frame_extraction(args.video_path):
        print("\nERROR: Frame extraction failed!")
        return 1

    if not await test_detection_on_video(args.video_path):
        print("\nERROR: Video detection failed!")
        return 1

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
