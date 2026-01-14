#!/usr/bin/env python3
"""Test TensorRT inference pipeline.

Usage:
    source /opt/venv/bin/activate
    cd /workspace/ronin-nvr/backend
    python tools/test_tensorrt_inference.py --video /opt3/ronin/storage/test.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.tensorrt_inference import TensorRTDetector, TRT_AVAILABLE
from app.services.ml.nvdec_extractor import NVDECExtractor


def test_basic_inference():
    """Test basic TensorRT inference with dummy data."""
    print("=" * 60)
    print("Basic TensorRT Inference Test")
    print("=" * 60)

    if not TRT_AVAILABLE:
        print("TensorRT not available, skipping")
        return

    engine_path = Path("/opt3/ronin/ml_models/yolov8n.engine")
    onnx_path = Path("/opt3/ronin/ml_models/yolov8n.onnx")

    model_path = engine_path if engine_path.exists() else onnx_path
    print(f"Using model: {model_path}")

    # Create detector
    detector = TensorRTDetector(
        model_path=model_path,
        precision="fp16",
        confidence_threshold=0.5,
        warmup_iterations=10,
    )

    # Create test image
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Run inference
    t0 = time.perf_counter()
    detections = detector.detect(test_frame)
    inference_time = (time.perf_counter() - t0) * 1000

    print(f"\nRandom frame test:")
    print(f"  Inference time: {inference_time:.1f} ms")
    print(f"  Detections: {len(detections)}")

    # Benchmark with multiple runs
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        detector.detect(test_frame)
        times.append((time.perf_counter() - t0) * 1000)

    print(f"\nBenchmark (50 runs):")
    print(f"  Mean: {np.mean(times):.2f} ms")
    print(f"  Std:  {np.std(times):.2f} ms")
    print(f"  Min:  {np.min(times):.2f} ms")
    print(f"  Max:  {np.max(times):.2f} ms")
    print(f"  FPS:  {1000/np.mean(times):.1f}")

    print("\n[PASS] Basic inference test")


def test_video_inference(video_path: str, max_frames: int = 300):
    """Test inference on video file."""
    print("=" * 60)
    print(f"Video Inference Test: {Path(video_path).name}")
    print("=" * 60)

    engine_path = Path("/opt3/ronin/ml_models/yolov8n.engine")
    onnx_path = Path("/opt3/ronin/ml_models/yolov8n.onnx")

    model_path = engine_path if engine_path.exists() else onnx_path
    print(f"Model: {model_path}")

    # Create detector
    detector = TensorRTDetector(
        model_path=model_path,
        precision="fp16",
        confidence_threshold=0.5,
        warmup_iterations=10,
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f} fps, {total} frames")

    # Process frames
    inference_times = []
    total_detections = 0
    frames_with_detections = 0
    detection_classes = {}

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        t0 = time.perf_counter()
        detections = detector.detect(frame)
        inference_time = (time.perf_counter() - t0) * 1000
        inference_times.append(inference_time)

        if detections:
            frames_with_detections += 1
            total_detections += len(detections)
            for det in detections:
                detection_classes[det.class_name] = detection_classes.get(det.class_name, 0) + 1

        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{min(max_frames, total)} frames...")

    cap.release()

    # Results
    print(f"\nResults ({frame_count} frames):")
    print(f"  Inference time:")
    print(f"    Mean: {np.mean(inference_times):.2f} ms")
    print(f"    Std:  {np.std(inference_times):.2f} ms")
    print(f"    P50:  {np.percentile(inference_times, 50):.2f} ms")
    print(f"    P95:  {np.percentile(inference_times, 95):.2f} ms")
    print(f"    FPS:  {1000/np.mean(inference_times):.1f}")

    print(f"\n  Detections:")
    print(f"    Total: {total_detections}")
    print(f"    Frames with detections: {frames_with_detections} ({100*frames_with_detections/frame_count:.1f}%)")

    if detection_classes:
        print(f"    By class:")
        for cls, count in sorted(detection_classes.items(), key=lambda x: -x[1])[:10]:
            print(f"      {cls}: {count}")


def test_nvdec_pipeline(video_path: str, max_frames: int = 100):
    """Test NVDEC + TensorRT pipeline."""
    print("=" * 60)
    print(f"NVDEC + TensorRT Pipeline Test")
    print("=" * 60)

    engine_path = Path("/opt3/ronin/ml_models/yolov8n.engine")
    onnx_path = Path("/opt3/ronin/ml_models/yolov8n.onnx")

    model_path = engine_path if engine_path.exists() else onnx_path

    # Create detector
    detector = TensorRTDetector(
        model_path=model_path,
        precision="fp16",
        confidence_threshold=0.5,
        warmup_iterations=10,
    )

    # Create NVDEC extractor
    extractor = NVDECExtractor(video_path)
    if not extractor.open():
        print("Failed to open video")
        return

    print(f"Video info: {extractor.info}")
    print(f"Using NVDEC: {extractor.using_hardware_decode}")

    # Process with GPU frames
    decode_times = []
    inference_times = []
    total_times = []

    for i, gpu_frame in enumerate(extractor.frames(max_frames=max_frames)):
        if i >= max_frames:
            break

        # Time decode (already done in frames())
        t_decode = time.perf_counter()

        # Download and detect
        frame = gpu_frame.download()
        t_download = time.perf_counter()

        detections = detector.detect(frame)
        t_inference = time.perf_counter()

        decode_times.append((t_download - t_decode) * 1000)
        inference_times.append((t_inference - t_download) * 1000)
        total_times.append((t_inference - t_decode) * 1000)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1} frames...")

    extractor.close()

    print(f"\nResults ({len(total_times)} frames):")
    print(f"  Download time: {np.mean(decode_times):.2f} ms avg")
    print(f"  Inference time: {np.mean(inference_times):.2f} ms avg")
    print(f"  Total time: {np.mean(total_times):.2f} ms avg")
    print(f"  Pipeline FPS: {1000/np.mean(total_times):.1f}")


def test_batch_inference():
    """Test sequential multi-frame inference (batch not supported by model)."""
    print("=" * 60)
    print("Sequential Multi-Frame Inference Test")
    print("=" * 60)

    engine_path = Path("/opt3/ronin/ml_models/yolov8n.engine")
    onnx_path = Path("/opt3/ronin/ml_models/yolov8n.onnx")

    model_path = engine_path if engine_path.exists() else onnx_path

    # Create detector
    detector = TensorRTDetector(
        model_path=model_path,
        precision="fp16",
        warmup_iterations=10,
    )

    # Test processing multiple frames sequentially
    num_frames_list = [1, 4, 8]

    for num_frames in num_frames_list:
        frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(num_frames)]

        # Benchmark
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            for frame in frames:
                detector.detect(frame)
            times.append((time.perf_counter() - t0) * 1000)

        mean_time = np.mean(times)
        print(f"  {num_frames} frames sequential:")
        print(f"    Total time: {mean_time:.2f} ms")
        print(f"    Per frame:  {mean_time/num_frames:.2f} ms")
        print(f"    FPS:        {num_frames * 1000/mean_time:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Test TensorRT inference")
    parser.add_argument("--video", type=str, help="Path to test video")
    parser.add_argument("--frames", type=int, default=300, help="Max frames")
    parser.add_argument("--skip-basic", action="store_true", help="Skip basic test")
    args = parser.parse_args()

    if not args.skip_basic:
        test_basic_inference()
        print()
        test_batch_inference()
        print()

    if args.video:
        test_video_inference(args.video, max_frames=args.frames)
        print()
        test_nvdec_pipeline(args.video, max_frames=min(100, args.frames))
        print()

    print("=" * 60)
    print("All TensorRT tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
