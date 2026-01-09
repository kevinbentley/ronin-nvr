#!/usr/bin/env python3
"""
NextGen Motion Detection Benchmark Pipeline

This tool benchmarks the detection pipeline performance by:
1. Loading test videos from storage
2. Running detection pipelines (baseline vs nextgen)
3. Measuring FPS, latency, GPU memory, CPU usage
4. Generating comparison reports

Usage:
    python backend/tools/benchmark_pipeline.py --video /opt3/ronin/storage/camera_1/2026-01-09/12-00-00.mp4
    python backend/tools/benchmark_pipeline.py --scan-storage --limit 5
    python backend/tools/benchmark_pipeline.py --compare baseline nextgen
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    video_path: str
    pipeline_name: str
    total_frames: int
    processed_frames: int
    total_time_seconds: float
    avg_fps: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    peak_gpu_memory_mb: float
    avg_cpu_percent: float
    detections_count: int
    motion_triggers: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    max_frames: int = 1000
    warmup_frames: int = 50
    target_fps: float = 15.0
    confidence_threshold: float = 0.5
    motion_threshold: float = 0.1
    scale_height: int = 720


class GPUMemoryTracker:
    """Track GPU memory usage during benchmark."""

    def __init__(self):
        self.peak_memory_mb = 0.0
        self.samples: list[float] = []
        self._nvidia_smi_available = self._check_nvidia_smi()

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available."""
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    def sample(self) -> float:
        """Sample current GPU memory usage."""
        if not self._nvidia_smi_available:
            return 0.0

        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            # Sum memory across all GPUs
            memory_mb = sum(float(x.strip()) for x in result.stdout.strip().split("\n"))
            self.samples.append(memory_mb)
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
            return memory_mb
        except Exception:
            return 0.0


class CPUTracker:
    """Track CPU usage during benchmark."""

    def __init__(self):
        self.samples: list[float] = []
        self._psutil_available = self._check_psutil()

    def _check_psutil(self) -> bool:
        try:
            import psutil  # noqa: F401

            return True
        except ImportError:
            return False

    def sample(self) -> float:
        """Sample current CPU usage."""
        if not self._psutil_available:
            return 0.0

        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=None)
            self.samples.append(cpu_percent)
            return cpu_percent
        except Exception:
            return 0.0

    def average(self) -> float:
        """Get average CPU usage."""
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


class BaselinePipeline:
    """
    Baseline detection pipeline (current implementation).
    Uses frame differencing + ONNX YOLO inference.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.prev_frame: Optional[np.ndarray] = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load ONNX model for inference."""
        try:
            import onnxruntime as ort

            # Try to find a YOLO model
            model_paths = [
                Path("/opt3/ronin/ml_models/yolov8n.onnx"),
                Path("/opt3/ronin/ml_models/yolov8l.onnx"),
                Path(os.environ.get("ML_MODELS_PATH", "")) / "yolov8n.onnx",
            ]

            for path in model_paths:
                if path.exists():
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    self.model = ort.InferenceSession(str(path), providers=providers)
                    print(f"Loaded baseline model: {path}")
                    return

            print("Warning: No YOLO model found, detection will be skipped")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")

    def check_motion(self, frame: np.ndarray) -> tuple[bool, float]:
        """Simple frame differencing motion detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0.0

        frame_diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        motion_percent = (np.count_nonzero(thresh) / thresh.size) * 100
        self.prev_frame = gray

        return motion_percent > self.config.motion_threshold, motion_percent

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run YOLO detection on frame."""
        if self.model is None:
            return []

        # Preprocess
        input_size = 640
        resized = cv2.resize(frame, (input_size, input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = normalized.transpose(2, 0, 1)
        batched = np.expand_dims(transposed, axis=0)

        # Inference
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: batched})

        # Simple postprocessing (count detections above threshold)
        detections = []
        if len(outputs) > 0:
            predictions = outputs[0]
            if predictions.shape[-1] > 4:  # Has class scores
                scores = predictions[0, :, 4:].max(axis=-1)
                count = np.sum(scores > self.config.confidence_threshold)
                for i in range(int(count)):
                    detections.append({"class": "object", "confidence": float(scores[i])})

        return detections

    def process_frame(self, frame: np.ndarray) -> tuple[bool, list[dict]]:
        """Process single frame through baseline pipeline."""
        # Scale frame
        height = frame.shape[0]
        if height > self.config.scale_height:
            scale = self.config.scale_height / height
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # Check motion
        has_motion, _ = self.check_motion(frame)

        # Run detection if motion
        detections = []
        if has_motion:
            detections = self.detect(frame)

        return has_motion, detections


class NextGenPipeline:
    """
    NextGen detection pipeline (new implementation).
    Uses GPU MOG2 + TensorRT inference + ByteTrack.

    Note: This is a placeholder that will be implemented incrementally.
    Currently falls back to baseline with GPU optimizations where available.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.gpu_available = self._check_gpu()
        self.bg_subtractor = None
        self.model = None
        self._initialize()

    def _check_gpu(self) -> bool:
        """Check if CUDA is available for OpenCV."""
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            return count > 0
        except Exception:
            return False

    def _initialize(self) -> None:
        """Initialize GPU components."""
        if self.gpu_available:
            try:
                # GPU MOG2
                self.bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
                    history=500,
                    varThreshold=16.0,
                    detectShadows=True,
                )
                print("Initialized GPU MOG2 background subtractor")
            except Exception as e:
                print(f"Warning: Could not initialize GPU MOG2: {e}")
                self.gpu_available = False

        # Try TensorRT, fall back to ONNX
        self._load_model()

    def _load_model(self) -> None:
        """Load TensorRT or ONNX model."""
        # TODO: Implement TensorRT loading
        # For now, fall back to ONNX
        try:
            import onnxruntime as ort

            model_paths = [
                Path("/opt3/ronin/ml_models/yolov8s.onnx"),
                Path("/opt3/ronin/ml_models/yolov8n.onnx"),
                Path(os.environ.get("ML_MODELS_PATH", "")) / "yolov8s.onnx",
            ]

            for path in model_paths:
                if path.exists():
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    self.model = ort.InferenceSession(str(path), providers=providers)
                    print(f"Loaded nextgen model: {path}")
                    return

            print("Warning: No model found for nextgen pipeline")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")

    def check_motion(self, frame: np.ndarray) -> tuple[bool, float]:
        """GPU MOG2 motion detection."""
        if not self.gpu_available or self.bg_subtractor is None:
            # Fallback to CPU
            return self._cpu_motion_check(frame)

        try:
            # Upload to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            # Convert to grayscale on GPU
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

            # Apply MOG2
            gpu_mask = self.bg_subtractor.apply(gpu_gray, learningRate=-1)

            # Download mask
            mask = gpu_mask.download()

            # Calculate motion percentage
            motion_percent = (np.count_nonzero(mask) / mask.size) * 100

            return motion_percent > self.config.motion_threshold, motion_percent

        except Exception as e:
            print(f"GPU motion check failed: {e}, falling back to CPU")
            return self._cpu_motion_check(frame)

    def _cpu_motion_check(self, frame: np.ndarray) -> tuple[bool, float]:
        """Fallback CPU motion detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Simple threshold-based detection as fallback
        motion_percent = 0.5  # Placeholder
        return True, motion_percent

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run detection on frame."""
        if self.model is None:
            return []

        # TODO: Implement TensorRT batched inference
        # For now, use ONNX with GPU
        input_size = 640
        resized = cv2.resize(frame, (input_size, input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = normalized.transpose(2, 0, 1)
        batched = np.expand_dims(transposed, axis=0)

        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: batched})

        detections = []
        if len(outputs) > 0:
            predictions = outputs[0]
            if predictions.shape[-1] > 4:
                scores = predictions[0, :, 4:].max(axis=-1)
                count = np.sum(scores > self.config.confidence_threshold)
                for i in range(int(count)):
                    detections.append({"class": "object", "confidence": float(scores[i])})

        return detections

    def process_frame(self, frame: np.ndarray) -> tuple[bool, list[dict]]:
        """Process single frame through nextgen pipeline."""
        # Scale frame
        height = frame.shape[0]
        if height > self.config.scale_height:
            scale = self.config.scale_height / height
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # Check motion with GPU MOG2
        has_motion, _ = self.check_motion(frame)

        # Run detection if motion
        detections = []
        if has_motion:
            detections = self.detect(frame)

        return has_motion, detections


def run_benchmark(
    video_path: Path,
    pipeline_name: str,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Run benchmark on a single video."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {pipeline_name}")
    print(f"Video: {video_path}")
    print(f"{'='*60}")

    # Initialize pipeline
    if pipeline_name == "baseline":
        pipeline = BaselinePipeline(config)
    else:
        pipeline = NextGenPipeline(config)

    # Initialize trackers
    gpu_tracker = GPUMemoryTracker()
    cpu_tracker = CPUTracker()

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Video info: {total_frames} frames @ {fps:.1f} FPS")

    # Run benchmark
    latencies: list[float] = []
    detections_count = 0
    motion_triggers = 0
    processed_frames = 0

    start_time = time.perf_counter()

    for frame_idx in range(min(total_frames, config.max_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Skip warmup frames for timing
        if frame_idx < config.warmup_frames:
            pipeline.process_frame(frame)
            continue

        # Time this frame
        frame_start = time.perf_counter()
        has_motion, detections = pipeline.process_frame(frame)
        frame_end = time.perf_counter()

        latency_ms = (frame_end - frame_start) * 1000
        latencies.append(latency_ms)

        if has_motion:
            motion_triggers += 1
        detections_count += len(detections)
        processed_frames += 1

        # Sample resources periodically
        if frame_idx % 50 == 0:
            gpu_tracker.sample()
            cpu_tracker.sample()

            # Progress update
            elapsed = time.perf_counter() - start_time
            current_fps = processed_frames / elapsed if elapsed > 0 else 0
            print(
                f"  Frame {frame_idx}/{min(total_frames, config.max_frames)} "
                f"| FPS: {current_fps:.1f} "
                f"| Latency: {latency_ms:.1f}ms "
                f"| Motion: {motion_triggers}"
            )

    end_time = time.perf_counter()
    cap.release()

    # Calculate statistics
    total_time = end_time - start_time
    avg_fps = processed_frames / total_time if total_time > 0 else 0

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    result = BenchmarkResult(
        video_path=str(video_path),
        pipeline_name=pipeline_name,
        total_frames=total_frames,
        processed_frames=processed_frames,
        total_time_seconds=total_time,
        avg_fps=avg_fps,
        avg_latency_ms=sum(latencies) / n if n > 0 else 0,
        p50_latency_ms=latencies_sorted[n // 2] if n > 0 else 0,
        p95_latency_ms=latencies_sorted[int(n * 0.95)] if n > 0 else 0,
        p99_latency_ms=latencies_sorted[int(n * 0.99)] if n > 0 else 0,
        peak_gpu_memory_mb=gpu_tracker.peak_memory_mb,
        avg_cpu_percent=cpu_tracker.average(),
        detections_count=detections_count,
        motion_triggers=motion_triggers,
    )

    print(f"\n{'-'*40}")
    print(f"Results for {pipeline_name}:")
    print(f"  Processed: {result.processed_frames} frames")
    print(f"  Avg FPS: {result.avg_fps:.2f}")
    print(f"  Avg Latency: {result.avg_latency_ms:.2f}ms")
    print(f"  P95 Latency: {result.p95_latency_ms:.2f}ms")
    print(f"  Peak GPU Memory: {result.peak_gpu_memory_mb:.0f}MB")
    print(f"  Detections: {result.detections_count}")
    print(f"  Motion Triggers: {result.motion_triggers}")

    return result


def find_test_videos(storage_root: Path, limit: int = 5) -> list[Path]:
    """Find test videos in storage directory."""
    videos: list[Path] = []

    if not storage_root.exists():
        print(f"Storage root not found: {storage_root}")
        return videos

    # Look for MP4 files in camera directories
    for mp4_file in storage_root.rglob("*.mp4"):
        # Skip very small files
        if mp4_file.stat().st_size > 1_000_000:  # > 1MB
            videos.append(mp4_file)
            if len(videos) >= limit:
                break

    return sorted(videos)[:limit]


def save_results(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save benchmark results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [r.to_dict() for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark detection pipeline performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark specific video
  python benchmark_pipeline.py --video /opt3/ronin/storage/cam1/video.mp4

  # Scan storage and benchmark found videos
  python benchmark_pipeline.py --scan-storage --limit 3

  # Compare baseline vs nextgen
  python benchmark_pipeline.py --video path/to/video.mp4 --compare
        """,
    )

    parser.add_argument("--video", type=Path, help="Path to video file to benchmark")
    parser.add_argument(
        "--scan-storage",
        action="store_true",
        help="Scan storage directory for test videos",
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=Path(os.environ.get("STORAGE_ROOT", "/opt3/ronin/storage")),
        help="Root directory for video storage",
    )
    parser.add_argument("--limit", type=int, default=5, help="Max videos to benchmark")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare baseline and nextgen pipelines",
    )
    parser.add_argument(
        "--pipeline",
        choices=["baseline", "nextgen"],
        default="nextgen",
        help="Pipeline to benchmark (default: nextgen)",
    )
    parser.add_argument("--max-frames", type=int, default=500, help="Max frames per video")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/results.json"),
        help="Output path for results",
    )

    args = parser.parse_args()

    # Build config
    config = BenchmarkConfig(max_frames=args.max_frames)

    # Find videos
    videos: list[Path] = []
    if args.video:
        if args.video.exists():
            videos = [args.video]
        else:
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)
    elif args.scan_storage:
        videos = find_test_videos(args.storage_root, args.limit)
        if not videos:
            print("No test videos found in storage")
            sys.exit(1)
        print(f"Found {len(videos)} test videos")
    else:
        parser.print_help()
        sys.exit(1)

    # Run benchmarks
    results: list[BenchmarkResult] = []

    for video in videos:
        if args.compare:
            # Run both pipelines
            results.append(run_benchmark(video, "baseline", config))
            results.append(run_benchmark(video, "nextgen", config))
        else:
            results.append(run_benchmark(video, args.pipeline, config))

    # Save results
    save_results(results, args.output)

    # Print comparison summary if comparing
    if args.compare and len(results) >= 2:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)

        baseline_results = [r for r in results if r.pipeline_name == "baseline"]
        nextgen_results = [r for r in results if r.pipeline_name == "nextgen"]

        if baseline_results and nextgen_results:
            baseline_avg_fps = sum(r.avg_fps for r in baseline_results) / len(baseline_results)
            nextgen_avg_fps = sum(r.avg_fps for r in nextgen_results) / len(nextgen_results)

            baseline_avg_latency = sum(r.avg_latency_ms for r in baseline_results) / len(
                baseline_results
            )
            nextgen_avg_latency = sum(r.avg_latency_ms for r in nextgen_results) / len(
                nextgen_results
            )

            fps_improvement = ((nextgen_avg_fps - baseline_avg_fps) / baseline_avg_fps) * 100
            latency_improvement = (
                (baseline_avg_latency - nextgen_avg_latency) / baseline_avg_latency
            ) * 100

            print(f"\nBaseline:  {baseline_avg_fps:.1f} FPS, {baseline_avg_latency:.1f}ms latency")
            print(f"NextGen:   {nextgen_avg_fps:.1f} FPS, {nextgen_avg_latency:.1f}ms latency")
            print(f"\nFPS Improvement: {fps_improvement:+.1f}%")
            print(f"Latency Improvement: {latency_improvement:+.1f}%")


if __name__ == "__main__":
    main()
