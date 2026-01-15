#!/usr/bin/env python3
"""Test detection using FFmpeg frame extraction (same as live_detection_worker)."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_frames_ffmpeg(video_path: Path, fps: float, scale_height: int = 720) -> list[np.ndarray]:
    """Extract frames using FFmpeg (same method as live_detection_worker.py)."""
    frames = []

    # Get video dimensions
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path),
    ]

    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFprobe failed: {result.stderr}")
        return frames

    # Parse first line only (ts files can have multiple lines)
    lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
    if not lines:
        logger.error(f"No dimensions found: {result.stdout}")
        return frames

    dims = lines[0].split(",")
    if len(dims) != 2:
        logger.error(f"Failed to parse dimensions: {lines[0]}")
        return frames

    src_width, src_height = int(dims[0]), int(dims[1])
    logger.info(f"Source dimensions: {src_width}x{src_height}")

    # Calculate scaled dimensions
    scale_ratio = scale_height / src_height
    scaled_width = int(src_width * scale_ratio)
    scaled_width = scaled_width + (scaled_width % 2)  # Round to even
    scaled_height = scale_height
    logger.info(f"Scaled dimensions: {scaled_width}x{scaled_height}")

    scale_filter = f"scale=-2:{scale_height}"

    # Extract frames with FFmpeg
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"{scale_filter},fps={fps}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        logger.error(f"FFmpeg failed: {result.stderr.decode()[:200]}")
        return frames

    stdout = result.stdout
    if not stdout:
        logger.error("FFmpeg returned no data")
        return frames

    frame_size = scaled_width * scaled_height * 3
    num_frames = len(stdout) // frame_size
    logger.info(f"Raw data: {len(stdout)} bytes, frame_size: {frame_size}, num_frames: {num_frames}")

    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        frame_data = stdout[start:end]
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
            (scaled_height, scaled_width, 3)
        )
        frames.append(frame)

    return frames


def main():
    parser = argparse.ArgumentParser(description="Test detection with FFmpeg extraction")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--fps", type=float, default=3.0, help="Frames per second")
    parser.add_argument("--model", default="/data/storage/.ml/models/yolo11l_dynamic.onnx", help="Model path")
    parser.add_argument("--scale-height", type=int, default=720, help="Scale height")
    parser.add_argument("--threshold", type=float, default=0.05, help="Detection threshold")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return 1

    logger.info(f"Testing: {video_path.name}")

    # Extract frames using FFmpeg
    frames = extract_frames_ffmpeg(video_path, args.fps, args.scale_height)
    if not frames:
        logger.error("No frames extracted")
        return 1

    logger.info(f"Extracted {len(frames)} frames")

    # Initialize detector
    logger.info(f"Loading model: {args.model}")
    from app.services.ml.tensorrt_inference import TensorRTDetector

    detector = TensorRTDetector(
        model_path=Path(args.model),
        confidence_threshold=args.threshold,
        class_thresholds={"person": args.threshold},
        device_id=0,
        warmup_iterations=5,
    )
    logger.info("Detector initialized")

    # Analyze frames
    person_scores = []
    for i, frame in enumerate(frames):
        input_data, scale_x, scale_y = detector.preprocess(frame)
        outputs = detector._run_inference(input_data)
        raw_output = outputs[0]
        class_scores = raw_output[4:]
        person_max = float(class_scores[0].max())
        car_max = float(class_scores[2].max()) if class_scores.shape[0] > 2 else 0
        person_scores.append(person_max)

        if person_max > 0.1:
            logger.info(f"Frame {i}: person_max={person_max:.3f}, car_max={car_max:.3f}")

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    if person_scores:
        scores = np.array(person_scores)
        logger.info(f"Frames: {len(scores)}")
        logger.info(f"Person max: {scores.max():.4f}")
        logger.info(f"Person mean: {scores.mean():.4f}")
        logger.info(f"Person median: {np.median(scores):.4f}")

        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            count = (scores >= thresh).sum()
            pct = 100 * count / len(scores)
            logger.info(f">={thresh}: {count}/{len(scores)} ({pct:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
