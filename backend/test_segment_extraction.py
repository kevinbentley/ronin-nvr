#!/usr/bin/env python3
"""Compare frame extraction methods: OpenCV vs FFmpeg.

This script compares the person detection scores when frames are extracted:
1. OpenCV (cv2.VideoCapture) - what test_detection_video.py uses
2. FFmpeg (subprocess) - what live_detection_worker.py uses

This helps diagnose why the same video shows different detection results.
"""

import argparse
import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_frames_opencv(video_path: Path, start_sec: float, fps: float, duration: float, scale_height: int = 720) -> list[np.ndarray]:
    """Extract frames using OpenCV (same method as test_detection_video.py)."""
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return frames

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * video_fps)
    end_frame = int((start_sec + duration) * video_fps)
    frame_interval = int(video_fps / fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_num = start_frame
    while frame_num < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_num - start_frame) % frame_interval == 0:
            # Scale if needed
            if scale_height > 0 and frame.shape[0] != scale_height:
                scale_factor = scale_height / frame.shape[0]
                new_width = int(frame.shape[1] * scale_factor)
                frame = cv2.resize(frame, (new_width, scale_height))
            frames.append(frame)

        frame_num += 1

    cap.release()
    return frames


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

    dims = result.stdout.strip().split(",")
    if len(dims) != 2:
        logger.error(f"Failed to parse dimensions: {result.stdout}")
        return frames

    src_width, src_height = int(dims[0]), int(dims[1])

    # Calculate scaled dimensions
    scale_ratio = scale_height / src_height
    scaled_width = int(src_width * scale_ratio)
    scaled_width = scaled_width + (scaled_width % 2)  # Round to even
    scaled_height = scale_height

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

    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        frame_data = stdout[start:end]
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
            (scaled_height, scaled_width, 3)
        )
        frames.append(frame)

    return frames


def analyze_frames(frames: list[np.ndarray], detector, method: str) -> tuple[list[float], list[float]]:
    """Analyze frames and return person/car scores."""
    person_scores = []
    car_scores = []

    for i, frame in enumerate(frames):
        # Run inference to get raw scores
        input_data, scale_x, scale_y = detector.preprocess(frame)
        outputs = detector._run_inference(input_data)
        raw_output = outputs[0]
        class_scores = raw_output[4:]

        person_max = float(class_scores[0].max())
        car_max = float(class_scores[2].max()) if class_scores.shape[0] > 2 else 0

        person_scores.append(person_max)
        car_scores.append(car_max)

        if i < 10 or person_max > 0.1:
            logger.info(f"{method} frame {i}: person_max={person_max:.3f}, car_max={car_max:.3f}, shape={frame.shape}")

    return person_scores, car_scores


def main():
    parser = argparse.ArgumentParser(description="Compare frame extraction methods")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--start", type=float, default=443, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=30, help="Duration in seconds")
    parser.add_argument("--fps", type=float, default=3.0, help="Frames per second")
    parser.add_argument("--model", default="/data/storage/.ml/models/yolov8l.onnx", help="Model path")
    parser.add_argument("--scale-height", type=int, default=720, help="Scale height")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return 1

    # Initialize detector
    logger.info(f"Loading model: {args.model}")
    from app.services.ml.tensorrt_inference import TensorRTDetector

    detector = TensorRTDetector(
        model_path=Path(args.model),
        confidence_threshold=0.05,
        class_thresholds={"person": 0.05},
        device_id=0,
        warmup_iterations=5,
    )
    logger.info("Detector initialized")

    # Extract frames using both methods
    logger.info("=" * 60)
    logger.info("OPENCV EXTRACTION")
    logger.info("=" * 60)
    opencv_frames = extract_frames_opencv(
        video_path, args.start, args.fps, args.duration, args.scale_height
    )
    logger.info(f"Extracted {len(opencv_frames)} frames with OpenCV")
    opencv_person, opencv_car = analyze_frames(opencv_frames[:20], detector, "OpenCV")  # First 20 frames

    logger.info("=" * 60)
    logger.info("FFMPEG EXTRACTION (same video, no start offset)")
    logger.info("=" * 60)

    # For ffmpeg, we need to specify start time differently
    # Create a temp clip first
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp_path = tmp.name

    # Extract the relevant portion with ffmpeg
    extract_cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-ss", str(args.start),
        "-i", str(video_path),
        "-t", str(args.duration),
        "-c", "copy",
        tmp_path,
    ]
    subprocess.run(extract_cmd, check=True)

    ffmpeg_frames = extract_frames_ffmpeg(Path(tmp_path), args.fps, args.scale_height)
    logger.info(f"Extracted {len(ffmpeg_frames)} frames with FFmpeg")
    ffmpeg_person, ffmpeg_car = analyze_frames(ffmpeg_frames[:20], detector, "FFmpeg")  # First 20 frames

    # Cleanup
    Path(tmp_path).unlink()

    # Summary
    logger.info("=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)

    logger.info(f"OpenCV frames: {len(opencv_frames)}, FFmpeg frames: {len(ffmpeg_frames)}")

    if opencv_person:
        logger.info(f"OpenCV person_max: min={min(opencv_person):.3f}, max={max(opencv_person):.3f}, mean={np.mean(opencv_person):.3f}")
    if ffmpeg_person:
        logger.info(f"FFmpeg person_max: min={min(ffmpeg_person):.3f}, max={max(ffmpeg_person):.3f}, mean={np.mean(ffmpeg_person):.3f}")

    # Visual comparison of first frame from each
    if opencv_frames and ffmpeg_frames:
        cv2.imwrite("opencv_frame_0.jpg", opencv_frames[0])
        cv2.imwrite("ffmpeg_frame_0.jpg", ffmpeg_frames[0])
        logger.info("Saved opencv_frame_0.jpg and ffmpeg_frame_0.jpg for visual comparison")

        # Check if frames are different
        if opencv_frames[0].shape == ffmpeg_frames[0].shape:
            diff = cv2.absdiff(opencv_frames[0], ffmpeg_frames[0])
            diff_mean = diff.mean()
            logger.info(f"Frame difference (mean pixel diff): {diff_mean:.2f}")
        else:
            logger.info(f"Frame shapes differ: OpenCV={opencv_frames[0].shape}, FFmpeg={ffmpeg_frames[0].shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
