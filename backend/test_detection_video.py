#!/usr/bin/env python3
"""Test detection on a video file using the same pipeline as live-detection.

This script processes a video frame-by-frame and reports detection scores,
helping diagnose why person detection might be failing.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test detection on video file")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=30, help="Duration to process in seconds")
    parser.add_argument("--fps", type=float, default=3.0, help="Frames per second to analyze")
    parser.add_argument("--model", default="/data/storage/.ml/models/yolov8l.onnx", help="Model path")
    parser.add_argument("--threshold", type=float, default=0.05, help="Person threshold")
    parser.add_argument("--save-frames", action="store_true", help="Save frames with high person scores")
    parser.add_argument("--scale-height", type=int, default=0, help="Scale frames to this height (0=no scaling)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return 1

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return 1

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    logger.info(f"Video: {video_path.name}")
    logger.info(f"Resolution: {width}x{height}, FPS: {video_fps:.1f}, Duration: {duration:.1f}s")
    logger.info(f"Processing from {args.start}s to {args.start + args.duration}s at {args.fps} FPS")

    # Seek to start position
    start_frame = int(args.start * video_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Import and initialize detector
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

    # Process frames
    frame_interval = int(video_fps / args.fps)
    end_frame = start_frame + int(args.duration * video_fps)

    person_scores = []
    frames_with_detections = 0
    total_analyzed = 0

    output_dir = Path("test_frames")
    if args.save_frames:
        output_dir.mkdir(exist_ok=True)

    frame_num = start_frame
    while frame_num < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_num - start_frame) % frame_interval == 0:
            total_analyzed += 1
            timestamp = frame_num / video_fps

            # Scale frame if requested (simulates live detection resolution)
            if args.scale_height > 0 and frame.shape[0] != args.scale_height:
                scale_factor = args.scale_height / frame.shape[0]
                new_width = int(frame.shape[1] * scale_factor)
                frame = cv2.resize(frame, (new_width, args.scale_height))

            # Run detection
            t0 = time.perf_counter()
            detections = detector.detect(frame, debug=False)
            det_time = (time.perf_counter() - t0) * 1000

            # Get raw scores for analysis
            # Re-run inference just for score analysis
            input_data, scale_x, scale_y = detector.preprocess(frame)
            outputs = detector._run_inference(input_data)
            raw_output = outputs[0]
            class_scores = raw_output[4:]
            person_max = float(class_scores[0].max())
            car_max = float(class_scores[2].max()) if class_scores.shape[0] > 2 else 0

            person_scores.append(person_max)

            # Count person detections
            person_dets = [d for d in detections if d.class_name == "person"]
            if person_dets:
                frames_with_detections += 1

            # Log interesting frames
            if person_max > 0.05 or person_dets:
                det_str = ", ".join(f"{d.class_name}:{d.confidence:.2f}" for d in detections[:5])
                logger.info(
                    f"[{timestamp:6.2f}s] person_max={person_max:.3f}, car_max={car_max:.3f}, "
                    f"dets={len(detections)}: {det_str or 'none'}"
                )

                # Save high-score frames
                if args.save_frames and person_max > 0.1:
                    frame_path = output_dir / f"frame_{timestamp:.2f}s_person{person_max:.3f}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    logger.info(f"  Saved: {frame_path}")
            else:
                # Print progress every 10 frames
                if total_analyzed % 10 == 0:
                    print(f"\r[{timestamp:6.2f}s] person_max={person_max:.3f}, frames analyzed: {total_analyzed}", end="")

        frame_num += 1

    print()  # Newline after progress
    cap.release()

    # Summary statistics
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Frames analyzed: {total_analyzed}")
    logger.info(f"Frames with person detections (>{args.threshold}): {frames_with_detections}")

    if person_scores:
        scores = np.array(person_scores)
        logger.info(f"Person score stats:")
        logger.info(f"  Min:    {scores.min():.4f}")
        logger.info(f"  Max:    {scores.max():.4f}")
        logger.info(f"  Mean:   {scores.mean():.4f}")
        logger.info(f"  Median: {np.median(scores):.4f}")
        logger.info(f"  Std:    {scores.std():.4f}")

        # Percentiles
        for p in [50, 75, 90, 95, 99]:
            logger.info(f"  P{p}:    {np.percentile(scores, p):.4f}")

        # Count by threshold
        for thresh in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            count = (scores >= thresh).sum()
            pct = 100 * count / len(scores)
            logger.info(f"  >={thresh}: {count}/{len(scores)} ({pct:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
