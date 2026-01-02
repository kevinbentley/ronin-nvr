#!/usr/bin/env python3
"""Run object detection on a video file.

This script processes a video file and detects objects (people, cars, trucks, etc.)
using YOLO. It can output results to console, JSON, or generate an annotated video.

Usage:
    # Basic usage - detect objects and print results
    python scripts/detect_video.py /path/to/video.mp4

    # Specify detection classes
    python scripts/detect_video.py video.mp4 --classes person,car,dog

    # Generate annotated video with bounding boxes
    python scripts/detect_video.py video.mp4 --output annotated.mp4

    # Save results to JSON
    python scripts/detect_video.py video.mp4 --json results.json

    # Process every Nth frame (faster processing)
    python scripts/detect_video.py video.mp4 --sample-rate 30

    # Lower confidence threshold
    python scripts/detect_video.py video.mp4 --confidence 0.4
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.services.ml.detection_service import DetectionResult, DetectionService
from app.services.ml.model_manager import ModelManager


def draw_detections(
    frame: np.ndarray, detections: list[DetectionResult]
) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    h, w = frame.shape[:2]
    annotated = frame.copy()

    colors = {
        "person": (0, 255, 0),  # Green
        "car": (255, 0, 0),  # Blue
        "truck": (255, 0, 0),
        "bus": (255, 0, 0),
        "motorcycle": (255, 165, 0),  # Orange
        "bicycle": (255, 165, 0),
        "dog": (0, 165, 255),
        "cat": (0, 165, 255),
    }

    for det in detections:
        # Convert normalized coords to pixels
        x1 = int(det.x * w)
        y1 = int(det.y * h)
        x2 = int((det.x + det.width) * w)
        y2 = int((det.y + det.height) * h)

        # Draw box
        color = colors.get(det.class_name, (0, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{det.class_name} {det.confidence:.0%}"
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1
        )
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return annotated


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def process_video(
    video_path: Path,
    model_name: str,
    confidence: float,
    classes: set[str],
    sample_rate: int,
    output_path: Optional[Path],
    json_path: Optional[Path],
    verbose: bool,
) -> dict:
    """Process video and detect objects.

    Args:
        video_path: Path to input video file
        model_name: YOLO model name to use
        confidence: Minimum confidence threshold
        classes: Set of class names to detect (empty = all)
        sample_rate: Process every Nth frame
        output_path: Path for annotated output video (optional)
        json_path: Path for JSON results (optional)
        verbose: Print per-frame results

    Returns:
        Summary dict with detection statistics
    """
    # Initialize ML components
    print(f"Loading model: {model_name}")
    model_manager = ModelManager()
    detector = DetectionService(model_mgr=model_manager)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {video_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {format_time(duration)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Processing every {sample_rate} frame(s)")
    print(f"  Confidence threshold: {confidence:.0%}")
    if classes:
        print(f"  Filtering classes: {', '.join(sorted(classes))}")
    print()

    # Setup output video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps / sample_rate
        writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, height))
        print(f"Writing annotated video to: {output_path}")

    # Process frames
    all_detections: list[dict] = []
    class_counts: dict[str, int] = {}
    frames_with_detections = 0
    frame_num = 0
    processed_frames = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Skip frames based on sample rate
            if frame_num % sample_rate != 0:
                continue

            processed_frames += 1
            timestamp = frame_num / fps if fps > 0 else 0

            # Run detection
            results = detector.detect(
                frame, model_name, confidence_threshold=confidence
            )

            # Filter by class if specified
            if classes:
                results = [r for r in results if r.class_name.lower() in classes]

            # Record results
            if results:
                frames_with_detections += 1

                for det in results:
                    class_counts[det.class_name] = (
                        class_counts.get(det.class_name, 0) + 1
                    )
                    all_detections.append({
                        "frame": frame_num,
                        "timestamp": timestamp,
                        "timestamp_formatted": format_time(timestamp),
                        "class_name": det.class_name,
                        "confidence": det.confidence,
                        "bbox": {
                            "x": det.x,
                            "y": det.y,
                            "width": det.width,
                            "height": det.height,
                        },
                    })

                if verbose:
                    det_str = ", ".join(
                        f"{d.class_name}({d.confidence:.0%})" for d in results
                    )
                    print(f"[{format_time(timestamp)}] Frame {frame_num}: {det_str}")

            # Write annotated frame
            if writer:
                annotated = draw_detections(frame, results)
                writer.write(annotated)

            # Progress update
            if processed_frames % 100 == 0:
                pct = frame_num / total_frames * 100
                elapsed = time.time() - start_time
                fps_proc = processed_frames / elapsed if elapsed > 0 else 0
                print(
                    f"  Progress: {pct:.1f}% ({frame_num}/{total_frames}) "
                    f"- {fps_proc:.1f} fps"
                )

    finally:
        cap.release()
        if writer:
            writer.release()
        model_manager.unload_all()

    # Calculate summary
    elapsed = time.time() - start_time
    summary = {
        "video_path": str(video_path),
        "video_info": {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": duration,
        },
        "processing": {
            "model": model_name,
            "confidence_threshold": confidence,
            "sample_rate": sample_rate,
            "frames_processed": processed_frames,
            "processing_time_seconds": elapsed,
            "processing_fps": processed_frames / elapsed if elapsed > 0 else 0,
        },
        "results": {
            "total_detections": len(all_detections),
            "frames_with_detections": frames_with_detections,
            "class_counts": class_counts,
        },
        "detections": all_detections,
    }

    # Save JSON if requested
    if json_path:
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {json_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run object detection on a video file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Path to video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="YOLO model name (default: from settings)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.6,
        help="Minimum confidence threshold (default: 0.6)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="",
        help="Comma-separated list of classes to detect (default: all)",
    )
    parser.add_argument(
        "--sample-rate",
        "-s",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 = all frames)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for annotated video",
    )
    parser.add_argument(
        "--json",
        "-j",
        type=Path,
        default=None,
        help="Output path for JSON results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-frame detection results",
    )

    args = parser.parse_args()

    # Validate input
    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    # Get model name
    settings = get_settings()
    model_name = args.model or settings.ml_default_model

    # Parse classes
    classes = set()
    if args.classes:
        classes = {c.strip().lower() for c in args.classes.split(",") if c.strip()}

    # Process video
    try:
        summary = process_video(
            video_path=args.video,
            model_name=model_name,
            confidence=args.confidence,
            classes=classes,
            sample_rate=args.sample_rate,
            output_path=args.output,
            json_path=args.json,
            verbose=args.verbose,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("DETECTION SUMMARY")
        print("=" * 60)
        print(f"Processed {summary['processing']['frames_processed']} frames "
              f"in {summary['processing']['processing_time_seconds']:.1f}s "
              f"({summary['processing']['processing_fps']:.1f} fps)")
        print(f"Total detections: {summary['results']['total_detections']}")
        print(f"Frames with detections: {summary['results']['frames_with_detections']}")

        if summary["results"]["class_counts"]:
            print("\nDetections by class:")
            for class_name, count in sorted(
                summary["results"]["class_counts"].items(),
                key=lambda x: -x[1]
            ):
                print(f"  {class_name}: {count}")
        else:
            print("\nNo objects detected.")

        if args.output:
            print(f"\nAnnotated video saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
