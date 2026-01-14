#!/usr/bin/env python3
"""Test nextgen pipeline for false positives across multiple videos.

This script processes multiple videos through the GPU pipeline and reports
statistics on motion detections, object detections, and potential false positives.

Usage:
    python tools/test_false_positives.py /path/to/video1.mp4 /path/to/video2.mp4
    python tools/test_false_positives.py --daytime  # Test daytime videos
    python tools/test_false_positives.py --nighttime  # Test nighttime videos
    python tools/test_false_positives.py --all  # Test both
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.gpu_orchestrator import GPUOrchestrator, GPUPipelineConfig
from app.services.ml.object_fsm import EventType


@dataclass
class VideoStats:
    """Statistics for a single video."""

    path: str
    total_frames: int = 0
    motion_frames: int = 0
    detection_frames: int = 0
    total_detections: int = 0
    detections_by_class: dict = field(default_factory=dict)
    confirmed_tracks: int = 0
    events: list = field(default_factory=list)
    processing_time_ms: float = 0.0
    false_positive_frames: list = field(default_factory=list)

    @property
    def motion_rate(self) -> float:
        return (self.motion_frames / self.total_frames * 100) if self.total_frames > 0 else 0

    @property
    def detection_rate(self) -> float:
        return (self.detection_frames / self.total_frames * 100) if self.total_frames > 0 else 0

    @property
    def fps(self) -> float:
        return (self.total_frames / (self.processing_time_ms / 1000)) if self.processing_time_ms > 0 else 0


@dataclass
class TestConfig:
    """Test configuration."""

    # Model
    model_path: str = "/opt3/ronin/ml_models/yolov8n_dynamic.onnx"

    # Motion detection
    motion_min_percent: float = 0.3
    motion_var_threshold: float = 16.0

    # Detection - default threshold for vehicles
    detection_confidence: float = 0.65
    # Per-class thresholds - lower for people since harder to detect at night
    class_thresholds: dict = field(default_factory=lambda: {
        "person": 0.45,
        "dog": 0.45,
        "cat": 0.45,
    })

    # Tracking
    track_min_hits: int = 5
    track_min_displacement: float = 0.02

    # Test settings
    sample_fps: float = 3.0  # Sample frames at this rate
    max_frames: int = 0  # 0 = no limit
    save_detections: bool = True
    output_dir: str = "test_output"


def find_videos(storage_root: str = "/opt3/ronin/storage") -> dict:
    """Find test videos categorized by time of day."""
    storage = Path(storage_root)
    videos = {"daytime": [], "nighttime": []}

    for mp4 in storage.rglob("*.mp4"):
        # Extract hour from filename (HH-MM-SS.mp4)
        try:
            hour = int(mp4.stem.split("-")[0])
            if 7 <= hour <= 18:
                videos["daytime"].append(str(mp4))
            else:
                videos["nighttime"].append(str(mp4))
        except (ValueError, IndexError):
            continue

    # Sort and limit
    videos["daytime"] = sorted(videos["daytime"])[:10]
    videos["nighttime"] = sorted(videos["nighttime"])[:10]

    return videos


def process_video(
    video_path: str,
    orchestrator: GPUOrchestrator,
    config: TestConfig,
    camera_id: int = 1,
) -> VideoStats:
    """Process a video and collect statistics."""

    stats = VideoStats(path=video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  ERROR: Could not open {video_path}")
        return stats

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Calculate frame skip for target sample rate
    frame_skip = max(1, int(video_fps / config.sample_fps))

    # Create output dir for this video if saving
    output_dir = None
    if config.save_detections:
        video_name = Path(video_path).stem
        output_dir = Path(config.output_dir) / video_name
        output_dir.mkdir(parents=True, exist_ok=True)

    # Reset orchestrator state for this video
    orchestrator.reset_camera(camera_id)

    frame_idx = 0
    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to achieve target FPS
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # Check max frames limit
        if config.max_frames > 0 and stats.total_frames >= config.max_frames:
            break

        stats.total_frames += 1
        timestamp = frame_idx / video_fps

        # Scale to 720p for processing
        h, w = frame.shape[:2]
        if h > 720:
            scale = 720 / h
            frame = cv2.resize(frame, (int(w * scale), 720))

        # Process through pipeline
        result = orchestrator.process(
            camera_id=camera_id,
            frame=frame,
            timestamp=timestamp,
        )

        # Collect stats
        if result.motion_detected:
            stats.motion_frames += 1

        if result.detections:
            stats.detection_frames += 1
            stats.total_detections += len(result.detections)

            for det in result.detections:
                class_name = det.class_name
                stats.detections_by_class[class_name] = stats.detections_by_class.get(class_name, 0) + 1

        # Count confirmed tracks
        confirmed = [t for t in result.tracks if t.hits >= config.track_min_hits]
        if confirmed:
            stats.confirmed_tracks += len(confirmed)

        # Record events
        for event in result.events:
            stats.events.append({
                "frame": frame_idx,
                "time": timestamp,
                "type": event.event_type.value,
                "class": event.class_name,
                "track_id": event.track_id,
            })

        # Save frame if there are confirmed detections (potential false positives to review)
        if output_dir and confirmed:
            # Draw detections
            annotated = frame.copy()
            h, w = annotated.shape[:2]

            for track in confirmed:
                x1 = int(track.x * w)
                y1 = int(track.y * h)
                x2 = int((track.x + track.width) * w)
                y2 = int((track.y + track.height) * h)

                color = (0, 255, 0) if track.class_name == "person" else (255, 0, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                label = f"{track.class_name} {track.confidence:.0%} T{track.track_id}"
                cv2.putText(annotated, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save frame
            frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), annotated)
            stats.false_positive_frames.append(str(frame_path))

        frame_idx += 1

    stats.processing_time_ms = (time.perf_counter() - start_time) * 1000
    cap.release()

    return stats


def print_video_stats(stats: VideoStats) -> None:
    """Print statistics for a video."""
    video_name = Path(stats.path).name

    print(f"\n{'=' * 60}")
    print(f"Video: {video_name}")
    print(f"{'=' * 60}")
    print(f"  Frames processed: {stats.total_frames}")
    print(f"  Processing time: {stats.processing_time_ms:.0f}ms ({stats.fps:.1f} FPS)")
    print(f"  Motion frames: {stats.motion_frames} ({stats.motion_rate:.1f}%)")
    print(f"  Detection frames: {stats.detection_frames} ({stats.detection_rate:.1f}%)")
    print(f"  Total detections: {stats.total_detections}")
    print(f"  Confirmed tracks: {stats.confirmed_tracks}")

    if stats.detections_by_class:
        print(f"  Detections by class:")
        for cls, count in sorted(stats.detections_by_class.items()):
            print(f"    - {cls}: {count}")

    if stats.events:
        print(f"  Events: {len(stats.events)}")
        for event in stats.events[:5]:  # Show first 5
            print(f"    - {event['type']}: {event['class']} (T{event['track_id']}) @ {event['time']:.1f}s")
        if len(stats.events) > 5:
            print(f"    ... and {len(stats.events) - 5} more")

    if stats.false_positive_frames:
        print(f"  Saved frames for review: {len(stats.false_positive_frames)}")


def print_summary(all_stats: list[VideoStats], category: str) -> None:
    """Print summary statistics."""

    if not all_stats:
        print(f"\nNo {category} videos processed.")
        return

    total_frames = sum(s.total_frames for s in all_stats)
    total_motion = sum(s.motion_frames for s in all_stats)
    total_detection_frames = sum(s.detection_frames for s in all_stats)
    total_detections = sum(s.total_detections for s in all_stats)
    total_confirmed = sum(s.confirmed_tracks for s in all_stats)
    total_time = sum(s.processing_time_ms for s in all_stats)

    # Aggregate detections by class
    all_classes = {}
    for s in all_stats:
        for cls, count in s.detections_by_class.items():
            all_classes[cls] = all_classes.get(cls, 0) + count

    print(f"\n{'#' * 60}")
    print(f"SUMMARY: {category.upper()} ({len(all_stats)} videos)")
    print(f"{'#' * 60}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total time: {total_time/1000:.1f}s ({total_frames/(total_time/1000):.1f} FPS)")
    print(f"  Motion rate: {total_motion/total_frames*100:.2f}%")
    print(f"  Detection rate: {total_detection_frames/total_frames*100:.2f}%")
    print(f"  Total detections: {total_detections}")
    print(f"  Confirmed tracks: {total_confirmed}")

    if all_classes:
        print(f"  Detections by class:")
        for cls, count in sorted(all_classes.items(), key=lambda x: -x[1]):
            print(f"    - {cls}: {count}")

    # False positive indicator
    if total_detection_frames > 0 and total_frames > 100:
        fp_rate = total_detection_frames / total_frames * 100
        if fp_rate > 5:
            print(f"\n  WARNING: High detection rate ({fp_rate:.1f}%) may indicate false positives")
            print(f"  Review saved frames in test_output/ directory")


def main():
    parser = argparse.ArgumentParser(
        description="Test nextgen pipeline for false positives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "videos",
        nargs="*",
        help="Video files to process",
    )
    parser.add_argument(
        "--daytime",
        action="store_true",
        help="Test daytime videos (10am-6pm)",
    )
    parser.add_argument(
        "--nighttime",
        action="store_true",
        help="Test nighttime videos (6pm-10am)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test both daytime and nighttime",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Max frames per video (default: 500, 0=unlimited)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="Sample rate in FPS (default: 3.0)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.65,
        help="Detection confidence threshold (default: 0.65)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save detection frames",
    )
    parser.add_argument(
        "--output-dir",
        default="test_output",
        help="Output directory for saved frames",
    )

    args = parser.parse_args()

    # Determine which videos to test
    videos_to_test = {"daytime": [], "nighttime": [], "custom": []}

    if args.videos:
        videos_to_test["custom"] = args.videos
    elif args.all or (args.daytime and args.nighttime):
        found = find_videos()
        videos_to_test["daytime"] = found["daytime"]
        videos_to_test["nighttime"] = found["nighttime"]
    elif args.daytime:
        videos_to_test["daytime"] = find_videos()["daytime"]
    elif args.nighttime:
        videos_to_test["nighttime"] = find_videos()["nighttime"]
    else:
        # Default: test a few of each
        found = find_videos()
        videos_to_test["daytime"] = found["daytime"][:3]
        videos_to_test["nighttime"] = found["nighttime"][:3]

    total_videos = sum(len(v) for v in videos_to_test.values())
    if total_videos == 0:
        print("No videos found to test!")
        print("Usage: python tools/test_false_positives.py --daytime")
        print("       python tools/test_false_positives.py /path/to/video.mp4")
        return 1

    # Create test config
    config = TestConfig(
        detection_confidence=args.confidence,
        sample_fps=args.fps,
        max_frames=args.max_frames,
        save_detections=not args.no_save,
        output_dir=args.output_dir,
    )

    print("=" * 60)
    print("NextGen Pipeline - False Positive Test")
    print("=" * 60)
    print(f"Videos to test: {total_videos}")
    print(f"Sample FPS: {config.sample_fps}")
    print(f"Max frames per video: {config.max_frames or 'unlimited'}")
    print(f"Detection confidence: {config.detection_confidence}")
    print(f"Track min hits: {config.track_min_hits}")
    print(f"Track min displacement: {config.track_min_displacement}")
    print(f"Save detections: {config.save_detections}")

    # Initialize pipeline
    print("\nInitializing GPU pipeline...")
    print(f"Per-class thresholds: {config.class_thresholds}")
    pipeline_config = GPUPipelineConfig(
        model_path=config.model_path,
        motion_min_percent=config.motion_min_percent,
        motion_var_threshold=config.motion_var_threshold,
        detection_confidence=config.detection_confidence,
        class_thresholds=config.class_thresholds,
        track_min_hits=config.track_min_hits,
        track_min_displacement=config.track_min_displacement,
    )

    orchestrator = GPUOrchestrator(
        device_ids=[0],
        config=pipeline_config,
    )
    print("Pipeline ready.")

    # Process videos by category
    all_results = {}

    for category, video_list in videos_to_test.items():
        if not video_list:
            continue

        print(f"\n{'=' * 60}")
        print(f"Testing {category.upper()} videos ({len(video_list)})")
        print("=" * 60)

        category_stats = []

        for i, video_path in enumerate(video_list):
            print(f"\n[{i+1}/{len(video_list)}] Processing {Path(video_path).name}...")

            stats = process_video(
                video_path=video_path,
                orchestrator=orchestrator,
                config=config,
                camera_id=i + 1,  # Use different camera ID to reset state
            )

            category_stats.append(stats)
            print_video_stats(stats)

        all_results[category] = category_stats
        print_summary(category_stats, category)

    # Overall summary
    if len(all_results) > 1:
        all_stats = []
        for stats_list in all_results.values():
            all_stats.extend(stats_list)
        print_summary(all_stats, "ALL VIDEOS")

    print(f"\nTest complete. Review saved frames in: {config.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
