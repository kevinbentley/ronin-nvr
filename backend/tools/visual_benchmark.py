#!/usr/bin/env python3
"""Visual benchmark tool for nextgen detection pipeline.

Generates annotated frame captures, motion visualizations, and detailed metrics.

Usage:
    source /opt/venv/bin/activate
    cd /workspace/ronin-nvr/backend

    # Test single video with captures every 30 frames
    python tools/visual_benchmark.py --video /path/to/video.mp4 --capture-interval 30

    # Test multiple videos from a camera directory
    python tools/visual_benchmark.py --video-dir /opt3/ronin/storage/Hangar_East/2025-12-31 --max-videos 5

    # Test with event-only captures (arrivals, departures)
    python tools/visual_benchmark.py --video /path/to/video.mp4 --events-only

    # Full benchmark with all captures
    python tools/visual_benchmark.py --video-dir /opt3/ronin/storage --cameras-sample 4
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.gpu_orchestrator import (
    GPUOrchestrator, GPUPipelineConfig, PipelineResult
)
from app.services.ml.object_fsm import EventType, ObjectState


# COCO class colors for visualization
CLASS_COLORS = {
    "person": (0, 255, 0),       # Green
    "bicycle": (255, 0, 255),    # Magenta
    "car": (255, 0, 0),          # Blue
    "motorcycle": (0, 255, 255), # Yellow
    "airplane": (128, 128, 255),
    "bus": (128, 0, 255),        # Purple
    "train": (255, 128, 0),
    "truck": (255, 128, 0),      # Orange
    "boat": (255, 255, 0),
    "bird": (0, 128, 255),
    "cat": (255, 0, 128),
    "dog": (128, 255, 0),
}
DEFAULT_COLOR = (0, 200, 255)  # Gold


@dataclass
class CaptureInfo:
    """Information about a saved capture."""
    frame_num: int
    video_time_sec: float
    filename: str
    reason: str
    num_tracks: int
    motion_percent: float
    events: list[str]


@dataclass
class VideoResult:
    """Results from processing a single video."""
    video_path: str
    video_name: str

    # Video info
    original_resolution: tuple[int, int] = (0, 0)
    processing_resolution: tuple[int, int] = (0, 0)
    video_fps: float = 0.0
    total_video_frames: int = 0

    # Processing info
    frames_processed: int = 0
    processing_time_sec: float = 0.0
    processing_fps: float = 0.0

    # Component timing (ms)
    motion_times: list[float] = field(default_factory=list)
    detection_times: list[float] = field(default_factory=list)
    tracking_times: list[float] = field(default_factory=list)
    fsm_times: list[float] = field(default_factory=list)

    # Motion stats
    frames_with_motion: int = 0
    motion_percentages: list[float] = field(default_factory=list)

    # Detection stats
    frames_with_detections: int = 0
    total_detections: int = 0
    detections_per_frame: list[int] = field(default_factory=list)
    confidence_values: list[float] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)

    # Tracking stats
    unique_track_ids: set[int] = field(default_factory=set)
    track_durations: dict[int, float] = field(default_factory=dict)

    # Events
    arrival_count: int = 0
    departure_count: int = 0
    state_change_count: int = 0
    loitering_count: int = 0
    events: list[dict] = field(default_factory=list)

    # Captures
    captures: list[CaptureInfo] = field(default_factory=list)


def format_video_time(seconds: float) -> str:
    """Format seconds as MM:SS.mmm"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:06.3f}"


def draw_annotated_frame(
    frame: np.ndarray,
    result: PipelineResult,
    video_time: float,
    frame_num: int,
    motion_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw annotations on a frame.

    Includes:
    - Bounding boxes with class labels and confidence
    - Track IDs
    - Motion overlay (green tint)
    - Info panel with timing and stats
    - Event alerts
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw motion overlay if available
    if motion_mask is not None and result.motion_detected:
        if motion_mask.shape[:2] != (h, w):
            motion_mask = cv2.resize(motion_mask, (w, h))

        # Create semi-transparent green overlay for motion areas
        overlay = np.zeros_like(annotated)
        overlay[:, :, 1] = motion_mask  # Green channel
        annotated = cv2.addWeighted(annotated, 0.85, overlay, 0.15, 0)

    # Draw each tracked object
    for track in result.tracks:
        # Convert normalized coords to pixels
        x1 = int(track.x * w)
        y1 = int(track.y * h)
        x2 = int((track.x + track.width) * w)
        y2 = int((track.y + track.height) * h)

        color = CLASS_COLORS.get(track.class_name, DEFAULT_COLOR)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label with track ID, class, confidence
        label = f"#{track.track_id} {track.class_name} {track.confidence:.2f}"

        # Get label size for background
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw label background
        cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)

        # Draw label text
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Draw velocity vector if moving
        if abs(track.velocity_x) > 0.001 or abs(track.velocity_y) > 0.001:
            cx = int((track.x + track.width / 2) * w)
            cy = int((track.y + track.height / 2) * h)
            vx = int(track.velocity_x * w * 10)  # Scale for visibility
            vy = int(track.velocity_y * h * 10)
            cv2.arrowedLine(annotated, (cx, cy), (cx + vx, cy + vy), color, 2)

    # Draw info panel at top
    panel_h = 90
    cv2.rectangle(annotated, (0, 0), (w, panel_h), (0, 0, 0), -1)

    time_str = format_video_time(video_time)

    lines = [
        f"Frame: {frame_num:6d} | Time: {time_str} | Video: {w}x{h}",
        f"Motion: {result.motion_percent:5.2f}% | Detections: {len(result.detections)} | Tracks: {len(result.tracks)}",
        f"Timing: Motion={result.motion_time_ms:5.1f}ms Det={result.detection_time_ms:5.1f}ms "
        f"Track={result.tracking_time_ms:4.2f}ms FSM={result.fsm_time_ms:4.2f}ms "
        f"Total={result.total_time_ms:5.1f}ms",
    ]

    for i, line in enumerate(lines):
        cv2.putText(
            annotated, line, (10, 22 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )

    # Draw events at bottom in red
    if result.events:
        event_y = h - 10
        for event in reversed(result.events):
            event_text = f"EVENT: {event.event_type.value.upper()} - {event.class_name} #{event.track_id}"
            cv2.putText(
                annotated, event_text, (10, event_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
            )
            event_y -= 30

    return annotated


def create_motion_heatmap(
    motion_mask: np.ndarray,
    motion_percent: float,
    video_time: float,
    frame_num: int,
) -> np.ndarray:
    """Create motion-only heatmap visualization."""
    h, w = motion_mask.shape[:2]

    # Create heatmap colorization
    heatmap = cv2.applyColorMap(motion_mask, cv2.COLORMAP_JET)

    # Add info overlay
    cv2.rectangle(heatmap, (0, 0), (w, 40), (0, 0, 0), -1)
    info = f"Motion Mask | Frame: {frame_num} | Time: {format_video_time(video_time)} | Motion: {motion_percent:.2f}%"
    cv2.putText(
        heatmap, info, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )

    return heatmap


def create_side_by_side(
    original: np.ndarray,
    annotated: np.ndarray,
    motion_vis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create side-by-side comparison image."""
    h, w = original.shape[:2]

    if motion_vis is not None:
        # Resize motion vis to match
        motion_vis = cv2.resize(motion_vis, (w, h))

        # Stack horizontally: original | annotated | motion
        combined = np.hstack([original, annotated, motion_vis])
    else:
        # Stack: original | annotated
        combined = np.hstack([original, annotated])

    return combined


def process_video(
    video_path: str,
    output_dir: Path,
    orchestrator: GPUOrchestrator,
    max_frames: int = 1000,
    capture_interval: int = 30,
    capture_on_events: bool = True,
    capture_on_motion_start: bool = True,
    save_motion_vis: bool = True,
    save_side_by_side: bool = True,
) -> VideoResult:
    """Process video and generate captures and metrics."""
    result = VideoResult(
        video_path=video_path,
        video_name=Path(video_path).name,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return result

    # Get video properties
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    result.original_resolution = (orig_w, orig_h)
    result.video_fps = fps
    result.total_video_frames = total_frames

    print(f"\n{'='*60}")
    print(f"Processing: {result.video_name}")
    print(f"  Resolution: {orig_w}x{orig_h} @ {fps:.1f} fps")
    print(f"  Duration: {total_frames/fps:.1f}s ({total_frames} frames)")
    print(f"  Processing up to {max_frames} frames")
    print(f"{'='*60}")

    # Create output subdirectory
    video_output_dir = output_dir / Path(video_path).stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Unique camera ID for this video
    camera_id = hash(video_path) % 10000 + 1

    frame_count = 0
    start_time = time.perf_counter()
    prev_motion = False
    track_first_seen: dict[int, float] = {}

    while frame_count < max_frames:
        ret, original_frame = cap.read()
        if not ret:
            break

        frame_count += 1
        video_time = (frame_count - 1) / fps

        # Scale to 720p for processing
        proc_frame = original_frame
        if orig_h > 720:
            scale = 720 / orig_h
            proc_frame = cv2.resize(original_frame, None, fx=scale, fy=scale)

        if frame_count == 1:
            result.processing_resolution = (proc_frame.shape[1], proc_frame.shape[0])

        # Process through pipeline
        pipeline_result = orchestrator.process(
            camera_id=camera_id,
            frame=proc_frame,
            timestamp=time.time(),
        )

        # Collect timing metrics
        result.motion_times.append(pipeline_result.motion_time_ms)
        result.detection_times.append(pipeline_result.detection_time_ms)
        result.tracking_times.append(pipeline_result.tracking_time_ms)
        result.fsm_times.append(pipeline_result.fsm_time_ms)

        # Motion metrics
        result.motion_percentages.append(pipeline_result.motion_percent)
        if pipeline_result.motion_detected:
            result.frames_with_motion += 1

        # Detection metrics
        result.detections_per_frame.append(len(pipeline_result.detections))
        if pipeline_result.detections:
            result.frames_with_detections += 1
            result.total_detections += len(pipeline_result.detections)

            for det in pipeline_result.detections:
                result.confidence_values.append(det.confidence)
                result.class_counts[det.class_name] = (
                    result.class_counts.get(det.class_name, 0) + 1
                )

        # Track metrics
        for track in pipeline_result.tracks:
            if track.track_id not in track_first_seen:
                track_first_seen[track.track_id] = video_time
                result.unique_track_ids.add(track.track_id)
            result.track_durations[track.track_id] = (
                video_time - track_first_seen[track.track_id]
            )

        # Event metrics
        event_names = []
        for event in pipeline_result.events:
            event_names.append(f"{event.event_type.value}:{event.class_name}")

            if event.event_type == EventType.ARRIVAL:
                result.arrival_count += 1
            elif event.event_type == EventType.DEPARTURE:
                result.departure_count += 1
            elif event.event_type == EventType.STATE_CHANGE:
                result.state_change_count += 1
            elif event.event_type == EventType.LOITERING:
                result.loitering_count += 1

            result.events.append({
                "frame": frame_count,
                "video_time": video_time,
                "type": event.event_type.value,
                "track_id": event.track_id,
                "class_name": event.class_name,
                "confidence": event.confidence,
                "duration_sec": event.duration_seconds,
            })

        # Determine if we should capture this frame
        should_capture = False
        capture_reason = ""

        if capture_interval > 0 and frame_count % capture_interval == 0:
            should_capture = True
            capture_reason = "interval"

        if capture_on_events and pipeline_result.events:
            should_capture = True
            capture_reason = "event"

        if capture_on_motion_start and pipeline_result.motion_detected and not prev_motion:
            should_capture = True
            capture_reason = "motion_start"

        # Save captures
        if should_capture:
            # Get motion mask
            motion_mask = None
            pipeline = orchestrator.get_pipeline(camera_id)
            if pipeline and hasattr(pipeline, '_motion_gate'):
                motion_mask = pipeline._motion_gate.get_last_mask(camera_id)

            # Generate annotated frame
            annotated = draw_annotated_frame(
                proc_frame, pipeline_result, video_time, frame_count, motion_mask
            )

            # Build filename with timestamp
            mins = int(video_time // 60)
            secs = int(video_time % 60)
            ms = int((video_time % 1) * 1000)
            base_name = f"f{frame_count:06d}_t{mins:02d}m{secs:02d}s{ms:03d}_{capture_reason}"

            # Save annotated frame
            annotated_path = video_output_dir / f"{base_name}_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Save motion visualization
            motion_vis = None
            if save_motion_vis and motion_mask is not None:
                motion_vis = create_motion_heatmap(
                    motion_mask, pipeline_result.motion_percent, video_time, frame_count
                )
                motion_path = video_output_dir / f"{base_name}_motion.jpg"
                cv2.imwrite(str(motion_path), motion_vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Save side-by-side comparison
            if save_side_by_side:
                combined = create_side_by_side(proc_frame, annotated, motion_vis)
                combined_path = video_output_dir / f"{base_name}_combined.jpg"
                cv2.imwrite(str(combined_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Record capture info
            result.captures.append(CaptureInfo(
                frame_num=frame_count,
                video_time_sec=video_time,
                filename=f"{base_name}_annotated.jpg",
                reason=capture_reason,
                num_tracks=len(pipeline_result.tracks),
                motion_percent=pipeline_result.motion_percent,
                events=event_names,
            ))

        prev_motion = pipeline_result.motion_detected

        # Progress update every 100 frames
        if frame_count % 100 == 0:
            elapsed = time.perf_counter() - start_time
            current_fps = frame_count / elapsed
            print(f"  Frame {frame_count:5d}/{min(max_frames, total_frames)}: "
                  f"{current_fps:5.1f} FPS | motion={result.frames_with_motion:4d} | "
                  f"tracks={len(result.unique_track_ids):3d} | events={len(result.events):3d}")

    cap.release()

    # Finalize metrics
    elapsed = time.perf_counter() - start_time
    result.frames_processed = frame_count
    result.processing_time_sec = elapsed
    result.processing_fps = frame_count / elapsed if elapsed > 0 else 0

    # Reset camera state for next video
    orchestrator.reset_camera(camera_id)

    print(f"\n  Completed: {frame_count} frames in {elapsed:.1f}s ({result.processing_fps:.1f} FPS)")
    print(f"  Motion frames: {result.frames_with_motion} ({100*result.frames_with_motion/frame_count:.1f}%)")
    print(f"  Detection frames: {result.frames_with_detections}")
    print(f"  Unique tracks: {len(result.unique_track_ids)}")
    print(f"  Events: {result.arrival_count} arrivals, {result.departure_count} departures")
    print(f"  Captures saved: {len(result.captures)}")

    return result


def compute_statistics(values: list[float]) -> dict:
    """Compute statistics for a list of values."""
    if not values:
        return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}

    arr = np.array(values)
    return {
        "count": len(values),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def generate_report(results: list[VideoResult], output_dir: Path) -> dict:
    """Generate comprehensive JSON report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {},
        "videos": [],
    }

    # Per-video details
    for r in results:
        video_report = {
            "name": r.video_name,
            "path": r.video_path,
            "video_info": {
                "original_resolution": f"{r.original_resolution[0]}x{r.original_resolution[1]}",
                "processing_resolution": f"{r.processing_resolution[0]}x{r.processing_resolution[1]}",
                "fps": r.video_fps,
                "total_frames": r.total_video_frames,
                "duration_sec": r.total_video_frames / r.video_fps if r.video_fps > 0 else 0,
            },
            "processing": {
                "frames_processed": r.frames_processed,
                "processing_time_sec": r.processing_time_sec,
                "processing_fps": r.processing_fps,
            },
            "timing_ms": {
                "motion": compute_statistics(r.motion_times),
                "detection": compute_statistics(r.detection_times),
                "tracking": compute_statistics(r.tracking_times),
                "fsm": compute_statistics(r.fsm_times),
            },
            "motion": {
                "frames_with_motion": r.frames_with_motion,
                "motion_rate": r.frames_with_motion / max(r.frames_processed, 1),
                "motion_percent_stats": compute_statistics(r.motion_percentages),
            },
            "detection": {
                "frames_with_detections": r.frames_with_detections,
                "detection_rate": r.frames_with_detections / max(r.frames_processed, 1),
                "total_detections": r.total_detections,
                "detections_per_frame": compute_statistics([float(x) for x in r.detections_per_frame]),
                "confidence_stats": compute_statistics(r.confidence_values),
                "class_distribution": r.class_counts,
            },
            "tracking": {
                "unique_tracks": len(r.unique_track_ids),
                "track_ids": sorted(r.unique_track_ids),
                "track_duration_stats": compute_statistics(list(r.track_durations.values())),
            },
            "events": {
                "arrivals": r.arrival_count,
                "departures": r.departure_count,
                "state_changes": r.state_change_count,
                "loitering": r.loitering_count,
                "total": len(r.events),
                "details": r.events,
            },
            "captures": [
                {
                    "frame": c.frame_num,
                    "time_sec": c.video_time_sec,
                    "time_str": format_video_time(c.video_time_sec),
                    "filename": c.filename,
                    "reason": c.reason,
                    "tracks": c.num_tracks,
                    "motion_pct": c.motion_percent,
                    "events": c.events,
                }
                for c in r.captures
            ],
        }
        report["videos"].append(video_report)

    # Aggregate summary
    if results:
        total_frames = sum(r.frames_processed for r in results)
        total_time = sum(r.processing_time_sec for r in results)
        total_motion_frames = sum(r.frames_with_motion for r in results)
        total_detections = sum(r.total_detections for r in results)
        total_tracks = sum(len(r.unique_track_ids) for r in results)
        total_arrivals = sum(r.arrival_count for r in results)
        total_departures = sum(r.departure_count for r in results)
        total_captures = sum(len(r.captures) for r in results)

        all_motion_times = [t for r in results for t in r.motion_times]
        all_detection_times = [t for r in results for t in r.detection_times]
        all_tracking_times = [t for r in results for t in r.tracking_times]
        all_fsm_times = [t for r in results for t in r.fsm_times]
        all_confidences = [c for r in results for c in r.confidence_values]

        # Combine class counts
        combined_classes: dict[str, int] = {}
        for r in results:
            for cls, cnt in r.class_counts.items():
                combined_classes[cls] = combined_classes.get(cls, 0) + cnt

        report["summary"] = {
            "videos_processed": len(results),
            "total_frames": total_frames,
            "total_time_sec": total_time,
            "overall_fps": total_frames / total_time if total_time > 0 else 0,
            "motion_rate": total_motion_frames / total_frames if total_frames > 0 else 0,
            "total_detections": total_detections,
            "total_unique_tracks": total_tracks,
            "total_arrivals": total_arrivals,
            "total_departures": total_departures,
            "total_captures": total_captures,
            "timing_ms": {
                "motion": compute_statistics(all_motion_times),
                "detection": compute_statistics(all_detection_times),
                "tracking": compute_statistics(all_tracking_times),
                "fsm": compute_statistics(all_fsm_times),
            },
            "confidence_stats": compute_statistics(all_confidences),
            "class_distribution": combined_classes,
        }

    # Save JSON report
    report_path = output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Save human-readable summary
    summary_path = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_path, 'w') as f:
        f.write("NextGen Detection Pipeline - Visual Benchmark Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {report['timestamp']}\n")
        f.write(f"Videos processed: {len(results)}\n\n")

        if "summary" in report and report["summary"]:
            s = report["summary"]
            f.write("AGGREGATE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total frames: {s['total_frames']}\n")
            f.write(f"Total time: {s['total_time_sec']:.1f}s\n")
            f.write(f"Overall FPS: {s['overall_fps']:.1f}\n")
            f.write(f"Motion rate: {100*s['motion_rate']:.1f}%\n")
            f.write(f"Total detections: {s['total_detections']}\n")
            f.write(f"Unique tracks: {s['total_unique_tracks']}\n")
            f.write(f"Arrivals: {s['total_arrivals']}\n")
            f.write(f"Departures: {s['total_departures']}\n")
            f.write(f"Captures saved: {s['total_captures']}\n\n")

            f.write("Component Timing (ms):\n")
            for comp in ["motion", "detection", "tracking", "fsm"]:
                stats = s["timing_ms"][comp]
                f.write(f"  {comp:12s}: mean={stats['mean']:6.2f}  p50={stats['p50']:6.2f}  p95={stats['p95']:6.2f}\n")

            f.write(f"\nClass distribution: {s['class_distribution']}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("PER-VIDEO DETAILS\n")
        f.write("=" * 60 + "\n")

        for v in report["videos"]:
            f.write(f"\n{v['name']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Resolution: {v['video_info']['original_resolution']} -> {v['video_info']['processing_resolution']}\n")
            f.write(f"  Frames: {v['processing']['frames_processed']}/{v['video_info']['total_frames']}\n")
            f.write(f"  Processing FPS: {v['processing']['processing_fps']:.1f}\n")
            f.write(f"  Motion rate: {100*v['motion']['motion_rate']:.1f}%\n")
            f.write(f"  Detections: {v['detection']['total_detections']}\n")
            f.write(f"  Tracks: {v['tracking']['unique_tracks']}\n")
            f.write(f"  Events: {v['events']['arrivals']} arrivals, {v['events']['departures']} departures\n")
            f.write(f"  Classes: {v['detection']['class_distribution']}\n")
            f.write(f"  Captures: {len(v['captures'])}\n")

            if v['captures']:
                f.write("  Capture timestamps:\n")
                for c in v['captures'][:20]:  # Show first 20
                    f.write(f"    {c['time_str']} ({c['reason']}): {c['tracks']} tracks, {c['motion_pct']:.1f}% motion\n")
                if len(v['captures']) > 20:
                    f.write(f"    ... and {len(v['captures']) - 20} more\n")

    print(f"\nReports saved:")
    print(f"  JSON: {report_path}")
    print(f"  Summary: {summary_path}")

    return report


def find_sample_videos(storage_root: Path, num_cameras: int = 4) -> list[str]:
    """Find sample videos from different camera directories."""
    videos = []

    # Find camera directories
    camera_dirs = [d for d in storage_root.iterdir() if d.is_dir() and not d.name.startswith('.')]

    # Sample from different cameras
    import random
    sampled_dirs = random.sample(camera_dirs, min(num_cameras, len(camera_dirs)))

    for cam_dir in sampled_dirs:
        # Find a video with some content (> 10MB)
        for mp4 in sorted(cam_dir.rglob("*.mp4"), reverse=True):
            if mp4.stat().st_size > 10_000_000:
                videos.append(str(mp4))
                break

    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Visual benchmark tool for nextgen detection pipeline"
    )
    parser.add_argument("--video", type=str, help="Path to single video file")
    parser.add_argument("--video-dir", type=str, help="Directory containing videos")
    parser.add_argument("--cameras-sample", type=int, default=0,
                       help="Sample N videos from different cameras in storage")
    parser.add_argument("--storage-root", type=str, default="/opt3/ronin/storage",
                       help="Root storage directory for camera sampling")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                       help="Output directory for captures and reports")
    parser.add_argument("--max-frames", type=int, default=500,
                       help="Maximum frames to process per video")
    parser.add_argument("--max-videos", type=int, default=10,
                       help="Maximum videos to process from directory")
    parser.add_argument("--capture-interval", type=int, default=30,
                       help="Capture frame every N frames (0=disable interval captures)")
    parser.add_argument("--events-only", action="store_true",
                       help="Only capture on events (disable interval and motion-start)")
    parser.add_argument("--no-motion-vis", action="store_true",
                       help="Disable motion visualization captures")
    parser.add_argument("--no-combined", action="store_true",
                       help="Disable side-by-side combined captures")
    parser.add_argument("--model-path", type=str,
                       default="/opt3/ronin/ml_models/yolov8n.onnx",
                       help="Path to ONNX model")

    args = parser.parse_args()

    print("=" * 70)
    print("NextGen Detection Pipeline - Visual Benchmark")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Collect videos
    videos: list[str] = []

    if args.video:
        videos.append(args.video)
    elif args.video_dir:
        video_dir = Path(args.video_dir)
        videos = sorted([str(p) for p in video_dir.glob("*.mp4")])[:args.max_videos]
    elif args.cameras_sample > 0:
        videos = find_sample_videos(Path(args.storage_root), args.cameras_sample)
    else:
        print("ERROR: Must specify --video, --video-dir, or --cameras-sample")
        sys.exit(1)

    if not videos:
        print("ERROR: No videos found")
        sys.exit(1)

    print(f"Videos to process: {len(videos)}")
    for v in videos:
        print(f"  - {v}")

    # Check model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    # Initialize orchestrator
    print("\nInitializing GPU pipeline...")
    # Tuned parameters to filter false positives (Christmas lights) while preserving real detections
    config = GPUPipelineConfig(
        device_id=0,
        model_path=str(model_path),
        motion_min_percent=0.3,         # Raised from 0.05 to filter tiny flickers
        detection_confidence=0.65,       # Raised from 0.4 (above false positive max of 0.635)
        track_high_thresh=0.65,
        track_buffer=30,
        track_min_hits=5,               # Raised from 3
        track_min_displacement=0.02,    # Require 2% frame movement to confirm track
        fsm_validation_frames=10,       # Raised from 5
        fsm_stationary_seconds=5.0,
        fsm_parked_seconds=60.0,
    )

    orchestrator = GPUOrchestrator(device_ids=[0], config=config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure capture options
    capture_interval = 0 if args.events_only else args.capture_interval
    capture_on_events = True
    capture_on_motion_start = not args.events_only

    # Process videos
    results: list[VideoResult] = []

    for video_path in videos:
        result = process_video(
            video_path=video_path,
            output_dir=output_dir,
            orchestrator=orchestrator,
            max_frames=args.max_frames,
            capture_interval=capture_interval,
            capture_on_events=capture_on_events,
            capture_on_motion_start=capture_on_motion_start,
            save_motion_vis=not args.no_motion_vis,
            save_side_by_side=not args.no_combined,
        )
        results.append(result)

    # Generate report
    report = generate_report(results, output_dir)

    # Print final summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    if "summary" in report and report["summary"]:
        s = report["summary"]
        print(f"\nVideos: {s['videos_processed']}")
        print(f"Frames: {s['total_frames']}")
        print(f"Time: {s['total_time_sec']:.1f}s")
        print(f"FPS: {s['overall_fps']:.1f}")
        print(f"Motion rate: {100*s['motion_rate']:.1f}%")
        print(f"Detections: {s['total_detections']}")
        print(f"Tracks: {s['total_unique_tracks']}")
        print(f"Events: {s['total_arrivals']} arrivals, {s['total_departures']} departures")
        print(f"Captures: {s['total_captures']}")

    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
