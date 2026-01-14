#!/usr/bin/env python3
"""Analyze batch pipeline detection quality using VLM.

This script reads JSON results from batch_pipeline_test.py and:
1. Identifies "flicker" patterns (rapid arrive/depart cycles)
2. Uses a VLM to analyze detection frame quality
3. Generates a quality report

Usage:
    python tools/analyze_batch_quality.py /path/to/batch_results
    python tools/analyze_batch_quality.py /path/to/batch_results --vlm-url http://192.168.1.125:9001
    python tools/analyze_batch_quality.py /path/to/batch_results --skip-vlm  # Flicker analysis only
"""

import argparse
import base64
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class FlickerEvent:
    """A detected flicker pattern (rapid arrive/depart cycle)."""

    track_id: int
    class_name: str
    arrival_time: float
    departure_time: float
    duration_seconds: float
    arrival_frame: str
    departure_frame: str
    video_name: str
    camera: str


@dataclass
class VLMAnalysis:
    """VLM analysis result for a detection frame."""

    frame_path: str
    event_type: str
    class_name: str
    detected_objects: str
    quality_score: int  # 1-5
    issues: list[str]
    notes: str


@dataclass
class VideoAnalysis:
    """Analysis results for a single video."""

    video_name: str
    camera: str
    date: str
    total_events: int
    flicker_events: list[FlickerEvent] = field(default_factory=list)
    vlm_analyses: list[VLMAnalysis] = field(default_factory=list)


def load_json_results(json_dir: Path) -> list[dict]:
    """Load all JSON result files."""
    results = []
    json_files = sorted(json_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
                data["_json_file"] = str(json_file)
                results.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    return results


def detect_flicker(
    events: list[dict],
    max_duration_seconds: float = 60.0,
) -> list[tuple[dict, dict]]:
    """Detect flicker patterns - rapid arrive/depart cycles.

    Args:
        events: List of events from JSON
        max_duration_seconds: Maximum time between arrive/depart to count as flicker

    Returns:
        List of (arrival_event, departure_event) tuples that are flicker
    """
    flickers = []

    # Group events by track_id
    by_track: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        by_track[event["track_id"]].append(event)

    # For each track, look for rapid arrive/depart pairs
    for track_id, track_events in by_track.items():
        # Sort by timestamp
        track_events.sort(key=lambda e: e["timestamp"])

        arrivals = [e for e in track_events if e["event_type"] == "arrival"]
        departures = [e for e in track_events if e["event_type"] == "departure"]

        # Match arrivals to departures
        for arrival in arrivals:
            # Find the next departure after this arrival
            matching_dep = None
            for dep in departures:
                if dep["timestamp"] > arrival["timestamp"]:
                    matching_dep = dep
                    break

            if matching_dep:
                duration = matching_dep["timestamp"] - arrival["timestamp"]
                if duration <= max_duration_seconds:
                    flickers.append((arrival, matching_dep))

    return flickers


def encode_image_base64(image_path: Path) -> Optional[str]:
    """Encode image to base64 for VLM API."""
    if not image_path.exists():
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_frame_with_vlm(
    frame_path: Path,
    event_type: str,
    class_name: str,
    vlm_url: str,
    timeout: int = 30,
) -> Optional[VLMAnalysis]:
    """Analyze a detection frame using the VLM.

    Args:
        frame_path: Path to the frame image
        event_type: "arrival" or "departure"
        class_name: Detected class (e.g., "truck", "person")
        vlm_url: Base URL of VLM API
        timeout: Request timeout in seconds

    Returns:
        VLMAnalysis result or None if failed
    """
    image_b64 = encode_image_base64(frame_path)
    if not image_b64:
        logger.warning(f"Could not load image: {frame_path}")
        return None

    prompt = f"""Analyze this security camera frame. The detection system marked this as a "{event_type}" event for a "{class_name}".

Please evaluate:
1. Is there actually a {class_name} visible in the frame? If not, what objects ARE visible?
2. Is the bounding box (shown in green) correctly positioned around the {class_name}?
3. Rate the detection quality from 1-5:
   - 5: Perfect detection, correct object, good bbox
   - 4: Correct object, minor bbox issues
   - 3: Correct object but significant bbox offset
   - 2: Wrong object or very poor bbox
   - 1: No relevant object visible (false positive)

Respond in JSON format:
{{
    "detected_objects": "list what objects you see",
    "has_target_object": true/false,
    "bbox_quality": "good/offset/poor",
    "quality_score": 1-5,
    "issues": ["list any issues"],
    "notes": "brief notes"
}}"""

    try:
        # OpenAI-compatible API format
        response = requests.post(
            f"{vlm_url}/v1/chat/completions",
            json={
                "model": "default",  # Use default model
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.1,
            },
            timeout=timeout,
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Try to parse JSON from response
        # Handle case where response might have markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        analysis_data = json.loads(content.strip())

        return VLMAnalysis(
            frame_path=str(frame_path),
            event_type=event_type,
            class_name=class_name,
            detected_objects=analysis_data.get("detected_objects", "unknown"),
            quality_score=analysis_data.get("quality_score", 0),
            issues=analysis_data.get("issues", []),
            notes=analysis_data.get("notes", ""),
        )

    except requests.exceptions.Timeout:
        logger.warning(f"VLM request timed out for {frame_path}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"VLM request failed for {frame_path}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse VLM response for {frame_path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error analyzing {frame_path}: {e}")
        return None


def analyze_results(
    results: list[dict],
    frames_dir: Path,
    vlm_url: Optional[str] = None,
    max_vlm_samples: int = 50,
    flicker_threshold: float = 60.0,
) -> dict:
    """Analyze all results for quality issues.

    Args:
        results: List of JSON result dicts
        frames_dir: Base directory for frame images
        vlm_url: VLM API URL (None to skip VLM analysis)
        max_vlm_samples: Maximum frames to analyze with VLM
        flicker_threshold: Max seconds between arrive/depart to count as flicker

    Returns:
        Analysis summary dict
    """
    all_flickers: list[FlickerEvent] = []
    all_vlm_analyses: list[VLMAnalysis] = []
    videos_with_events = 0
    total_events = 0
    events_by_class: dict[str, int] = defaultdict(int)

    # Collect frames for VLM analysis (sample across videos)
    frames_to_analyze: list[tuple[Path, str, str]] = []

    for result in results:
        events = result.get("events", [])
        if not events:
            continue

        videos_with_events += 1
        total_events += len(events)

        camera = result.get("camera", "unknown")
        date = result.get("date", "unknown")
        video_name = result.get("video_name", "unknown")
        dir_name = f"{camera}_{date}_{video_name}"

        # Count by class
        for event in events:
            events_by_class[event["class_name"]] += 1

        # Detect flicker
        flickers = detect_flicker(events, max_duration_seconds=flicker_threshold)
        for arrival, departure in flickers:
            duration = departure["timestamp"] - arrival["timestamp"]
            flicker = FlickerEvent(
                track_id=arrival["track_id"],
                class_name=arrival["class_name"],
                arrival_time=arrival["timestamp"],
                departure_time=departure["timestamp"],
                duration_seconds=duration,
                arrival_frame=arrival.get("frame_path", ""),
                departure_frame=departure.get("frame_path", ""),
                video_name=video_name,
                camera=camera,
            )
            all_flickers.append(flicker)

        # Collect frames for VLM
        for event in events:
            frame_filename = f"{event['event_type']}_{event['track_id']}_{event['frame_number']}.jpg"
            frame_path = frames_dir / dir_name / frame_filename
            if frame_path.exists():
                frames_to_analyze.append((
                    frame_path,
                    event["event_type"],
                    event["class_name"],
                ))

    # Sample frames for VLM analysis
    if vlm_url and frames_to_analyze:
        # Sample evenly across the dataset
        step = max(1, len(frames_to_analyze) // max_vlm_samples)
        sampled_frames = frames_to_analyze[::step][:max_vlm_samples]

        logger.info(f"Analyzing {len(sampled_frames)} frames with VLM...")
        for i, (frame_path, event_type, class_name) in enumerate(sampled_frames):
            logger.info(f"  [{i+1}/{len(sampled_frames)}] {frame_path.name}")
            analysis = analyze_frame_with_vlm(
                frame_path, event_type, class_name, vlm_url
            )
            if analysis:
                all_vlm_analyses.append(analysis)

    # Build summary
    summary = {
        "total_videos": len(results),
        "videos_with_events": videos_with_events,
        "total_events": total_events,
        "events_by_class": dict(events_by_class),
        "flicker_analysis": {
            "threshold_seconds": flicker_threshold,
            "total_flickers": len(all_flickers),
            "flickers_by_camera": defaultdict(int),
            "flickers_by_class": defaultdict(int),
            "flicker_details": [],
        },
        "vlm_analysis": {
            "frames_analyzed": len(all_vlm_analyses),
            "average_quality_score": 0.0,
            "quality_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "common_issues": defaultdict(int),
            "details": [],
        },
    }

    # Flicker summary
    for flicker in all_flickers:
        summary["flicker_analysis"]["flickers_by_camera"][flicker.camera] += 1
        summary["flicker_analysis"]["flickers_by_class"][flicker.class_name] += 1
        summary["flicker_analysis"]["flicker_details"].append({
            "camera": flicker.camera,
            "video": flicker.video_name,
            "class": flicker.class_name,
            "track_id": flicker.track_id,
            "duration_seconds": round(flicker.duration_seconds, 1),
            "arrival_time": round(flicker.arrival_time, 1),
        })

    # Convert defaultdicts
    summary["flicker_analysis"]["flickers_by_camera"] = dict(
        summary["flicker_analysis"]["flickers_by_camera"]
    )
    summary["flicker_analysis"]["flickers_by_class"] = dict(
        summary["flicker_analysis"]["flickers_by_class"]
    )

    # VLM summary
    if all_vlm_analyses:
        scores = [a.quality_score for a in all_vlm_analyses if a.quality_score > 0]
        if scores:
            summary["vlm_analysis"]["average_quality_score"] = round(
                sum(scores) / len(scores), 2
            )

        for analysis in all_vlm_analyses:
            if analysis.quality_score > 0:
                summary["vlm_analysis"]["quality_distribution"][analysis.quality_score] += 1
            for issue in analysis.issues:
                summary["vlm_analysis"]["common_issues"][issue] += 1
            summary["vlm_analysis"]["details"].append({
                "frame": analysis.frame_path,
                "event_type": analysis.event_type,
                "class_name": analysis.class_name,
                "quality_score": analysis.quality_score,
                "detected_objects": analysis.detected_objects,
                "issues": analysis.issues,
                "notes": analysis.notes,
            })

        summary["vlm_analysis"]["common_issues"] = dict(
            summary["vlm_analysis"]["common_issues"]
        )

    return summary


def print_summary(summary: dict) -> None:
    """Print a human-readable summary."""
    print("\n" + "=" * 70)
    print("BATCH DETECTION QUALITY ANALYSIS")
    print("=" * 70)

    print(f"\nVideos analyzed: {summary['total_videos']}")
    print(f"Videos with events: {summary['videos_with_events']}")
    print(f"Total events: {summary['total_events']}")

    print("\nEvents by class:")
    for cls, count in sorted(summary["events_by_class"].items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")

    # Flicker analysis
    flicker = summary["flicker_analysis"]
    print(f"\n--- FLICKER ANALYSIS (threshold: {flicker['threshold_seconds']}s) ---")
    print(f"Total flicker events: {flicker['total_flickers']}")

    if flicker["total_flickers"] > 0:
        print("\nFlickers by camera:")
        for cam, count in sorted(flicker["flickers_by_camera"].items(), key=lambda x: -x[1]):
            print(f"  {cam}: {count}")

        print("\nFlickers by class:")
        for cls, count in sorted(flicker["flickers_by_class"].items(), key=lambda x: -x[1]):
            print(f"  {cls}: {count}")

        print("\nFlicker details (showing first 20):")
        for detail in flicker["flicker_details"][:20]:
            print(
                f"  {detail['camera']}/{detail['video']}: "
                f"{detail['class']} track {detail['track_id']} - "
                f"{detail['duration_seconds']}s at t={detail['arrival_time']}s"
            )

    # VLM analysis
    vlm = summary["vlm_analysis"]
    if vlm["frames_analyzed"] > 0:
        print(f"\n--- VLM QUALITY ANALYSIS ---")
        print(f"Frames analyzed: {vlm['frames_analyzed']}")
        print(f"Average quality score: {vlm['average_quality_score']}/5")

        print("\nQuality distribution:")
        for score in range(5, 0, -1):
            count = vlm["quality_distribution"][score]
            pct = (count / vlm["frames_analyzed"] * 100) if vlm["frames_analyzed"] > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"  {score}: {count:3d} ({pct:5.1f}%) {bar}")

        if vlm["common_issues"]:
            print("\nCommon issues:")
            for issue, count in sorted(vlm["common_issues"].items(), key=lambda x: -x[1])[:10]:
                print(f"  {issue}: {count}")

        # Show low quality detections
        low_quality = [d for d in vlm["details"] if d["quality_score"] <= 2]
        if low_quality:
            print(f"\nLow quality detections ({len(low_quality)}):")
            for det in low_quality[:10]:
                print(f"  {Path(det['frame']).name}:")
                print(f"    Expected: {det['class_name']}, Found: {det['detected_objects']}")
                print(f"    Score: {det['quality_score']}, Issues: {det['issues']}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze batch detection quality using VLM"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to batch_results directory",
    )
    parser.add_argument(
        "--vlm-url",
        type=str,
        default="http://192.168.1.125:9001",
        help="VLM API URL (default: http://192.168.1.125:9001)",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip VLM analysis, only do flicker detection",
    )
    parser.add_argument(
        "--max-vlm-samples",
        type=int,
        default=50,
        help="Maximum frames to analyze with VLM (default: 50)",
    )
    parser.add_argument(
        "--flicker-threshold",
        type=float,
        default=60.0,
        help="Max seconds between arrive/depart to count as flicker (default: 60)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for detailed results",
    )
    args = parser.parse_args()

    # Validate paths
    json_dir = args.results_dir / "json"
    frames_dir = args.results_dir / "frames"

    if not json_dir.exists():
        logger.error(f"JSON directory not found: {json_dir}")
        sys.exit(1)

    if not frames_dir.exists():
        logger.error(f"Frames directory not found: {frames_dir}")
        sys.exit(1)

    # Load results
    logger.info(f"Loading results from {json_dir}")
    results = load_json_results(json_dir)

    if not results:
        logger.error("No results to analyze")
        sys.exit(1)

    # Analyze
    vlm_url = None if args.skip_vlm else args.vlm_url
    summary = analyze_results(
        results=results,
        frames_dir=frames_dir,
        vlm_url=vlm_url,
        max_vlm_samples=args.max_vlm_samples,
        flicker_threshold=args.flicker_threshold,
    )

    # Print summary
    print_summary(summary)

    # Save detailed results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
