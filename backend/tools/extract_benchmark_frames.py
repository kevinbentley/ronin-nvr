#!/usr/bin/env python3
"""Extract interesting frames from benchmark results for experimentation.

Usage:
    python extract_benchmark_frames.py benchmark_results/benchmark_XXX.json output_dir/

This extracts frames that are worth investigating:
- High confidence detections (>0.7)
- Unusual class detections (toilet, airplane, etc.)
- VLM disagreements (false positives/negatives)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2


def extract_frame(video_path: str, frame_number: int, output_path: Path) -> bool:
    """Extract a single frame from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Could not open: {video_path}")
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"  Could not read frame {frame_number}")
        return False

    cv2.imwrite(str(output_path), frame)
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract frames from benchmark results")
    parser.add_argument("benchmark_json", type=Path, help="Path to benchmark JSON file")
    parser.add_argument("output_dir", type=Path, help="Output directory for frames")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum confidence")
    parser.add_argument("--max-frames", type=int, default=50, help="Maximum frames to extract")
    parser.add_argument("--classes", nargs="*", help="Specific classes to extract (e.g., toilet airplane)")
    args = parser.parse_args()

    # Load benchmark results
    with open(args.benchmark_json) as f:
        data = json.load(f)

    events = data.get('full_result', {}).get('candidate_events', [])
    print(f"Loaded {len(events)} events from benchmark")

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Categorize events
    categories = defaultdict(list)

    for e in events:
        for d in e['detections']:
            class_name = d.get('metadata', {}).get('class_name', d['event_type'])
            conf = d['confidence']

            # Skip low confidence
            if conf < args.min_confidence:
                continue

            # Filter by class if specified
            if args.classes and class_name not in args.classes:
                continue

            categories[class_name].append({
                'video': e['video_path'],
                'frame': e['frame_number'],
                'time': e['timestamp_seconds'],
                'confidence': conf,
                'event_type': d['event_type'],
                'vlm_label': e.get('vlm_label'),
            })

    # Report what we found
    print("\nDetections by class:")
    for class_name, detections in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"  {class_name}: {len(detections)}")

    # Extract frames
    print(f"\nExtracting frames (max {args.max_frames})...")

    extracted = 0
    for class_name, detections in categories.items():
        class_dir = args.output_dir / class_name
        class_dir.mkdir(exist_ok=True)

        # Sort by confidence, take top ones
        detections.sort(key=lambda x: -x['confidence'])

        for det in detections[:args.max_frames // len(categories) + 1]:
            if extracted >= args.max_frames:
                break

            video_name = Path(det['video']).stem
            output_name = f"{video_name}_f{det['frame']}_c{det['confidence']:.2f}.jpg"
            output_path = class_dir / output_name

            print(f"Extracting: {class_name}/{output_name}")
            if extract_frame(det['video'], det['frame'], output_path):
                # Write metadata
                meta_path = output_path.with_suffix('.json')
                with open(meta_path, 'w') as f:
                    json.dump(det, f, indent=2)
                extracted += 1

    print(f"\nExtracted {extracted} frames to {args.output_dir}")
    print("\nTo view:")
    print(f"  ls {args.output_dir}/*/")


if __name__ == "__main__":
    main()
