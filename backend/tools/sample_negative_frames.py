#!/usr/bin/env python3
"""Sample frames where no detector fired and send to VLM to find false negatives.

This helps identify cases where people/vehicles/animals were present but
all detectors missed them.

Usage:
    python sample_negative_frames.py /path/to/video.mp4 --samples 20
"""

import argparse
import base64
import json
import random
import sys
from pathlib import Path

import cv2
import httpx


VLM_ENDPOINT = "http://192.168.1.125:9001/v1/chat/completions"
VLM_PROMPT = """Analyze this security camera frame. Focus on identifying:
1. People - any humans visible
2. Vehicles - cars, trucks, motorcycles, etc.
3. Animals - pets, wildlife

Respond with JSON:
{
    "has_person": true/false,
    "has_vehicle": true/false,
    "has_animal": true/false,
    "objects_detected": ["list", "of", "objects"],
    "confidence": "high" | "medium" | "low",
    "notes": "description"
}"""


def encode_frame(frame) -> str:
    """Encode frame as base64 JPEG."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def query_vlm(frame) -> dict:
    """Send frame to VLM and get response."""
    image_data = encode_frame(frame)

    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": VLM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    },
                    {"type": "text", "text": "What do you see in this security camera image?"}
                ]
            }
        ],
        "max_tokens": 500,
        "temperature": 0.1
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(VLM_ENDPOINT, json=payload)
        response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    # Parse JSON from response
    try:
        if "{" in content:
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
    except:
        pass

    return {"raw_response": content}


def sample_frames(video_path: str, num_samples: int = 20, sample_fps: float = 1.0):
    """Sample random frames from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    # Sample at 1 FPS like the benchmark does
    frame_interval = max(1, int(fps / sample_fps))
    sample_indices = list(range(0, total_frames, frame_interval))

    # Randomly select frames to check
    if len(sample_indices) > num_samples:
        selected = random.sample(sample_indices, num_samples)
    else:
        selected = sample_indices

    selected.sort()

    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps:.1f}")
    print(f"Sampling {len(selected)} frames for VLM analysis...")
    print()

    results = {
        "video": video_path,
        "frames_analyzed": 0,
        "people_found": 0,
        "vehicles_found": 0,
        "animals_found": 0,
        "interesting_frames": []
    }

    for i, frame_idx in enumerate(selected):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        timestamp = frame_idx / fps
        print(f"[{i+1}/{len(selected)}] Frame {frame_idx} ({timestamp:.1f}s)...", end=" ", flush=True)

        try:
            vlm_result = query_vlm(frame)
            results["frames_analyzed"] += 1

            has_person = vlm_result.get("has_person", False)
            has_vehicle = vlm_result.get("has_vehicle", False)
            has_animal = vlm_result.get("has_animal", False)

            if has_person:
                results["people_found"] += 1
            if has_vehicle:
                results["vehicles_found"] += 1
            if has_animal:
                results["animals_found"] += 1

            if has_person or has_vehicle or has_animal:
                status = []
                if has_person:
                    status.append("PERSON")
                if has_vehicle:
                    status.append("VEHICLE")
                if has_animal:
                    status.append("ANIMAL")
                print(f"FOUND: {', '.join(status)}")
                results["interesting_frames"].append({
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "vlm_result": vlm_result
                })
            else:
                print("nothing")

        except Exception as e:
            print(f"ERROR: {e}")

    cap.release()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Frames analyzed: {results['frames_analyzed']}")
    print(f"People found: {results['people_found']}")
    print(f"Vehicles found: {results['vehicles_found']}")
    print(f"Animals found: {results['animals_found']}")
    print()

    if results["interesting_frames"]:
        print("Interesting frames (potential false negatives if detectors missed these):")
        for f in results["interesting_frames"]:
            print(f"  Frame {f['frame']} ({f['timestamp']:.1f}s): {f['vlm_result'].get('objects_detected', [])}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Sample frames and check with VLM for false negatives")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--samples", type=int, default=20, help="Number of frames to sample (default: 20)")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    results = sample_frames(args.video, args.samples)

    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
