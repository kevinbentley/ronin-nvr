#!/usr/bin/env python3
"""Send a video frame to vLLM for description."""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import requests


def extract_frame(video_path: Path, timestamp: float, scale_height: int = 0) -> bytes:
    """Extract a frame from video and return as JPEG bytes."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read frame at {timestamp}s")

    # Scale if requested
    if scale_height > 0 and frame.shape[0] != scale_height:
        scale_factor = scale_height / frame.shape[0]
        new_width = int(frame.shape[1] * scale_factor)
        frame = cv2.resize(frame, (new_width, scale_height))

    # Convert to JPEG
    _, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jpeg_bytes.tobytes()


def describe_image(image_bytes: bytes, vllm_url: str, prompt: str) -> str:
    """Send image to vLLM and get description."""
    # Encode image as base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # OpenAI-compatible API request
    payload = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",  # Adjust model name as needed
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 500,
        "temperature": 0.1,
    }

    response = requests.post(
        f"{vllm_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Send video frame to vLLM for description")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--timestamp", type=float, default=0, help="Timestamp in seconds")
    parser.add_argument("--scale-height", type=int, default=0, help="Scale to this height (0=no scaling)")
    parser.add_argument("--vllm-url", default="http://192.168.1.125:9001", help="vLLM endpoint URL")
    parser.add_argument("--prompt", default="Describe what you see in this image. Is there a person visible? If so, describe their location and what they are doing.", help="Prompt for the model")
    parser.add_argument("--save-frame", help="Save the frame to this path")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        return 1

    print(f"Extracting frame at {args.timestamp}s from {video_path.name}...")
    frame_bytes = extract_frame(video_path, args.timestamp, args.scale_height)
    print(f"Frame size: {len(frame_bytes)} bytes")

    if args.save_frame:
        with open(args.save_frame, "wb") as f:
            f.write(frame_bytes)
        print(f"Saved frame to: {args.save_frame}")

    print(f"\nSending to vLLM at {args.vllm_url}...")
    print(f"Prompt: {args.prompt}\n")

    try:
        description = describe_image(frame_bytes, args.vllm_url, args.prompt)
        print("=" * 60)
        print("vLLM RESPONSE:")
        print("=" * 60)
        print(description)
    except requests.exceptions.RequestException as e:
        print(f"Error calling vLLM: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
