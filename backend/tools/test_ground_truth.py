#!/usr/bin/env python3
"""Test detection ground truth on specific videos.

Runs low-threshold MOG2 + YOLO on test videos, then VLM labels detections.
This helps determine what percentage of real objects we're catching.

Usage:
    python test_ground_truth.py /path/to/video.mp4 --output results.json
"""

import argparse
import base64
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import httpx
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# VLM configuration
VLM_ENDPOINT = "http://192.168.1.125:9001/v1/chat/completions"
VLM_PROMPT = """Analyze this security camera frame. Focus on identifying:
1. People - any humans visible
2. Vehicles - cars, trucks, motorcycles, trailers, etc.
3. Animals - pets, wildlife

Respond with JSON:
{
    "has_person": true/false,
    "has_vehicle": true/false,
    "has_animal": true/false,
    "person_count": 0,
    "objects_detected": ["list", "of", "objects"],
    "confidence": "high" | "medium" | "low",
    "notes": "brief description"
}"""


@dataclass
class FrameResult:
    """Result for a single frame."""
    frame_number: int
    timestamp: float
    mog2_motion: bool
    mog2_area_percent: float
    yolo_detections: list = field(default_factory=list)
    vlm_result: dict = field(default_factory=dict)
    vlm_has_person: bool = False
    vlm_has_vehicle: bool = False
    vlm_has_animal: bool = False


def encode_frame(frame) -> str:
    """Encode frame as base64 JPEG."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def query_vlm(frame, timeout: float = 30.0) -> dict:
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

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(VLM_ENDPOINT, json=payload)
            response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Parse JSON from response
        import re
        if "{" in content:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

        return {"raw_response": content, "parse_error": True}

    except Exception as e:
        return {"error": str(e)}


class SimpleYOLODetector:
    """Simple YOLO detector using ONNX Runtime."""

    def __init__(self, model_path: str, confidence: float = 0.25):
        import onnxruntime as ort

        self.confidence = confidence
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

        # Get input shape
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = input_info.shape
        self.input_height = 640
        self.input_width = 640

    def detect(self, frame: np.ndarray) -> list:
        """Run detection on frame."""
        # Preprocess
        original_h, original_w = frame.shape[:2]

        # Resize and normalize
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = normalized.transpose(2, 0, 1)
        batched = np.expand_dims(transposed, axis=0)

        # Run inference
        outputs = self.session.run(None, {self.input_name: batched})
        output = outputs[0]

        # Parse output - shape is (1, 84, 8400) for YOLOv8
        if len(output.shape) == 3:
            output = output[0].T  # Now (8400, 84)

        detections = []

        for i in range(output.shape[0]):
            row = output[i]

            # First 4 values are box coords, rest are class scores
            x_center, y_center, width, height = row[:4]
            class_scores = row[4:]

            max_score_idx = np.argmax(class_scores)
            max_score = class_scores[max_score_idx]

            if max_score >= self.confidence:
                class_name = self.class_names[max_score_idx] if max_score_idx < len(self.class_names) else f"class_{max_score_idx}"

                detections.append({
                    "class": class_name,
                    "confidence": float(max_score),
                    "class_id": int(max_score_idx)
                })

        # NMS - simple version, just take top detections per class
        seen_classes = {}
        filtered = []
        for d in sorted(detections, key=lambda x: -x['confidence']):
            cls = d['class']
            if cls not in seen_classes:
                seen_classes[cls] = 0
            if seen_classes[cls] < 3:  # Max 3 per class
                filtered.append(d)
                seen_classes[cls] += 1

        return filtered


class SimpleMOG2Detector:
    """Simple MOG2 motion detector."""

    def __init__(self, history: int = 500, var_threshold: float = 16.0, min_area_percent: float = 0.05):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )
        self.min_area_percent = min_area_percent

    def detect(self, frame: np.ndarray) -> tuple[bool, float]:
        """Detect motion, return (has_motion, area_percent)."""
        fg_mask = self.bg_subtractor.apply(frame)

        # Calculate motion area
        total_pixels = frame.shape[0] * frame.shape[1]
        motion_pixels = np.count_nonzero(fg_mask)
        area_percent = (motion_pixels / total_pixels) * 100

        has_motion = area_percent >= self.min_area_percent
        return has_motion, area_percent


def process_video(
    video_path: str,
    sample_fps: float = 1.0,
    max_frames: int = 500,
    yolo_model: str = "/opt3/ronin/ml_models/yolo11l_dynamic.onnx",
    mog2_min_area: float = 0.05,
    yolo_confidence: float = 0.25,
) -> dict:
    """Process video and return detection results."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return {"error": "Could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"Video: {video_path}")
    logger.info(f"FPS: {fps:.1f}, Duration: {duration:.1f}s, Frames: {total_frames}")

    # Initialize detectors
    logger.info("Initializing YOLO detector...")
    yolo = SimpleYOLODetector(yolo_model, confidence=yolo_confidence)

    logger.info("Initializing MOG2 detector...")
    mog2 = SimpleMOG2Detector(min_area_percent=mog2_min_area)

    # Sample frames
    frame_interval = max(1, int(fps / sample_fps))

    results = []
    frames_with_motion = 0
    frames_with_yolo = 0
    frames_with_objects = 0  # VLM confirmed

    frame_idx = 0
    sample_count = 0

    while sample_count < max_frames:
        target_frame = frame_idx * frame_interval
        if target_frame >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()

        if not ret:
            break

        timestamp = target_frame / fps

        # Run MOG2
        has_motion, motion_area = mog2.detect(frame)

        # Run YOLO
        yolo_dets = yolo.detect(frame)

        # Create result
        result = FrameResult(
            frame_number=target_frame,
            timestamp=timestamp,
            mog2_motion=has_motion,
            mog2_area_percent=motion_area,
            yolo_detections=yolo_dets
        )

        if has_motion:
            frames_with_motion += 1
        if yolo_dets:
            frames_with_yolo += 1

        # Send to VLM if either detector fired
        if has_motion or yolo_dets:
            logger.info(f"Frame {target_frame} ({timestamp:.1f}s): MOG2={has_motion} ({motion_area:.2f}%), YOLO={len(yolo_dets)} dets")

            vlm_result = query_vlm(frame)
            result.vlm_result = vlm_result
            result.vlm_has_person = vlm_result.get('has_person', False)
            result.vlm_has_vehicle = vlm_result.get('has_vehicle', False)
            result.vlm_has_animal = vlm_result.get('has_animal', False)

            if result.vlm_has_person or result.vlm_has_vehicle or result.vlm_has_animal:
                frames_with_objects += 1
                logger.info(f"  VLM: person={result.vlm_has_person}, vehicle={result.vlm_has_vehicle}, animal={result.vlm_has_animal}")
                logger.info(f"  Objects: {vlm_result.get('objects_detected', [])}")
            else:
                logger.info(f"  VLM: no objects of interest")

        results.append(result)
        frame_idx += 1
        sample_count += 1

        if sample_count % 50 == 0:
            logger.info(f"Progress: {sample_count} frames sampled")

    cap.release()

    # Summary
    summary = {
        "video": video_path,
        "total_frames": total_frames,
        "duration_seconds": duration,
        "frames_sampled": sample_count,
        "frames_with_motion": frames_with_motion,
        "frames_with_yolo_detection": frames_with_yolo,
        "frames_sent_to_vlm": frames_with_motion + frames_with_yolo - len([r for r in results if r.mog2_motion and r.yolo_detections]),
        "frames_with_real_objects": frames_with_objects,
        "detection_rate": frames_with_objects / sample_count if sample_count > 0 else 0,
    }

    # Detailed results
    detailed = []
    for r in results:
        if r.mog2_motion or r.yolo_detections:
            detailed.append({
                "frame": r.frame_number,
                "timestamp": r.timestamp,
                "mog2_motion": r.mog2_motion,
                "mog2_area": r.mog2_area_percent,
                "yolo": r.yolo_detections,
                "vlm_person": r.vlm_has_person,
                "vlm_vehicle": r.vlm_has_vehicle,
                "vlm_animal": r.vlm_has_animal,
                "vlm_objects": r.vlm_result.get('objects_detected', []),
            })

    return {
        "summary": summary,
        "detections": detailed
    }


def main():
    parser = argparse.ArgumentParser(description="Test detection ground truth on video")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--output", type=str, default="ground_truth_results.json", help="Output JSON file")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Frames per second to sample")
    parser.add_argument("--max-frames", type=int, default=500, help="Maximum frames to process")
    parser.add_argument("--yolo-confidence", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--mog2-min-area", type=float, default=0.05, help="MOG2 minimum motion area percent")
    parser.add_argument("--yolo-model", type=str, default="/opt3/ronin/ml_models/yolo11l_dynamic.onnx")
    args = parser.parse_args()

    results = process_video(
        args.video,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        yolo_model=args.yolo_model,
        mog2_min_area=args.mog2_min_area,
        yolo_confidence=args.yolo_confidence,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    s = results["summary"]
    print(f"Video duration: {s['duration_seconds']:.1f}s")
    print(f"Frames sampled: {s['frames_sampled']}")
    print(f"Frames with MOG2 motion: {s['frames_with_motion']}")
    print(f"Frames with YOLO detection: {s['frames_with_yolo_detection']}")
    print(f"Frames with real objects (VLM): {s['frames_with_real_objects']}")
    print(f"Detection rate: {s['detection_rate']*100:.1f}%")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
