#!/usr/bin/env python3
"""Analyze a specific segment of video frame-by-frame.

Usage:
    python analyze_video_segment.py video.mp4 --start 835 --duration 30 --fps 2
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import httpx
import numpy as np

VLM_ENDPOINT = "http://192.168.1.125:9001/v1/chat/completions"
VLM_PROMPT = """Analyze this security camera frame carefully. List ALL objects you can see, especially:
1. People - any humans, even partially visible
2. Vehicles - cars, trucks, trailers, motorcycles
3. Animals - any animals

Be thorough - list everything visible.

Respond with JSON:
{
    "has_person": true/false,
    "has_vehicle": true/false,
    "has_animal": true/false,
    "person_count": 0,
    "all_objects": ["list", "every", "object", "visible"],
    "description": "brief scene description"
}"""


def encode_frame(frame) -> str:
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def query_vlm(frame) -> dict:
    image_data = encode_frame(frame)
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": VLM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    {"type": "text", "text": "Describe everything you see in this security camera image."}
                ]
            }
        ],
        "max_tokens": 500,
        "temperature": 0.1
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(VLM_ENDPOINT, json=payload)
            response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        import re
        if "{" in content:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        return {"raw": content}
    except Exception as e:
        return {"error": str(e)}


class SimpleYOLO:
    def __init__(self, model_path: str, confidence: float = 0.25):
        import onnxruntime as ort
        self.confidence = confidence
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
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
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame):
        resized = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        outputs = self.session.run(None, {self.input_name: blob})
        output = outputs[0][0].T if len(outputs[0].shape) == 3 else outputs[0]

        detections = []
        for row in output:
            scores = row[4:]
            max_idx = np.argmax(scores)
            max_score = scores[max_idx]
            if max_score >= self.confidence:
                cls = self.class_names[max_idx] if max_idx < len(self.class_names) else f"class_{max_idx}"
                detections.append({"class": cls, "confidence": float(max_score)})

        # Dedupe by class, keep top confidence
        seen = {}
        for d in sorted(detections, key=lambda x: -x['confidence']):
            if d['class'] not in seen:
                seen[d['class']] = d
        return list(seen.values())


def analyze_segment(video_path: str, start_sec: float, duration_sec: float, sample_fps: float, yolo_model: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.1f}, Duration: {video_duration:.1f}s")
    print(f"Analyzing: {start_sec:.1f}s to {start_sec + duration_sec:.1f}s at {sample_fps} fps")
    print()

    # Initialize YOLO
    print("Loading YOLO...")
    yolo = SimpleYOLO(yolo_model, confidence=0.25)

    # MOG2 for motion
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + duration_sec) * fps)
    frame_interval = max(1, int(fps / sample_fps))

    print(f"Frame range: {start_frame} to {end_frame}, interval: {frame_interval}")
    print("=" * 70)

    results = []

    # Warm up MOG2 with a few frames before start
    warmup_start = max(0, start_frame - int(fps * 5))  # 5 seconds before
    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
    for _ in range(start_frame - warmup_start):
        ret, frame = cap.read()
        if ret:
            mog2.apply(frame)

    frame_num = start_frame
    while frame_num < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_num / fps

        # MOG2 motion
        fg_mask = mog2.apply(frame)
        motion_percent = (np.count_nonzero(fg_mask) / (frame.shape[0] * frame.shape[1])) * 100
        has_motion = motion_percent > 0.05

        # YOLO
        yolo_dets = yolo.detect(frame)
        yolo_classes = [d['class'] for d in yolo_dets]

        # VLM
        print(f"Frame {frame_num} ({timestamp:.1f}s): MOG2={motion_percent:.2f}%, YOLO={yolo_classes}")
        vlm_result = query_vlm(frame)

        has_person = vlm_result.get('has_person', False)
        has_vehicle = vlm_result.get('has_vehicle', False)
        has_animal = vlm_result.get('has_animal', False)

        status = []
        if has_person: status.append("PERSON")
        if has_vehicle: status.append("VEHICLE")
        if has_animal: status.append("ANIMAL")

        if status:
            print(f"  VLM FOUND: {', '.join(status)}")
            print(f"  Objects: {vlm_result.get('all_objects', vlm_result.get('objects_detected', []))}")
            print(f"  Description: {vlm_result.get('description', '')[:100]}")
        else:
            print(f"  VLM: nothing of interest")

        results.append({
            "frame": frame_num,
            "timestamp": timestamp,
            "motion_percent": motion_percent,
            "yolo": yolo_dets,
            "vlm_person": has_person,
            "vlm_vehicle": has_vehicle,
            "vlm_animal": has_animal,
            "vlm_objects": vlm_result.get('all_objects', []),
            "vlm_description": vlm_result.get('description', '')
        })

        frame_num += frame_interval

    cap.release()

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    frames_analyzed = len(results)
    frames_with_person = sum(1 for r in results if r['vlm_person'])
    frames_with_vehicle = sum(1 for r in results if r['vlm_vehicle'])
    frames_with_animal = sum(1 for r in results if r['vlm_animal'])

    print(f"Frames analyzed: {frames_analyzed}")
    print(f"Frames with person: {frames_with_person}")
    print(f"Frames with vehicle: {frames_with_vehicle}")
    print(f"Frames with animal: {frames_with_animal}")

    # Check YOLO vs VLM agreement
    yolo_person = sum(1 for r in results if 'person' in [d['class'] for d in r['yolo']])
    print()
    print(f"YOLO detected 'person' in {yolo_person} frames")
    print(f"VLM confirmed person in {frames_with_person} frames")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Video file path")
    parser.add_argument("--start", type=float, required=True, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=30, help="Duration to analyze in seconds")
    parser.add_argument("--fps", type=float, default=2, help="Frames per second to sample")
    parser.add_argument("--yolo", default="/opt3/ronin/ml_models/yolo11l_dynamic.onnx")
    parser.add_argument("--output", help="Save results to JSON")
    args = parser.parse_args()

    results = analyze_segment(args.video, args.start, args.duration, args.fps, args.yolo)

    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
