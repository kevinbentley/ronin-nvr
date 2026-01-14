#!/usr/bin/env python3
"""Test tuned parameters to filter Christmas lights false positives."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from app.services.ml.gpu_orchestrator import GPUOrchestrator, GPUPipelineConfig
from app.services.ml.object_fsm import EventType


def test_video(video_path: str, config: GPUPipelineConfig, max_frames: int = 3000):
    """Test a video with given config."""
    print(f"\n{'='*60}")
    print(f"Testing: {Path(video_path).name}")
    print(f"Config: motion_min={config.motion_min_percent}%, "
          f"det_conf={config.detection_confidence}, "
          f"min_disp={config.track_min_displacement}")
    print(f"{'='*60}")
    
    orchestrator = GPUOrchestrator(device_ids=[0], config=config)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return None
    
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    camera_id = 1
    frame_count = 0
    motion_frames = 0
    detection_frames = 0
    unique_tracks = set()
    events = []
    
    start_time = time.perf_counter()
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        video_time = frame_count / fps
        
        # Scale to 720p
        if orig_h > 720:
            scale = 720 / orig_h
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        result = orchestrator.process(camera_id, frame, time.time())
        
        if result.motion_detected:
            motion_frames += 1
        
        if result.detections:
            detection_frames += 1
        
        for track in result.tracks:
            unique_tracks.add(track.track_id)
        
        for event in result.events:
            events.append({
                'frame': frame_count,
                'time': video_time,
                'type': event.event_type.value,
                'class': event.class_name,
                'track_id': event.track_id,
            })
            print(f"  [{frame_count:5d}] {video_time:6.2f}s: {event.event_type.value:12s} "
                  f"{event.class_name} #{event.track_id}")
        
        if frame_count % 500 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  Frame {frame_count}: {frame_count/elapsed:.1f} FPS, "
                  f"motion={motion_frames}, tracks={len(unique_tracks)}")
    
    cap.release()
    elapsed = time.perf_counter() - start_time
    
    arrivals = [e for e in events if e['type'] == 'arrival']
    departures = [e for e in events if e['type'] == 'departure']
    
    print(f"\nResults:")
    print(f"  Frames: {frame_count}")
    print(f"  FPS: {frame_count/elapsed:.1f}")
    print(f"  Motion frames: {motion_frames} ({100*motion_frames/frame_count:.1f}%)")
    print(f"  Detection frames: {detection_frames}")
    print(f"  Unique tracks: {len(unique_tracks)}")
    print(f"  Arrivals: {len(arrivals)}")
    print(f"  Departures: {len(departures)}")
    
    return {
        'frames': frame_count,
        'motion': motion_frames,
        'detections': detection_frames,
        'tracks': len(unique_tracks),
        'arrivals': len(arrivals),
        'departures': len(departures),
        'events': events,
    }


def main():
    # Videos to test
    christmas_video = "/opt3/ronin/storage/Hangar_East/2025-12-31/03-05-54.mp4"
    vehicle_video = "/workspace/ronin-nvr/15-01-38.mp4"
    
    # Original config (for comparison)
    original_config = GPUPipelineConfig(
        device_id=0,
        model_path="/opt3/ronin/ml_models/yolov8n.onnx",
        motion_min_percent=0.05,
        detection_confidence=0.4,
        track_high_thresh=0.5,
        track_min_hits=3,
        track_min_displacement=0.0,  # Disabled
        fsm_validation_frames=5,
    )
    
    # Tuned config
    tuned_config = GPUPipelineConfig(
        device_id=0,
        model_path="/opt3/ronin/ml_models/yolov8n.onnx",
        motion_min_percent=0.3,       # Raised from 0.05%
        detection_confidence=0.5,      # Raised from 0.4
        track_high_thresh=0.5,
        track_min_hits=5,              # Raised from 3
        track_min_displacement=0.02,   # Require 2% movement to confirm
        fsm_validation_frames=10,      # Raised from 5
    )
    
    print("\n" + "="*70)
    print("TESTING ORIGINAL CONFIG ON CHRISTMAS LIGHTS VIDEO")
    print("="*70)
    orig_christmas = test_video(christmas_video, original_config, max_frames=3000)
    
    print("\n" + "="*70)
    print("TESTING TUNED CONFIG ON CHRISTMAS LIGHTS VIDEO")
    print("="*70)
    tuned_christmas = test_video(christmas_video, tuned_config, max_frames=3000)
    
    print("\n" + "="*70)
    print("TESTING TUNED CONFIG ON VEHICLE VIDEO")
    print("="*70)
    tuned_vehicle = test_video(vehicle_video, tuned_config, max_frames=6000)
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print("\nChristmas Lights Video:")
    print(f"  Original: {orig_christmas['arrivals']} arrivals, {orig_christmas['tracks']} tracks")
    print(f"  Tuned:    {tuned_christmas['arrivals']} arrivals, {tuned_christmas['tracks']} tracks")
    
    print("\nVehicle Video (tuned only):")
    print(f"  {tuned_vehicle['arrivals']} arrivals, {tuned_vehicle['departures']} departures, "
          f"{tuned_vehicle['tracks']} tracks")
    
    print("\nFalse positive reduction:")
    if orig_christmas['arrivals'] > 0:
        reduction = 100 * (1 - tuned_christmas['arrivals'] / orig_christmas['arrivals'])
        print(f"  {reduction:.0f}% fewer false arrivals on Christmas lights")
    
    print()


if __name__ == "__main__":
    main()
