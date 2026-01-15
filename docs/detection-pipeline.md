# Detection Pipeline Architecture

This document describes how motion detection, object recognition, tracking, and the finite state machine (FSM) work together to provide intelligent object detection and event generation in Ronin NVR.

## Overview

The detection pipeline processes video streams through four stages:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Motion    │    │   Object    │    │   Object    │    │    FSM      │
│  Detection  │───▶│ Detection   │───▶│  Tracking   │───▶│  Lifecycle  │
│   (MOG2)    │    │   (YOLO)    │    │ (ByteTrack) │    │  Management │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                   │
      ▼                  ▼                  ▼                   ▼
  "Is there          "What is           "Same object       "Arrival/
   movement?"         there?"            over time?"       Departure?"
```

**Why this architecture?**

- **Motion detection** acts as a gate to avoid running expensive object detection on static scenes
- **Object detection** identifies what objects are present in each frame
- **Tracking** associates detections across frames to maintain object identity
- **FSM** manages object lifecycles and generates meaningful events (arrivals, departures)

---

## Stage 1: Motion Detection (MOG2)

**Source:** `backend/app/services/ml/gpu_motion.py`

### Purpose

Motion detection determines whether a frame contains any significant movement. This acts as an efficiency gate—if nothing is moving, we skip the expensive object detection step.

### Algorithm: MOG2 (Mixture of Gaussians)

MOG2 is a background subtraction algorithm that maintains a statistical model of the background. Each pixel is modeled as a mixture of Gaussian distributions, allowing the algorithm to:

- Learn what the "normal" background looks like
- Detect foreground (moving) objects
- Adapt to gradual lighting changes
- Distinguish shadows from actual motion

### Processing Pipeline

```
Input Frame
     │
     ▼
┌─────────────────┐
│ Convert to      │
│ Grayscale       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MOG2 Background │  ◀── Maintains background model
│ Subtraction     │      (history of ~500 frames)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Morphological   │  ◀── Erosion removes noise (rain, sensor noise)
│ Filtering       │      Dilation connects nearby regions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Shadow Removal  │  ◀── Shadows marked as 127, foreground as 255
│ (threshold)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Contour         │  ◀── Find connected regions
│ Analysis        │      Filter by minimum area
└────────┬────────┘
         │
         ▼
   Motion Result
   (detected: bool, percent: float, contours: int)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `history` | 500 | Frames for background model (~17s at 30fps) |
| `var_threshold` | 16.0 | Variance threshold (lower = more sensitive) |
| `detect_shadows` | true | Mark shadows separately from foreground |
| `min_motion_percent` | 0.1 | Minimum % of frame with motion to trigger |
| `min_contour_area` | 500 | Minimum pixel area for valid motion region |
| `erosion_kernel_size` | 3 | Erosion kernel (removes small noise) |
| `dilation_kernel_size` | 5 | Dilation kernel (connects regions) |

### GPU Acceleration

The implementation uses OpenCV's CUDA backend for GPU acceleration:

```python
# GPU-accelerated MOG2
self._mog2 = cv2.cuda.createBackgroundSubtractorMOG2(
    history=history,
    varThreshold=var_threshold,
    detectShadows=detect_shadows,
)

# GPU morphological operations
self._erode_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_ERODE, ...)
self._dilate_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, ...)
```

This provides ~10x speedup over CPU processing and keeps data in GPU memory for the subsequent YOLO detection.

---

## Stage 2: Object Detection (YOLO)

**Source:** `backend/app/services/ml/tensorrt_inference.py`

### Purpose

When motion is detected, YOLO identifies what objects are present in the frame and their locations.

### Algorithm: YOLOv8/YOLO11

YOLO (You Only Look Once) is a real-time object detection neural network that:

- Processes the entire image in a single forward pass
- Outputs bounding boxes, class labels, and confidence scores
- Detects 80 COCO classes (person, car, truck, dog, etc.)

### Processing Pipeline

```
Input Frame (BGR)
     │
     ▼
┌─────────────────┐
│ Resize to       │  ◀── Model expects 640x640
│ 640x640         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Normalize       │  ◀── Scale pixels to 0-1
│ (0-255 → 0-1)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ BGR → RGB       │  ◀── Model trained on RGB
│ HWC → CHW       │      Channels first format
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Neural Network  │  ◀── GPU inference via ONNX Runtime
│ Inference       │      or TensorRT
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Post-process    │  ◀── Filter by confidence threshold
│ & NMS           │      Non-maximum suppression
└────────┬────────┘
         │
         ▼
   List[Detection]
   (class, confidence, bbox)
```

### Output Format

Each detection contains:

```python
@dataclass
class TensorRTDetection:
    class_name: str      # "person", "car", etc.
    class_id: int        # COCO class index
    confidence: float    # 0.0 - 1.0
    x: float            # Normalized bbox (0-1)
    y: float
    width: float
    height: float
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.5 | Minimum confidence to keep detection |
| `nms_threshold` | 0.45 | IoU threshold for NMS |
| `input_size` | (640, 640) | Model input dimensions |
| `class_thresholds` | {} | Per-class confidence overrides |

### Per-Class Thresholds

Different object types may need different confidence thresholds:

```python
detector = TensorRTDetector(
    model_path="yolov8l.onnx",
    confidence_threshold=0.5,
    class_thresholds={
        "person": 0.45,  # Lower threshold for people
        "car": 0.5,
        "dog": 0.6,      # Higher threshold to reduce false positives
    }
)
```

---

## Stage 3: Object Tracking (ByteTrack)

**Source:** `backend/app/services/ml/tracker.py`

### Purpose

Tracking associates detections across frames to maintain consistent object identity. Without tracking, each frame's detections are independent—we wouldn't know if the "person" in frame 100 is the same as in frame 99.

### Algorithm: ByteTrack

ByteTrack is a simple yet effective multi-object tracking algorithm that:

- Uses a Kalman filter to predict object motion
- Matches detections to tracks using IoU (Intersection over Union)
- Handles occlusions by keeping "lost" tracks in a buffer
- Uses two-stage matching (high-confidence then low-confidence detections)

### Kalman Filter

The Kalman filter maintains state for each tracked object:

```
State vector: [x, y, w, h, vx, vy, vw, vh]
              └─position─┘ └──velocity──┘
```

This allows predicting where an object will be in the next frame, even if detection temporarily fails.

### Matching Process

```
Frame N Detections          Existing Tracks
        │                          │
        ▼                          ▼
┌───────────────────────────────────────────┐
│           Kalman Prediction               │
│    (predict where tracks should be)       │
└─────────────────────┬─────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────┐
│         IoU Distance Matrix               │
│   (how much do boxes overlap?)            │
└─────────────────────┬─────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────┐
│       Hungarian Algorithm                 │
│   (optimal assignment of dets to tracks)  │
└─────────────────────┬─────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
    Matched Pairs           Unmatched
    (update tracks)         (new tracks / lost)
```

### Track States

```
TENTATIVE ──(enough hits)──▶ CONFIRMED ──(no detections)──▶ LOST
     │                            │                           │
     │                            │                           │
     ▼                            ▼                           ▼
 (removed if              (active tracking)            (kept in buffer,
  no matches)                                          may be recovered)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `track_high_thresh` | 0.5 | High-confidence detection threshold |
| `track_low_thresh` | 0.1 | Low-confidence detection threshold |
| `match_thresh` | 0.8 | Maximum IoU distance for matching |
| `track_buffer` | 30 | Frames to keep lost tracks |
| `min_hits` | 3 | Detections needed to confirm track |

### Output

Each tracked object contains:

```python
@dataclass
class TrackedObject:
    track_id: int        # Unique ID (persistent across frames)
    class_id: int
    class_name: str
    x, y, width, height: float  # Current position
    velocity_x, velocity_y: float  # Estimated velocity
    confidence: float
    state: TrackState    # TENTATIVE, CONFIRMED, or LOST
    hits: int            # Number of matched detections
    age: int             # Total frames since creation
```

---

## Stage 4: FSM Lifecycle Management

**Source:** `backend/app/services/ml/object_fsm.py`

### Purpose

The FSM (Finite State Machine) manages the lifecycle of tracked objects and generates meaningful events like arrivals and departures. It solves several problems:

1. **Noise reduction**: A brief detection shouldn't trigger an alert
2. **Parked object handling**: A car parked for hours shouldn't generate repeated events
3. **Arrival/departure detection**: Know when objects enter and leave the scene

### State Diagram

```
                              ┌─────────────────────────────────────┐
                              │                                     │
                              ▼                                     │
┌───────────┐  validated   ┌────────┐  stopped    ┌────────────┐   │
│ TENTATIVE │─────────────▶│ ACTIVE │────────────▶│ STATIONARY │   │
└───────────┘   + moved    └────────┘  (10 sec)   └────────────┘   │
      │                         │                       │          │
      │                         │                       │ (5 min)  │
      │ validated               │                       ▼          │
      │ + not moved             │                  ┌────────┐      │
      │                         │                  │ PARKED │──────┘
      ▼                         │                  └────────┘ moved
┌───────────┐                   │                       │
│  PARKED   │◀──────────────────┘                       │
│(deferred) │   moved within                            │
└───────────┘   threshold?                              │
      │              │                                  │
      │ YES          │ NO                               │
      ▼              ▼                                  │
  ARRIVAL        (no event)                             │
  (delayed)      truly pre-                             │
                 existing                               │
                                                        ▼
                                                   DEPARTED
                                                 (left scene)
```

### State Descriptions

| State | Description | Duration |
|-------|-------------|----------|
| **TENTATIVE** | New detection, not yet validated | < validation_frames |
| **ACTIVE** | Confirmed moving object | While moving |
| **STATIONARY** | Object stopped but may move again | 10 seconds - 5 minutes |
| **PARKED** | Object stationary for extended period | > 5 minutes |
| **DEPARTED** | Object left the scene | Terminal state |

### Event Types

| Event | Trigger | Use Case |
|-------|---------|----------|
| **ARRIVAL** | Object confirmed and has moved | "Person entered area" |
| **DEPARTURE** | Active/stationary object left | "Car left parking lot" |
| **STATE_CHANGE** | Any state transition | Logging, debugging |
| **LOITERING** | Stationary too long | "Person loitering > 60s" |

### Handling Edge Cases

#### Pre-existing Parked Objects

When the system starts, there may be objects already in the scene (parked cars). These shouldn't trigger arrival events.

**Solution:** Objects that never move go directly to PARKED state without an ARRIVAL event.

```
Camera starts → Car detected → Never moves → TENTATIVE → PARKED (no ARRIVAL)
```

#### Objects That Arrive and Stand Still

A person might walk into frame and stand still. The old logic would mark them as "pre-existing parked" and miss the arrival.

**Solution:** Delayed arrival detection. If an object was PARKED briefly (< 60s) before becoming ACTIVE, generate a delayed ARRIVAL.

```
Person enters → Stands still → TENTATIVE → PARKED (deferred)
                    │
                    ▼
            Starts walking → PARKED → ACTIVE → ARRIVAL (delayed)
```

#### Detection Gaps

Object detection isn't perfect. An object might not be detected for a few frames due to:
- Occlusion
- Low confidence
- Motion blur

**Solution:** The tracker maintains a buffer of lost tracks (`track_buffer=30` frames). If the object reappears within this window, it's matched to the existing track.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `validation_frames` | 10 | Frames to confirm a track |
| `velocity_threshold` | 0.002 | Min velocity to be "moving" (normalized) |
| `displacement_threshold` | 0.02 | Min displacement to be "moved" (normalized) |
| `stationary_seconds` | 10.0 | Time stopped → STATIONARY |
| `parked_seconds` | 300.0 | Time stationary → PARKED (5 min) |
| `lost_seconds` | 5.0 | Time without detection → departed |
| `delayed_arrival_threshold` | 60.0 | Max parked time for delayed arrival |
| `loitering_seconds` | 60.0 | Time stationary → loitering alert |

---

## Integration: The Complete Pipeline

### Live Detection Worker

**Source:** `backend/live_detection_worker.py`

The live detection worker orchestrates all components:

```python
# Simplified flow
for camera in cameras:
    # 1. Extract frame from stream
    frame = extract_frame(camera.rtsp_url)

    # 2. Motion detection (gate)
    motion_result = motion_gate.check(camera.id, frame)

    if motion_result.motion_detected:
        # 3. Object detection
        detections = detector.detect(frame)

        # 4. Filter by configured classes
        detections = [d for d in detections if d.class_name in camera.detect_classes]

        # 5. Update tracker
        tracks = tracker.update(detections)

        # 6. Update FSM and get events
        events = fsm.update(tracks)

        # 7. Handle events
        for event in events:
            if event.event_type == EventType.ARRIVAL:
                save_detection(camera, event)
                send_notification(camera, event)
```

### Multi-GPU Processing

The GPU orchestrator distributes cameras across available GPUs:

```
GPU 0                          GPU 1
├── Camera 1                   ├── Camera 2
├── Camera 3                   ├── Camera 4
├── Camera 5                   ├── Camera 6
└── ...                        └── ...

Each GPU has:
- Its own CUDA context
- Its own YOLO model instance
- Its own motion detector instances
```

### Per-Camera State

Each camera maintains independent state:

```python
camera_state = {
    "motion_detector": GPUBackgroundSubtractor(camera_id),
    "tracker": ByteTracker(),
    "fsm": ObjectStateMachine(),
}
```

This ensures that:
- Background models are camera-specific
- Track IDs are per-camera
- State transitions are independent

---

## Configuration

### Environment Variables

```bash
# Detection
ML_CONFIDENCE_THRESHOLD=0.5    # Default YOLO confidence
LIVE_DETECTION_FPS=1.0         # Frames per second to process
LIVE_DETECTION_COOLDOWN=30.0   # Seconds between notifications

# Model
ML_MODEL_PATH=/models/yolov8l.onnx
```

### Per-Camera Settings (Database)

Each camera can have custom settings:

- `detect_classes`: Which object types to detect (person, car, etc.)
- `detection_threshold`: Override confidence threshold
- `notifications_enabled`: Whether to send alerts

---

## Performance Characteristics

### Typical Latency

| Stage | Latency (GPU) | Latency (CPU) |
|-------|---------------|---------------|
| Motion Detection | ~2ms | ~20ms |
| YOLO Inference | ~15ms | ~500ms |
| Tracking | ~1ms | ~1ms |
| FSM | <1ms | <1ms |
| **Total** | **~20ms** | **~520ms** |

### Memory Usage

| Component | GPU Memory | System Memory |
|-----------|------------|---------------|
| YOLO Model | ~500MB | - |
| MOG2 (per camera) | ~50MB | - |
| Tracker (per camera) | - | ~1MB |
| FSM (per camera) | - | <1MB |

### Scaling

- **Cameras**: Limited by GPU memory and processing time
- **GPUs**: Multiple GPUs supported, cameras distributed automatically
- **Typical capacity**: 8-16 cameras per GPU at 1 FPS

---

## Debugging

### Enable Debug Logging

```python
import logging
logging.getLogger("app.services.ml.tracker").setLevel(logging.DEBUG)
logging.getLogger("app.services.ml.object_fsm").setLevel(logging.DEBUG)
```

### Key Log Messages

```
# Motion detection
Camera 1: motion=2.3%, detected=True, contours=3

# Object detection
YOLO batch[0]: best=0.87, person_max=0.87, car_max=0.12

# Tracking
New track 1: person conf=0.87 pos=(0.5, 0.3)
Track 1 CONFIRMED: person hits=3

# FSM
ARRIVAL: person (track 1)
ARRIVAL (delayed): person (track 2) - was stationary for 1.2s before moving
Deferring ARRIVAL for car (track 3) - not moving yet
DEPARTURE: person (track 1)
```

### Test Tools

```bash
# Analyze video for motion and detections
python tools/analyze_motion_and_objects.py /path/to/video.mp4 --start-time 0:00

# Test FSM tracking on video
python tools/test_fsm_tracking.py /path/to/video.mp4 --start-time 0:00
```

---

## Summary

The detection pipeline transforms raw video into meaningful events:

1. **MOG2** filters out static scenes (efficiency)
2. **YOLO** identifies objects (recognition)
3. **ByteTrack** maintains identity across frames (continuity)
4. **FSM** generates arrival/departure events (semantics)

Together, these components enable intelligent surveillance that:
- Ignores parked cars and static backgrounds
- Tracks people and vehicles as they move
- Generates alerts only for meaningful events
- Handles edge cases like brief stops and detection gaps
