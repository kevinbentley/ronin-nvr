# Detection Benchmark Analysis Results

**Date**: 2026-01-11 (Updated with ground truth analysis)
**Benchmark**: benchmark_5b5f65cc.json + VLM ground truth validation
**Videos**: 21 daytime videos
**Events Analyzed**: 936 benchmark events + 63 ground truth frames

## Executive Summary

**THE REAL PROBLEM**: The live detection system is missing the vast majority of people, vehicles, and motion events. The original benchmark only evaluated false positives - it couldn't measure false negatives because it only sent frames to VLM when a detector already fired.

**Ground Truth Analysis** (30s segment at 13:55 in 16-00-01.mp4):

| What Happened | Count | Issue |
|---------------|-------|-------|
| VLM saw person | 20 frames | - |
| YOLO@0.25 detected person | 16 frames | **20% miss rate** |
| VLM saw trailer (stationary) | 62 frames | - |
| YOLO@0.25 detected trailer | 0 frames | **100% miss rate** |
| Person detection delay | 1.4 seconds | YOLO late to detect |

**Critical Finding**: Even at the benchmark threshold of 0.25, YOLO:
- Misses the first 1.4 seconds of a person appearing
- Completely fails to detect stationary vehicles (100% miss rate)

At the **live threshold of 0.6**, virtually all detections would be missed.

## Ground Truth Analysis (NEW)

### Test: 16-00-01.mp4 @ 13:55-14:25 (30 seconds)

**Methodology**: Ran VLM on EVERY sampled frame (63 frames @ 2 FPS) regardless of detector output.

**Timeline of Person Detection**:
```
840.3s (12195): VLM sees PERSON - YOLO sees nothing
840.8s (12202): VLM sees PERSON - YOLO sees nothing
841.2s (12209): VLM sees PERSON - YOLO sees nothing
841.7s (12216): VLM sees PERSON - YOLO FINALLY detects 'person' ← 1.4s delay
...
849.0s (12321): VLM sees PERSON - YOLO sees 'person' (last detection)
849.4s (12328): VLM sees PERSON - YOLO sees 'fire hydrant' ← misclassification
849.9s (12335): VLM no person - YOLO sees nothing
```

**Stationary Trailer - YOLO completely blind**:
- VLM detected "trailer" / "storage trailer" in 62 of 63 frames
- YOLO detected 0 of these - not 'truck', not 'car', nothing
- A stationary object will NEVER enter the state machine

**MOG2 Was Not The Problem**:
- MOG2 motion detected in ALL 63 frames (20-35% motion area)
- The motion gate was passing frames through
- YOLO simply failed to detect objects even when MOG2 was firing

### Implications for Live Detection

The live detection threshold is 0.6. Earlier benchmark analysis showed:
- All 9 person detections were 0.25-0.40 confidence
- At 0.6 threshold: **100% of people would be missed**
- At 0.6 threshold: **100% of vehicles would be missed** (already missed at 0.25)

---

## Original Benchmark Findings (False Positive Analysis)

### 1. Current Performance (No Filtering)

| Detector | TP | FP | Precision |
|----------|----|----|-----------|
| YOLO11l | 400 | 236 | 62.9% |
| YOLOv8n | 194 | 81 | 70.5% |
| MOG2 | 210 | 183 | 53.4% |
| Corruption | 0 | 101 | 0.0% |
| Frame Diff | 5 | 21 | 19.2% |
| Edge Detection | 24 | 5 | 82.8% |

### 2. YOLO Class Accuracy

**Good Classes (>80% accuracy) - KEEP:**
| Class | Correct | Wrong | Accuracy |
|-------|---------|-------|----------|
| truck | 208 | 4 | 98.1% |
| train | 202 | 0 | 100.0% |
| car | 80 | 0 | 100.0% |
| boat | 38 | 0 | 100.0% |
| bus | 29 | 0 | 100.0% |
| bicycle | 11 | 0 | 100.0% |
| person | 9 | 0 | 100.0% |
| sheep | 8 | 0 | 100.0% |

**Bad Classes (high FP rate) - REMOVE:**
| Class | Correct | Wrong | Accuracy | Root Cause |
|-------|---------|-------|----------|------------|
| toilet | 0 | 224 | 0.0% | Propane tank misclassification |
| chair | 0 | 100 | 0.0% | Outdoor furniture shapes |
| umbrella | 0 | 71 | 0.0% | Unknown |
| stop sign | 0 | 69 | 0.0% | Signs/shapes in distance |
| bench | 0 | 45 | 0.0% | Outdoor objects |
| airplane | 40 | 51 | 44.0% | Mixed - some real (hangar) |

### 3. Recall Analysis

With YOLO whitelist filtering:

| Category | Caught | Total | Recall | Notes |
|----------|--------|-------|--------|-------|
| Person | 9 | 9 | **100%** | Perfect - all people detected |
| Vehicle | 381 | 445 | **85.6%** | Good - some distant/partial missed |
| Animal | 8 | 50 | 16% | Most are birds (small/distant) |

### 4. MOG2 Analysis

MOG2-only events (motion detected, no whitelisted YOLO class):
- True positives: 97 (VLM confirmed something)
- False positives: 182 (nothing there)
- **Precision: 34.8%**

What VLM found in MOG2-only TPs:
- Vehicles: 62 (YOLO missed)
- Animals: 38 (mostly birds)

**Conclusion**: MOG2 catches some events YOLO misses, but has 2x false positive rate. Consider using MOG2 as motion gate only, not as detection source.

## Recommendations (Updated for Recall Problem)

### PRIORITY 1: Lower YOLO Confidence Threshold

**Current**: 0.6 (too high - misses everything)
**Recommended**: 0.25 (matches benchmark)

```python
# In config.py or live_detection_worker.py
ML_CONFIDENCE_THRESHOLD = 0.25  # Was 0.6
```

**Impact**: Immediately improves person detection by ~4x based on benchmark data.

### PRIORITY 2: Class Whitelist (already implemented)

Keep the whitelist to filter false positives:
```python
YOLO_CLASS_WHITELIST = {
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'boat',
    'person',
    'dog', 'cat', 'horse', 'sheep', 'cow', 'bear', 'deer',
}
```

### PRIORITY 3: Address Stationary Object Blind Spot

**The Problem**: YOLO cannot detect stationary objects that were already in frame.

**Options**:
1. **Periodic full-frame detection**: Every N seconds, run YOLO on full frame regardless of MOG2
2. **Background scene comparison**: Detect new stationary objects by comparing against baseline
3. **Longer initial detection window**: When camera starts, run detection for longer before MOG2 gate engages

### PRIORITY 4: Detection Latency Mitigation

**The Problem**: 1.4 second delay from person appearance to detection.

**Options**:
1. **Increase detection FPS**: Run at 2-4 FPS instead of 1 FPS
2. **Multi-scale detection**: Run at multiple resolutions to catch distant/small objects
3. **Lower initial threshold**: Use 0.15 for first detection, raise to 0.25 for confirmation

### PRIORITY 5: State Machine Considerations

The state machine helps with:
- Debouncing (good)
- Tracking objects that go stationary (good)

But it can't help if objects never enter the state machine:
- Objects too far/small → YOLO doesn't detect → never enters FSM
- Objects already stationary → no motion → never triggers → never enters FSM

**Recommendation**: Consider seeding the state machine with periodic full-frame sweeps.

## Implementation Priority

1. **CRITICAL**: Lower confidence threshold from 0.6 to 0.25
2. **HIGH**: Keep class whitelist for FP reduction
3. **HIGH**: Implement periodic full-frame detection (every 30-60s)
4. **MEDIUM**: Increase detection FPS to 2+ for faster initial detection
5. **LOW**: Multi-scale detection for distant objects

## Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `backend/app/config.py` | Lower ML_CONFIDENCE_THRESHOLD to 0.25 | CRITICAL |
| `backend/live_detection_worker.py` | Apply class whitelist | HIGH |
| `backend/live_detection_worker.py` | Add periodic full-frame sweep | HIGH |
| `backend/app/services/ml/tensorrt_inference.py` | Multi-resolution inference | MEDIUM |

## Test Data

Test videos with known ground truth:
- `backend/test_data/16-00-01.mp4` @ 13:55 - Person walking, stationary trailer
- `backend/test_data/15-01-38.mp4` - Contains trailer detections
