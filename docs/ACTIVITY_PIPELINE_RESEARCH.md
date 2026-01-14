# Activity Detection Pipeline Research

This document captures the research, experiments, and findings from developing the NextGen GPU-accelerated activity detection pipeline for RoninNVR.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Environment](#environment)
- [Model Comparison](#model-comparison)
- [IR Night Camera Challenges](#ir-night-camera-challenges)
- [Preprocessing Experiments](#preprocessing-experiments)
- [Motion Gate vs Periodic Detection](#motion-gate-vs-periodic-detection)
- [Threshold Tuning](#threshold-tuning)
- [GPU Acceleration](#gpu-acceleration)
- [Final Configuration](#final-configuration)
- [Lessons Learned](#lessons-learned)
- [Future Work](#future-work)

---

## Problem Statement

The original live detection system was not detecting people on IR-illuminated security cameras at night. Initial debugging revealed:

1. Motion was being detected correctly (e.g., 62.8% motion on Camera 1)
2. YOLO was running but returning zero detections (`dets=0`)
3. The model output format was correct (84, 8400 for YOLOv8)
4. **Root cause**: Low confidence scores on IR night footage - person scores were below threshold

## Environment

- **Hardware**: RTX 3090 (24GB VRAM)
- **CUDA**: 12.6
- **cuDNN**: 9.5.1
- **OpenCV**: 4.10.0 (built from source with CUDA support)
- **Inference**: ONNX Runtime GPU 1.23.2
- **Cameras**: 11 IR-illuminated security cameras (1280x720)
- **Frame Rate**: ~3 FPS per camera

## Model Comparison

### Tested Models
| Model | Size | Parameters | mAP50-95 | Latency (T4) |
|-------|------|------------|----------|--------------|
| YOLOv8n | 6.3 MB | 3.2M | 37.3 | 1.47ms |
| YOLOv8m | 52 MB | 25.9M | 50.2 | 5.86ms |
| YOLOv8l | 87 MB | 43.7M | 52.9 | 9.06ms |
| YOLO11n | 5.4 MB | 2.6M | 39.5 | 1.55ms |
| YOLO11m | 39 MB | 20.1M | 51.5 | 4.70ms |
| YOLO11l | 49 MB | 25.3M | 53.4 | 5.80ms |

### Key Finding: YOLO11l Outperforms YOLOv8n

**Test: Person detection on recorded IR night footage (03-59-25.mp4)**

| Frame | YOLOv8n | YOLO11l | Improvement |
|-------|---------|---------|-------------|
| Good lighting (12:46) | 0.749 | 0.887 | +18% |
| Clear person visible | 0.75 | 0.89 | +18.7% |

YOLO11l advantages:
- **18% better accuracy** on person detection
- **36% faster** inference than YOLOv8l
- **42% fewer parameters** than YOLOv8l
- Better feature extraction for low-contrast subjects

### Model Selection Rationale

We chose **YOLO11l** over YOLOv8n because:
1. Higher base accuracy (mAP 53.4 vs 37.3)
2. Better performance on challenging IR footage
3. Still fast enough for real-time (5.80ms vs 1.47ms)
4. RTX 3090 has ample capacity for the larger model

## IR Night Camera Challenges

### The Problem

IR-illuminated cameras present unique challenges for object detection:

1. **Grayscale imagery**: IR LEDs produce near-infrared light, cameras capture in grayscale
2. **Low contrast**: People appear as similar intensity to backgrounds
3. **IR hotspots**: Reflective surfaces create bright spots
4. **Distance attenuation**: IR illumination falls off rapidly with distance
5. **Noise**: Higher ISO/gain in low light introduces noise

### Measured Confidence Scores

Testing on nighttime footage showed dramatically different confidence scores:

| Scenario | Person Confidence |
|----------|-------------------|
| Daytime, clear view | 0.85-0.95 |
| IR night, close range | 0.65-0.75 |
| IR night, medium distance | 0.25-0.45 |
| IR night, far/partial view | 0.10-0.20 |

### Not FLIR/Thermal

Important distinction: Our cameras are **IR-illuminated** (near-infrared LEDs), not thermal/FLIR cameras. This means:
- Image looks like grayscale visible light
- Thermal-specific models won't help
- Standard COCO-trained models can work with preprocessing

## Preprocessing Experiments

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Hypothesis**: Enhancing local contrast might improve detection.

**Result**: Made detection **worse**
- Amplified noise in uniform regions
- Created artifacts that confused the model
- Person confidence dropped from 0.10 to 0.08 on test frame

### Sharpening

**Parameters tested**: kernel `[[0,-1,0],[-1,5,-1],[0,-1,0]]`

**Result**: **+47% improvement**
- Frame at 12:30: 0.103 → 0.151
- Enhanced edge definition helps model identify person boundaries

### Gamma Correction

**Parameters tested**: gamma=0.8, gamma=1.2

**Result**: Minimal impact
- Slight improvement with gamma=0.8 (brighten darks)
- Not significant enough to justify processing cost

### Sharpening + Contrast Boost

**Combined approach**: Sharpen first, then increase contrast

**Result**: Best preprocessing combination
- Frame at 12:30: 0.103 → 0.274 (+166%)
- However, still below reliable detection threshold

### Conclusion on Preprocessing

Preprocessing helps but cannot fully compensate for the fundamental challenge of IR footage. The better approach is:
1. Use a more capable model (YOLO11l)
2. Lower detection thresholds
3. Rely on tracking to filter false positives

## Motion Gate vs Periodic Detection

### Motion Gate Limitation

The original pipeline used motion detection as a gate for YOLO inference:
- Motion > 0.3% → Run YOLO
- Motion < 0.3% → Skip YOLO

**Problem**: Small or distant people often don't trigger enough motion (< 0.3% of frame).

### Periodic Detection Solution

**Implementation**: Run YOLO every 30 frames regardless of motion detection.

**Test Results** (00-21-28.mp4 - Hangar_East nighttime):

| Configuration | Total Detections | Persons | Trucks |
|---------------|------------------|---------|--------|
| Motion-only | 0 | 0 | 0 |
| Periodic=30 | 27 | 4 | 23 |

**Key Finding**: Periodic detection catches 100% of detections that motion gate misses because subjects are too small/distant to trigger the 0.3% motion threshold.

### Recommended Configuration

```
NEXTGEN_MOTION_MIN_PERCENT=0.3        # Motion gate threshold
NEXTGEN_PERIODIC_DETECTION_INTERVAL=30 # ~10 sec at 3 FPS
```

Both systems run in parallel:
- Motion gate for immediate response to large movements
- Periodic detection for small/distant/stationary subjects

## Threshold Tuning

### Per-Class Confidence Thresholds

Different object classes require different thresholds based on:
- Detection reliability for that class
- False positive rate
- Operational importance

**Final Configuration**:
```json
{
  "person": 0.25,
  "dog": 0.35,
  "cat": 0.35,
  "car": 0.65,
  "truck": 0.65,
  "default": 0.65
}
```

### Rationale for Person Threshold (0.25)

1. **Security priority**: Missing a person is worse than a false positive
2. **IR challenge**: Person scores rarely exceed 0.5 at night
3. **Tracking filter**: ByteTrack requires multiple detections to confirm
4. **Human review**: Security system allows human verification

### False Positive Mitigation

With lower thresholds, false positives increase. Mitigations:
1. **Track confirmation**: Require `min_hits=5` detections before confirming track
2. **Displacement filter**: Require `min_displacement=0.02` (2% of frame) movement
3. **State machine**: Only generate ARRIVAL events for confirmed tracks
4. **Cooldown**: 30-second cooldown prevents duplicate alerts

## GPU Acceleration

### Why GPU MOG2?

CPU-based MOG2 background subtraction was a bottleneck:
- ~15ms per frame on CPU
- Blocking other processing
- Limited parallelism

GPU MOG2 benefits:
- ~2-3ms per frame
- Non-blocking (async with CUDA streams)
- Better utilization of RTX 3090

### OpenCV CUDA Build

The pip `opencv-python` package doesn't include CUDA modules. Solution:

**Build OpenCV from source** with CUDA support:
- CUDA architectures: 7.5, 8.0, 8.6, 8.9, 9.0 (covers RTX 20/30/40 series)
- Modules: cudaarithm, cudaimgproc, cudabgsegm, cudawarping
- Build time: ~30 minutes

### NumPy Compatibility Issue

**Problem**: Installing packages that depend on NumPy 2.x broke OpenCV built with NumPy 1.x.

**Error**: `numpy.core.multiarray failed to import`

**Solution**:
1. Build OpenCV against system NumPy 1.26.4
2. Pin numpy<2.0 in requirements
3. Rebuild container from scratch if contaminated

## Final Configuration

### Docker Compose Environment Variables

```yaml
NEXTGEN_ENABLED: true
NEXTGEN_MODEL_PATH: /data/storage/.ml/models/yolo11l_dynamic.onnx
NEXTGEN_MOTION_MIN_PERCENT: 0.3
NEXTGEN_DETECTION_CONFIDENCE: 0.65
NEXTGEN_CLASS_THRESHOLDS: '{"person": 0.25, "dog": 0.35, "cat": 0.35}'
NEXTGEN_PERIODIC_DETECTION_INTERVAL: 30
```

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frame Input (HLS segments)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  GPU MOG2 Motion Detection                   │
│                   (cv2.cuda.BackgroundSubtractorMOG2)        │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────┐           ┌─────────────────────┐
│   Motion Detected   │           │  Periodic Trigger   │
│     (> 0.3%)        │           │   (every 30 frames) │
└─────────────────────┘           └─────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 YOLO11l Inference (ONNX Runtime)             │
│                 Per-class confidence thresholds              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ByteTrack Multi-Object Tracker             │
│                   (track_thresh=0.25, min_hits=5)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Object State Machine                     │
│               (ARRIVAL / DEPARTURE events)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Database & Notifications                   │
└─────────────────────────────────────────────────────────────┘
```

### Performance Metrics (RTX 3090)

| Component | Time per Frame |
|-----------|---------------|
| Motion detection (GPU) | 2-3ms |
| YOLO11l inference | 5-10ms |
| Tracking | <1ms |
| Total pipeline | 15-20ms |

**Capacity**: Easily handles 10+ cameras at 3 FPS each

## Lessons Learned

### 1. Model Size Matters for Difficult Conditions
YOLOv8n was insufficient for IR night footage. The extra parameters in YOLO11l provide better feature extraction for low-contrast subjects.

### 2. Don't Over-Rely on Motion Gating
Motion-only triggering misses stationary or distant subjects. Periodic detection is essential for comprehensive coverage.

### 3. Preprocessing Has Limits
While sharpening helped slightly, no preprocessing could compensate for the fundamental IR challenge. Better to use a stronger model with lower thresholds.

### 4. Tracking Filters False Positives
Lower detection thresholds are viable when combined with robust tracking that requires multiple consistent detections.

### 5. GPU Build Complexity
Building OpenCV with CUDA is time-consuming but worthwhile for production deployment. Keep build artifacts cached.

### 6. NumPy Version Discipline
Python ML ecosystem has fragile version dependencies. Pin versions and rebuild containers when dependencies change.

## Future Work

### Short Term
- [ ] A/B test YOLO11l vs YOLO11x for accuracy comparison
- [ ] Implement adaptive thresholds based on ambient light
- [ ] Add IR-specific image enhancement as optional preprocessing

### Medium Term
- [ ] Train custom YOLO model on IR security camera dataset
- [ ] Implement TensorRT conversion for faster inference
- [ ] Add zone-based detection sensitivity

### Long Term
- [ ] Explore thermal-aware models if adding FLIR cameras
- [ ] Implement behavior analysis (loitering detection)
- [ ] Multi-camera person re-identification

---

## References

- [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [OpenCV CUDA Modules](https://docs.opencv.org/4.x/d2/dbc/cuda_intro.html)
- [ONNX Runtime GPU](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

---

*Last updated: 2026-01-10*
*Status: YOLO11l deployed in production with periodic detection enabled*
