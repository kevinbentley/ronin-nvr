# Next Generation Motion/Object Detection Refactor Plan

## Overview

Refactor the motion/object detection system based on research document guidance to dramatically improve reliability while maintaining performance for 16 cameras on dual RTX 2070 GPUs.

**Branch name:** `feature/nextgen-motion-detection`
**Task:** TASK-32
**Priority:** Dev Docker first, then phased implementation
**Framework:** Python + TensorRT (confirmed)

## Current Problems
- 3 FPS analysis causes "teleportation effects" breaking tracker continuity
- Frame differencing lacks temporal memory for complex backgrounds
- High false positive rate from rain, swaying trees, lighting changes
- Parked vehicles trigger constant false alerts
- Detection "flicker" from yolo8n model instability

## Target Architecture: Cascaded Hybrid Pipeline

```
RTSP Stream → NVDEC Decode → GPU MOG2 → TensorRT YOLO → ByteTrack → FSM → Events
                (VRAM)        (motion)    (detection)    (tracking)  (state)
```

## Key Architectural Decision

**Enhanced Python/OpenCV + TensorRT** (not DeepStream)

Rationale: Existing codebase is well-structured Python with ~60% code reuse potential. Dual RTX 2070s provide sufficient headroom for Python+TensorRT to achieve 15+ FPS. DeepStream would require complete rewrite with high risk.

---

## Phase 0: Development Environment (PRIORITY - DO FIRST)

Create isolated Docker environment for safe development with GPU access.

### Tasks
1. Create `docker/Dockerfile.nextgen-dev` based on `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04`
   - Add TensorRT 8.6+, OpenCV with CUDA support
   - Add ByteTrack dependencies (`lap`, `scipy`, `cython-bbox`)
   - Python 3.11 + pip dependencies from requirements.txt
2. Create `docker/docker-compose.nextgen.yml`
   - Dual GPU passthrough (`device_ids: ["0", "1"]`)
   - Mount repository at `/workspace/ronin-nvr`
   - Mount `/opt3/ronin/storage` read-only for test videos
   - Mount `/opt3/ronin/ml_models` for model files
   - Network isolation from production services
3. Create `backend/tools/benchmark_pipeline.py`
   - Load test videos from storage
   - Measure FPS, latency, GPU memory, CPU usage
   - Compare old vs new pipeline performance

### Files to Create
- `docker/Dockerfile.nextgen-dev`
- `docker/docker-compose.nextgen.yml`
- `backend/tools/benchmark_pipeline.py`

### Verification
```bash
# Build and test dev environment
docker compose -f docker/docker-compose.nextgen.yml build
docker compose -f docker/docker-compose.nextgen.yml run --rm dev nvidia-smi
docker compose -f docker/docker-compose.nextgen.yml run --rm dev python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```

---

## Phase 1: GPU-Accelerated MOG2 Background Subtraction

### Tasks
1. Create `backend/app/services/ml/gpu_motion.py`
   - `GPUBackgroundSubtractor` class using `cv2.cuda.createBackgroundSubtractorMOG2`
   - Morphological filtering (erosion + dilation) for rain speckle removal
   - Per-camera background model management
2. Create `backend/app/services/ml/gpu_frame_manager.py`
   - GpuMat memory pooling
   - Zero-copy frame management
3. Update `motion_gate.py` with `backend="gpu"` option for backward compatibility

### Key Parameters
- `history`: 500 frames
- `varThreshold`: 16.0
- `detectShadows`: True
- Erosion kernel: 3x3, Dilation kernel: 5x5

---

## Phase 2: TensorRT Inference Pipeline

### Tasks
1. Create `backend/app/services/ml/tensorrt_inference.py`
   - `TensorRTDetector` class with FP16 precision
   - Dynamic batching (batch size 8-16)
   - Model warmup on initialization
2. Create `backend/app/services/ml/nvdec_extractor.py`
   - Hardware decode using `cv2.cudacodec.VideoReader`
   - Frames decode directly to GpuMat
3. Create `backend/tools/convert_to_tensorrt.py`
   - Convert ONNX models to TensorRT engines
   - Support FP16/FP32 precision
4. Update `detection_service.py` with `backend="tensorrt"` option

### Model Upgrade
- Upgrade from `yolov8n` to `yolov8s` (Small) for stability
- Consider `yolo11s` for better small object detection

---

## Phase 3: ByteTrack Multi-Object Tracking

### Tasks
1. Create `backend/app/services/ml/tracker.py`
   - `ByteTracker` class with Kalman filter and IoU matching
   - `TrackedObject` dataclass with track_id, velocity, state
   - Persistence buffer: 90 frames (3 seconds at 30fps)
2. Database migration for tracking columns:
   ```sql
   ALTER TABLE detections ADD COLUMN track_id INTEGER;
   ALTER TABLE detections ADD COLUMN velocity_x FLOAT;
   ALTER TABLE detections ADD COLUMN velocity_y FLOAT;
   ALTER TABLE detections ADD COLUMN object_state VARCHAR(20);
   ```

### Key Parameters
- `track_buffer`: 90 frames
- `match_thresh`: 0.8
- `track_thresh`: 0.5

---

## Phase 4: Finite State Machine for Object Lifecycle

### Tasks
1. Create `backend/app/services/ml/object_fsm.py`
   - `ObjectState` enum: TENTATIVE, ACTIVE, STATIONARY, PARKED
   - `ObjectStateMachine` class managing state transitions
   - Event generation: arrival, departure, state_change
2. Create dynamic exclusion masks for PARKED objects
   - Feed bbox back to MOG2 as ignore region

### State Transitions
```
TENTATIVE (< 10 frames)
    → ACTIVE (velocity > 2 px/frame, validated)
        → STATIONARY (velocity ~0 for 10 seconds)
            → PARKED (stationary for 5 minutes)
                → ACTIVE (movement detected) → exit frame = "Departure"
```

### Notification Rules
- ACTIVE → triggers notification (arrival)
- STATIONARY/PARKED → suppress notifications
- PARKED → ACTIVE → exit = departure notification

---

## Phase 5: Dual-GPU Split-Stream Architecture

### Tasks
1. Create `backend/app/services/ml/gpu_orchestrator.py`
   - `GPUPipeline` class: owns all components for one GPU
   - `GPUOrchestrator`: manages camera-to-GPU assignments
2. Static sharding:
   - GPU 0: Cameras 1-8
   - GPU 1: Cameras 9-16
3. Configuration in Settings:
   - `gpu_camera_assignments: dict[int, list[int]]`

---

## Phase 6: Integration

### Tasks
1. Create `backend/live_detection_worker_v2.py`
   - Integrates all new components
   - Same database interface and pg_notify mechanism
2. Add feature flags to `ml_settings`:
   - `nextgen_motion_enabled`
   - `nextgen_tensorrt_enabled`
   - `nextgen_tracking_enabled`
   - `nextgen_fsm_enabled`
3. Update `backend/Dockerfile.worker` with TensorRT dependencies

---

## Database Schema Changes

### Migration: `YYYYMMDD_nextgen_tracking.py`

```python
# detections table additions
track_id = Column(Integer, nullable=True, index=True)
velocity_x = Column(Float, nullable=True)
velocity_y = Column(Float, nullable=True)
object_state = Column(String(20), nullable=True, index=True)

# ml_settings table additions
nextgen_motion_enabled = Column(Boolean, default=False)
nextgen_tensorrt_enabled = Column(Boolean, default=False)
nextgen_tracking_enabled = Column(Boolean, default=False)
nextgen_fsm_enabled = Column(Boolean, default=False)
fsm_validation_frames = Column(Integer, default=10)
fsm_velocity_threshold = Column(Float, default=2.0)
fsm_stationary_seconds = Column(Float, default=10.0)
fsm_parked_minutes = Column(Float, default=5.0)
```

---

## New Files Summary

```
backend/
  app/services/ml/
    gpu_motion.py              # GPU MOG2 background subtraction
    gpu_frame_manager.py       # GpuMat memory management
    tensorrt_inference.py      # TensorRT detection
    nvdec_extractor.py         # Hardware video decode
    tracker.py                 # ByteTrack implementation
    object_fsm.py              # Finite State Machine
    gpu_orchestrator.py        # Dual-GPU management

  live_detection_worker_v2.py  # New nextgen worker

  tools/
    benchmark_pipeline.py      # Performance testing
    convert_to_tensorrt.py     # Model conversion
    validate_detections.py     # Visual validation

docker/
  Dockerfile.nextgen-dev       # Development environment
  docker-compose.nextgen.yml   # Dev compose file
```

---

## Critical Files to Modify

| File | Changes |
|------|---------|
| `backend/app/services/ml/detection_service.py` | Add tensorrt backend option |
| `backend/app/services/ml/motion_gate.py` | Add gpu backend option |
| `backend/app/models/detection.py` | Add tracking columns |
| `backend/app/models/ml_settings.py` | Add feature flags and FSM config |
| `backend/Dockerfile.worker` | Add TensorRT dependencies |

---

## Verification Strategy

1. **Unit tests** for each new module
2. **Benchmark suite** comparing old vs new pipeline:
   - FPS throughput
   - Latency distribution (p50, p95, p99)
   - GPU memory usage
3. **Visual validation** using annotated video output
4. **A/B testing** with feature flags on production recordings

### Test Videos
- Use existing recordings in `/opt3/ronin/storage/`
- Create synthetic test cases for rain, parked cars, lighting changes

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Analysis FPS | 3 | 15+ |
| End-to-end latency | 2-5s | <500ms |
| Rain false positives | High | 90% reduction |
| Parked car false positives | High | 95% reduction |
| Detection flicker | Frequent | <5% frames |

---

## Risk Mitigation

- **Feature flags** enable gradual rollout
- **Backward compatibility** - old worker unchanged
- **Additive schema changes** - nullable columns only
- **Docker isolation** - dev environment separate from production
