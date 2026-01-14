# NextGen Motion Detection - Session Context

**Last Updated:** 2026-01-09
**Branch:** `feature/nextgen-motion-detection`
**Task:** TASK-32
**Status:** ALL PHASES COMPLETE + WORKER INTEGRATION

## Implementation Summary

All 6 phases of the nextgen motion/object detection pipeline have been implemented and tested.
Worker integration is complete - GPU pipeline is enabled by default. Use `--legacy` flag to disable.

### Phase 1: GPU MOG2 Background Subtraction
- `backend/app/services/ml/gpu_motion.py` - GPU-accelerated MOG2
- `backend/app/services/ml/gpu_frame_manager.py` - GpuMat memory pooling
- Updated `motion_gate.py` with `backend="gpu"` option

**Performance:** 3.17ms per frame at 720p (vs ~8ms CPU)

### Phase 2: TensorRT Inference Pipeline
- `backend/app/services/ml/tensorrt_inference.py` - ONNX Runtime with CUDA EP
- `backend/app/services/ml/nvdec_extractor.py` - Hardware video decode
- `backend/tools/convert_to_tensorrt.py` - Model conversion tool
- Dynamic batch support for multi-camera processing

**Performance:** ~4ms per frame for YOLO detection (after vectorization optimizations)

### Phase 3: ByteTrack Multi-Object Tracking
- `backend/app/services/ml/tracker.py` - Kalman filter + IoU matching

**Performance:** 0.04ms per frame

### Phase 4: Finite State Machine
- `backend/app/services/ml/object_fsm.py` - Object lifecycle management
- States: TENTATIVE → ACTIVE → STATIONARY → PARKED
- Events: ARRIVAL, DEPARTURE, LOITERING

**Performance:** 0.01ms per frame

### Phase 5: GPU Orchestrator
- `backend/app/services/ml/gpu_orchestrator.py` - Multi-GPU pipeline management
- Camera-to-GPU assignment
- Per-camera trackers and FSMs

### Phase 6: Integration
- `backend/tools/test_full_pipeline.py` - End-to-end testing

## Performance Results

### After Optimizations (2026-01-09)

Key optimizations applied:
1. **CUDA EP Fix** - Ensured ONNX Runtime uses GPU, not CPU fallback
2. **Vectorized Postprocessing** - 43x faster (39ms → 0.9ms per frame)
3. **Dynamic Batch Model** - Exported YOLOv8n with variable batch size
4. **Optimized Batch Preprocessing** - Pre-allocated numpy arrays

### Multi-Camera Performance (8 cameras)
- **Direct YOLO (worst case):** 224 FPS total (28 FPS per camera)
- **With Motion Gating:** ~340 FPS total (~42 FPS per camera)
- **Batch Inference Speedup:** 1.42x over sequential

### Component Timing (Optimized)
| Component | Mean | Notes |
|-----------|------|-------|
| Motion (GPU MOG2) | ~3ms | P99: 3.47ms |
| Detection (YOLO) | ~4ms | Per frame, batched |
| Postprocessing | 0.9ms | Vectorized (was 39ms) |
| Tracking | 0.04ms | Per camera |
| FSM | 0.01ms | Per camera |

### Motion Gate Reliability
- **100% recall** - Catches all frames containing objects
- **14x faster** than running YOLO on every frame
- Effective false positive filter when combined with tuned confidence (0.65)

## Files Created

```
backend/app/services/ml/
├── gpu_motion.py           # GPU MOG2 background subtraction
├── gpu_frame_manager.py    # GpuMat memory pooling
├── tensorrt_inference.py   # TensorRT/ONNX detection
├── nvdec_extractor.py      # Hardware video decode
├── tracker.py              # ByteTrack implementation
├── object_fsm.py           # Finite state machine
└── gpu_orchestrator.py     # Multi-GPU management

backend/tools/
├── test_gpu_motion.py          # GPU motion tests
├── test_tensorrt_inference.py  # Detection tests
├── test_tracker.py             # Tracking tests
├── test_fsm.py                 # FSM tests
├── test_full_pipeline.py       # End-to-end tests
├── convert_to_tensorrt.py      # Model conversion
├── visual_benchmark.py         # Visual benchmarking with annotated output
└── benchmark_batch_inference.py # Multi-camera batch performance tests
```

## Worker Integration (Completed)

### How to Enable NextGen Pipeline

```bash
# Option 1: CLI flag
./live_detection_worker.py  # GPU pipeline is default
./live_detection_worker.py --legacy  # Use legacy CPU pipeline

# Option 2: Environment variable
NEXTGEN_ENABLED=true ./live_detection_worker.py

# Option 3: Config setting (in .env)
NEXTGEN_ENABLED=true
```

### Configuration (app/config.py)
All nextgen settings are prefixed with `nextgen_`:
- `nextgen_enabled` - Enable GPU pipeline (default: true)
- `nextgen_model_path` - Path to ONNX model
- `nextgen_detection_confidence` - YOLO confidence (default: 0.65)
- `nextgen_track_min_hits` - Frames to confirm track (default: 5)
- `nextgen_track_min_displacement` - Min movement to confirm (default: 0.02)

### Files Modified
- `backend/app/config.py` - Added nextgen_* settings
- `backend/live_detection_worker.py` - Integrated GPUOrchestrator

## Next Steps (Remaining)

1. **Database Migration** - Add tracking columns to detections table (track_id, state)
2. **A/B Testing** - Run old and new pipelines in parallel for comparison
3. **Dashboard Updates** - Show tracking info in frontend

## Key Improvements Over Original

| Metric | Original | NextGen |
|--------|----------|---------|
| Analysis FPS | 3 | 15+ per camera |
| Motion Detection | Frame diff | GPU MOG2 |
| Object Tracking | None | ByteTrack |
| State Management | None | FSM |
| False Positive Handling | High | Suppressed (parked vehicles) |

## Environment

- **GPU:** RTX 3090 (24GB VRAM, compute 8.6)
- **CUDA:** 12.6
- **OpenCV:** 4.10.0 with CUDA
- **TensorRT:** 10.7.0
- **Python:** 3.12
