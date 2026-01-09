# NextGen Motion Detection - Session Context

**Last Updated:** 2026-01-09
**Branch:** `feature/nextgen-motion-detection`
**Task:** TASK-32

## Current Status

**Phase 0: Development Environment** - IN PROGRESS (awaiting verification)

### Completed
- [x] Created `docker/Dockerfile.nextgen-dev` with CUDA 12.6, TensorRT 10.x, OpenCV CUDA
- [x] Created `docker/docker-compose.nextgen.yml` with dual GPU passthrough
- [x] Created `backend/tools/benchmark_pipeline.py` for performance testing
- [x] Committed initial Phase 0 files (commit `183411c`)

### Pending
- [ ] Verify Docker build completes successfully
- [ ] Verify GPU access (nvidia-smi)
- [ ] Verify OpenCV CUDA (`cv2.cuda.getCudaEnabledDeviceCount()`)
- [ ] Verify TensorRT import
- [ ] Run initial benchmark

## Next Phase

**Phase 1: GPU-Accelerated MOG2 Background Subtraction**

Create these files:
1. `backend/app/services/ml/gpu_motion.py` - GPU MOG2 background subtractor
2. `backend/app/services/ml/gpu_frame_manager.py` - GpuMat memory management
3. Update `motion_gate.py` with `backend="gpu"` option

## Key Files

| File | Purpose |
|------|---------|
| `docs/NEXTGEN_PLAN.md` | Full implementation plan |
| `docs/research/Motion_and_Object_Detection_Research.pdf` | Research document |
| `NEXTGEN_MOTION.md` | Original design goals |
| `backend/tools/benchmark_pipeline.py` | Performance testing |
| `backend/live_detection_worker.py` | Current detection worker (to extend) |
| `backend/app/services/ml/motion_gate.py` | Current motion detection (to enhance) |

## Architecture Summary

```
RTSP Stream → NVDEC Decode → GPU MOG2 → TensorRT YOLO → ByteTrack → FSM → Events
                (VRAM)        (motion)    (detection)    (tracking)  (state)
```

## To Resume Work

When starting a new Claude Code session inside Docker:

```
I'm continuing the nextgen motion detection refactor (TASK-32).
Please read docs/NEXTGEN_CONTEXT.md and docs/NEXTGEN_PLAN.md for context.
We just verified the Docker environment works. Ready to proceed to Phase 1.
```

## Environment Verification Commands

```bash
# Test GPU access
nvidia-smi

# Test OpenCV CUDA
python -c "import cv2; print(f'OpenCV: {cv2.__version__}'); print(f'CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}')"

# Test TensorRT
python -c "import tensorrt; print(f'TensorRT: {tensorrt.__version__}')"

# Test MOG2 GPU
python -c "import cv2; bg = cv2.cuda.createBackgroundSubtractorMOG2(); print('GPU MOG2: OK')"

# Run benchmark
python backend/tools/benchmark_pipeline.py --scan-storage --limit 1 --compare
```
