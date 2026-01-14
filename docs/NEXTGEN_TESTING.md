# NextGen Pipeline Testing Guide

## Current Status (2026-01-10)

The NextGen GPU-accelerated motion detection pipeline is running in production with YOLO11l.

### Recent Changes
- **Model upgraded to YOLO11l** - 18% better accuracy on person detection (0.89 vs 0.75)
- **Person threshold lowered to 0.25** - Better detection in IR night conditions
- **OpenCV with CUDA** - GPU-accelerated MOG2 background subtraction

## What's New

### Core Pipeline Files
- `backend/app/services/ml/gpu_motion.py` - GPU MOG2 motion detection
- `backend/app/services/ml/gpu_orchestrator.py` - Main pipeline orchestrator
- `backend/app/services/ml/tensorrt_inference.py` - YOLO inference with per-class thresholds
- `backend/app/services/ml/tracker.py` - ByteTrack multi-object tracking
- `backend/app/services/ml/object_fsm.py` - Object state machine (ARRIVAL/DEPARTURE events)

### Modified Files
- `backend/app/config.py` - Added nextgen_* settings
- `backend/live_detection_worker.py` - GPU pipeline is now default (use --legacy to disable)

### Key Features
1. **Per-class confidence thresholds**: Lower threshold for people (0.45) vs vehicles (0.65)
2. **Periodic detection**: Runs YOLO every 30 frames regardless of motion to catch small/distant objects
3. **GPU-accelerated motion detection**: MOG2 runs on GPU
4. **Multi-object tracking**: ByteTrack maintains object IDs across frames
5. **State machine**: Generates ARRIVAL/DEPARTURE events

## Running the NextGen Worker

### On Host (Recommended for Testing)

```bash
cd /workspace/ronin-nvr/backend

# Activate your Python environment
source venv/bin/activate  # or however you activate it

# Run with GPU pipeline (default)
python live_detection_worker.py --verbose

# Or use legacy CPU pipeline
python live_detection_worker.py --legacy --verbose
```

### Configuration (Environment Variables)

```bash
# GPU pipeline is enabled by default (set to false to disable)
NEXTGEN_ENABLED=true

# Model path (YOLO11l recommended for IR cameras)
NEXTGEN_MODEL_PATH=/opt3/ronin/ml_models/yolo11l_dynamic.onnx

# Motion detection
NEXTGEN_MOTION_MIN_PERCENT=0.3  # Min % of frame for motion trigger

# Detection thresholds
NEXTGEN_DETECTION_CONFIDENCE=0.65  # Default threshold
NEXTGEN_CLASS_THRESHOLDS='{"person": 0.25, "dog": 0.35, "cat": 0.35}'  # Lower for IR

# Periodic detection (bypasses motion gate)
NEXTGEN_PERIODIC_DETECTION_INTERVAL=30  # Every 30 frames (~10 sec at 3 FPS)

# Tracking
NEXTGEN_TRACK_MIN_HITS=5  # Min detections to confirm track
NEXTGEN_TRACK_MIN_DISPLACEMENT=0.02  # Min movement to confirm
```

### Via Docker Compose

The GPU pipeline is enabled by default in docker-compose.yml. To use the legacy CPU pipeline:

```yaml
live-detection:
  command: python live_detection_worker.py --legacy
  environment:
    - NEXTGEN_ENABLED=false
```

## Testing Results

### Video Test: 00-21-28.mp4 (Hangar_East nighttime)

| Config | Detections | Persons | Trucks |
|--------|-----------|---------|--------|
| Motion only | 0 | 0 | 0 |
| Periodic=30 | 27 | 4 | 23 |

**Key finding**: Periodic detection catches people that motion gate misses because they're too small/distant to trigger the 0.3% motion threshold.

### Visualizations

Test output with annotated frames saved to:
```
/workspace/ronin-nvr/backend/tools/detection_output/
├── montage.jpg           # 4x4 grid of detection frames
├── frame_000957_9.6s.jpg # Frame with 2 people detected
└── ... (27 total frames)
```

## Performance Expectations (RTX 3090)

- **10 cameras at 3 FPS**: Should handle easily
- **YOLO inference**: ~5-10ms per frame
- **Motion detection**: ~2-3ms per frame
- **Total pipeline**: ~15-20ms per frame

The 3090 has 24GB VRAM and is significantly faster than the 2070 used during development.

## Monitoring

Watch the logs for:
```
[INFO] NextGen pipeline initialized: 1 GPU(s), model=/opt3/ronin/ml_models/yolov8n_dynamic.onnx
[INFO] Detection on Camera_Name: person, truck [nextgen]
```

Check database for new detections:
```sql
SELECT * FROM detections
WHERE model_name = 'nextgen'
ORDER BY detected_at DESC
LIMIT 20;
```

## Troubleshooting

### "TensorRT not available - falling back to ONNX Runtime"
This is fine for testing. ONNX Runtime CUDA is still GPU-accelerated.

### No detections appearing
1. Check if cameras are recording (segments in `.streams/`)
2. Verify model path exists
3. Check `--verbose` output for motion/detection activity

### High GPU memory usage
Reduce batch size or number of cameras if needed.
