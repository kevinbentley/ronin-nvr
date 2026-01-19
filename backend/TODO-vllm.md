# VLLM Activity Characterization - Implementation Plan

## Overview
A backend service that uses a Vision LLM to characterize detected activities, distinguishing normal activities (UPS delivery) from suspicious behavior (prowling, peering into windows).

## Architecture

```
Detection Triggered
       ↓
Activity Characterization Worker
       ↓
Extract 4 frames (1 second apart)
       ↓
Create 2x2 mosaic image
       ↓
Send to VLLM with:
  - Camera baseline description
  - Analysis prompt
       ↓
Parse response → Store in detection.llm_description
       ↓
Log result to Docker output
```

## Phases

### Phase 1: Database & Configuration
- [x] Add `scene_description` field to Camera model
- [x] Create Alembic migration
- [x] Add VLLM configuration to settings/env

### Phase 2: Core Services
- [x] Create VLLM client service (`app/services/vllm/client.py`)
- [x] Create frame mosaic utility (`app/services/vllm/mosaic.py`)
- [x] Create activity characterization service (`app/services/vllm/characterization.py`)

### Phase 3: Worker Implementation
- [x] Create activity characterization worker (`activity_worker.py`)
- [x] Implement detection monitoring loop
- [x] Implement frame extraction from recordings
- [x] Integrate mosaic creation and VLLM analysis

### Phase 4: Manual Processing Mode
- [x] Add CLI argument for manual video processing
- [x] Query detections for specified recording
- [x] Process detections with VLLM analysis

### Phase 5: Testing & Integration
- [x] Test with sample video
- [x] Verify VLLM endpoint connectivity
- [x] Test mosaic generation
- [x] End-to-end test with real detection

## Configuration

```env
# VLLM Settings
VLLM_ENDPOINT=http://192.168.1.125:9001
VLLM_MODEL=default
VLLM_TIMEOUT=60
ACTIVITY_ANALYSIS_ENABLED=true
ACTIVITY_FRAME_COUNT=4
ACTIVITY_FRAME_INTERVAL_MS=1000
```

## Concern Levels

The VLLM will classify activities into:
- **NONE**: No activity detected / false positive
- **LOW**: Normal expected activity (delivery, mail, resident)
- **MEDIUM**: Unusual but not threatening (unfamiliar person, late night activity)
- **HIGH**: Suspicious behavior (peering in windows, checking doors)
- **EMERGENCY**: Immediate threat (break-in attempt, assault)

## Files to Create/Modify

### New Files
- `backend/app/services/vllm/__init__.py`
- `backend/app/services/vllm/client.py`
- `backend/app/services/vllm/mosaic.py`
- `backend/app/services/vllm/characterization.py`
- `backend/activity_worker.py`
- `backend/alembic/versions/xxx_add_camera_scene_description.py`

### Modified Files
- `backend/app/models/camera.py` (add scene_description field)
- `backend/.env.example` (add VLLM config)
