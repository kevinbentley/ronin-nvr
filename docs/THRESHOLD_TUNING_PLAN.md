# Detection Threshold Tuning Plan

## Objective

Use VLM-labeled benchmark results to determine if MOG2/YOLO can produce acceptable detection results with threshold tuning alone, and if so, what thresholds to use.

## Scope

- **In scope**: Daytime video only (UTC 15:00-23:59, 00:00-01:00)
- **Out of scope**: Nighttime/IR video (will be addressed with fine-tuned YOLO model later)

## Current Problem

From benchmark logs, we're seeing:
- MOG2 triggering constantly on environmental motion (wind, clouds, shadows)
- YOLO false positives on static objects (propane tank → "toilet" @ 0.92, "airplane" @ 0.83)
- Very low precision due to high false positive rate

## Analysis Approach

### Phase 1: Data Extraction

Extract from benchmark JSON:
```
For each candidate_event:
  - detector_method (mog2, yolo11l, yolov8n, frame_diff, edge)
  - event_type (person, vehicle, animal, motion, unknown)
  - confidence score
  - vlm_label (true_positive, false_positive, uncertain)
  - video timestamp (to filter daytime)
  - class_name (for YOLO detections)
```

### Phase 2: MOG2 Analysis

**Question**: At what motion threshold does MOG2 achieve acceptable precision?

Analysis:
1. Plot FP rate vs motion_area_percent threshold (current: 0.05%)
2. Plot TP rate vs threshold (how many real events are we missing?)
3. Find threshold where precision > 0.5 (half of detections are real)

**Key insight from logs**: MOG2 alone is insufficient - it detects ALL motion including:
- Vegetation swaying
- Cloud shadows
- Lighting changes
- Camera noise

**Hypothesis**: MOG2 should be a pre-filter for YOLO, not a detection method itself.

### Phase 3: YOLO Analysis

**Question**: What confidence threshold + class filtering achieves acceptable precision?

Analysis:
1. Plot precision/recall curves by confidence threshold
2. Identify problematic classes (toilet, airplane, etc.)
3. Test class whitelisting: only [person, car, truck, motorcycle, bicycle, dog, cat, bird, deer]
4. Find threshold where precision > 0.8 for whitelisted classes

**Class whitelist rationale**:
- **Include**: person, car, truck, motorcycle, bicycle, bus, dog, cat, bird, deer, bear, horse, cow, sheep
- **Exclude**: toilet, airplane, boat, train, bench, couch, bed, dining table, etc.

### Phase 4: Combined Pipeline Analysis

Test combinations:
1. **MOG2 → YOLO**: Only run YOLO on frames where MOG2 detected motion
   - Does this reduce YOLO FPs? (propane tank is static)

2. **YOLO confidence thresholds by class**:
   - person: 0.4 (want high recall)
   - vehicle: 0.5
   - animal: 0.6
   - unknown/other: 0.8 (very conservative)

3. **Temporal filtering**:
   - Require N detections in M seconds before triggering event
   - Reduces single-frame false positives

### Phase 5: Performance Projection

Calculate expected metrics with optimal thresholds:
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1: 2 * (precision * recall) / (precision + recall)
- FP reduction: (original_FP - tuned_FP) / original_FP

## Implementation Plan

### Tool 1: Benchmark Analyzer Script

```
backend/tools/analyze_benchmark_thresholds.py

Inputs:
  - benchmark_*.json result file
  - --daytime-only flag

Outputs:
  - Threshold sweep plots (MOG2, YOLO)
  - Precision/recall curves
  - Recommended thresholds
  - Projected performance metrics
```

### Tool 2: Threshold Tester

```
backend/tools/test_thresholds.py

Inputs:
  - Video file or directory
  - Threshold configuration

Outputs:
  - Annotated video with detections
  - Detection log with timestamps
  - Comparison to VLM ground truth (if available)
```

## Decision Criteria

**Can we ship MOG2/YOLO with threshold tuning?**

| Metric | Minimum | Target |
|--------|---------|--------|
| Precision | 0.5 | 0.8 |
| Recall (person) | 0.8 | 0.95 |
| Recall (vehicle) | 0.7 | 0.9 |
| FP per hour | < 10 | < 2 |

If we can't achieve minimum precision with threshold tuning alone, we need:
1. Motion-gated YOLO (only detect in moving regions)
2. Static object filtering (track and ignore stationary "detections")
3. Per-camera masking (ignore propane tank region)

## Timeline

1. **Benchmark completion**: Wait for current run
2. **Data extraction**: Parse results, filter daytime
3. **Analysis scripts**: Build threshold sweep tools
4. **Threshold optimization**: Run analysis, document findings
5. **Validation**: Test on held-out videos
6. **Implementation**: Update live_detection_worker with optimal thresholds

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `backend/tools/analyze_benchmark_thresholds.py` | Threshold analysis script |
| `backend/tools/test_thresholds.py` | Validation tool |
| `backend/app/services/ml/detection_config.py` | Threshold configuration |
| `backend/live_detection_worker.py` | Apply optimized thresholds |
