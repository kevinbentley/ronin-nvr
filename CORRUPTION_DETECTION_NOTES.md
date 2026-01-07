# Video Frame Corruption Detection - Analysis Notes

## Problem Statement

UDP RTSP streams from Ubiquiti cameras experience packet loss that causes video corruption. This corruption manifests as vertical banding/striping patterns in decoded frames. The corruption:
1. Triggers false motion detection (frames flagged as having motion when they don't)
2. Can cause video playback errors in the browser (PIPELINE_ERROR_DECODE)

## Root Cause Understanding

When UDP packets are lost during video decode, the decoder **repeats the last successfully decoded row of pixels** for all remaining rows in the affected macroblocks. This creates a distinctive pattern:
- Columns have **nearly identical pixel values** down their entire length
- This is something that virtually **never happens in natural scenes**

## Final Detection Algorithm

The key insight is to measure **row-to-row pixel differences** within each column:
- **Corrupted columns**: Mean row diff ≈ 0 (same value repeated)
- **Natural scenes**: Mean row diff >> 0 (variation down the column)

### Algorithm Steps
1. Focus on bottom 40% of frame (where corruption typically manifests)
2. Compute absolute difference between consecutive rows for each column
3. Calculate mean difference per column
4. Count columns with mean diff < 0.5 ("repeated" columns)
5. Flag as corrupt if >50% of columns are repeated OR overall mean < 0.5

### Test Results

| Image | Repeated Cols | Mean Diff | Result |
|-------|---------------|-----------|--------|
| Clean daytime | 2.7% | 15.01 | Clean ✓ |
| Clean night | 15.2% | 1.34 | Clean ✓ |
| Corrupted 12:11 | 92.1% | 0.13 | Corrupt ✓ |
| Corrupted 12:07 | 99.6% | 0.05 | Corrupt ✓ |
| Corrupted 10:40 | 92.0% | 0.12 | Corrupt ✓ |
| Corrupted 10:50 | 92.7% | 0.11 | Corrupt ✓ |

**Clean separation**: Corrupted images have 92-99% repeated columns with mean diff 0.05-0.13. Clean images have <16% repeated columns with mean diff >1.3.

## Previous Approaches (Superseded)

### Method 1: Thin Band Detection
- Counted clusters of uniform columns in bottom 40%
- **Problem**: False positives on natural uniform areas (gravel, walls)

### Method 2: Autocorrelation
- Detected periodic stripe patterns via autocorrelation
- **Problem**: Required tuning, still had edge cases

## Implementation

### Backend
File: `backend/app/services/ml/motion_gate.py`

```python
def detect_frame_corruption(frame, repeated_col_threshold=0.5, mean_diff_threshold=0.5):
    """Detect vertical banding corruption from decode errors."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    # Focus on bottom 40%
    bottom = gray[int(height * 0.6):, :]

    # Row-to-row differences per column
    row_diffs = np.abs(np.diff(bottom.astype(float), axis=0))
    col_mean_diff = np.mean(row_diffs, axis=0)

    # Count repeated columns
    repeated_pct = np.sum(col_mean_diff < mean_diff_threshold) / width
    overall_mean = np.mean(col_mean_diff)

    return repeated_pct > repeated_col_threshold or overall_mean < mean_diff_threshold
```

### Frontend
File: `frontend/src/components/UnifiedVideoPlayer/hooks/useHlsPlayer.ts`
- Added decode error recovery (seek forward + HLS.js recoverMediaError)
- Prevents playback from stopping on corrupted segments

## Files Modified

- `backend/app/services/ml/motion_gate.py` - `detect_frame_corruption()` function
- `backend/live_detection_worker.py` - Skip corrupted frames in processing loop
- `frontend/src/components/UnifiedVideoPlayer/hooks/useHlsPlayer.ts` - Decode error recovery

## Test Images Location

Camera 10 (Trashcan) snapshots in `/opt3/ronin/storage/.snapshots/10/2026-01-06/`:
- Clean daytime: `00-23-45-737.jpg`
- Clean night: `05-22-11-082.jpg`
- Corrupted: `19-11-06-309.jpg`, `19-07-35-909.jpg`, `17-40-03-133.jpg`, `17-50-33-185.jpg`

## Limitations

The current algorithm detects **vertical repetition** corruption (the most common type from UDP packet loss). It does NOT detect:
- Horizontal band artifacts (different corruption type, less common)
- Color channel-specific corruption
- Partial frame corruption in the top portion

These are less common and typically less severe than the vertical banding that covers large portions of the frame.
