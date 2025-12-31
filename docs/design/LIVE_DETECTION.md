# Live Detection Design Document

## Overview

This document describes the design for real-time object detection from live camera streams. Instead of waiting for recordings to complete (up to 15 minutes), the system will tap into existing HLS segments as they are written and run YOLO inference in near real-time.

**Goal:** Detect objects within 2-5 seconds of them appearing on camera, enabling immediate notifications.

## Current Architecture (File-Based ML)

```
Camera → FFmpeg → MP4 (15 min) → [wait] → ML Worker scans file → Detection stored
                                          ↑
                               Latency: 15+ minutes
```

The existing ML pipeline processes completed recording files:
- `ml_worker.py` claims jobs from PostgreSQL
- Extracts frames at 1-2 fps from MP4 files
- Runs YOLO inference via `DetectionService`
- Stores `Detection` records in database

## Proposed Architecture (Stream-Based ML)

```
Camera → FFmpeg → HLS segments (.ts) ←──┐
              ↓                          │
         MP4 recording                   │
                                         │
        LiveDetectionWorker ─────────────┘
              ↓
         Extract frame (1-2 fps)
              ↓
         YOLO inference
              ↓
         Debounce filter
              ↓
     ┌────────┼────────┐
     ↓        ↓        ↓
  Database  Snapshot  WebSocket
             (JPG)    notification
              ↓
        [Future: Vision LLM]
```

### Key Design Decisions

1. **Single Process, Multiple Cameras**
   - At 1-2 fps per camera, GPU inference (~20ms/frame) is not the bottleneck
   - A single worker can handle 8-16 cameras easily
   - Simpler deployment, shared model memory

2. **HLS Tapping (Not RTSP)**
   - Reuse existing HLS segments written by camera_stream.py
   - No additional RTSP connections (cameras have limited concurrent streams)
   - Graceful degradation: if ML falls behind, skip segments

3. **Separate Detection Table**
   - Live detections go to `live_detections` table (not `detections`)
   - No `recording_id` - these aren't tied to specific recordings
   - Shorter retention (24-48 hours vs 90 days)
   - Prevents confusion between real-time alerts and historical analysis

4. **Debouncing**
   - Don't spam notifications: "person detected" once per 30 seconds per camera
   - Track last detection time per (camera_id, class_name)
   - Configurable cooldown period

5. **Snapshot Capture**
   - Save JPG snapshot when detection triggers notification
   - Draw bounding boxes on snapshot for visual confirmation
   - Store in `.snapshots/{camera_id}/{timestamp}.jpg`
   - Snapshots enable future Vision LLM analysis
   - Same retention as live_detections (48 hours default)

## Components

### 1. LiveDetectionWorker (`live_detection_worker.py`)

Standalone process (like ml_worker.py and transcode_worker.py).

```python
class LiveDetectionWorker:
    """Single worker monitoring all active camera HLS streams."""

    def __init__(self):
        self.detector: DetectionService      # Shared YOLO model
        self.cameras: dict[int, CameraState] # Per-camera state
        self.debounce: DebounceTracker       # Notification cooldowns

    async def run(self):
        """Main loop - round-robin through cameras."""
        while True:
            for camera_id, state in self.cameras.items():
                segment = await self.get_latest_segment(camera_id)
                if segment and segment != state.last_processed:
                    await self.process_segment(camera_id, segment)
                    state.last_processed = segment
            await asyncio.sleep(0.5)  # 500ms cycle
```

#### Segment Processing

```python
async def process_segment(self, camera_id: int, segment_path: Path):
    """Extract frame and run detection."""
    # Extract single frame from segment (middle of 2-sec segment)
    frame = await self.extract_frame_from_segment(segment_path)
    if frame is None:
        return

    # Run YOLO inference
    results = self.detector.detect(frame, self.model_name)

    # Filter and debounce
    for detection in results:
        if self.debounce.should_notify(camera_id, detection.class_name):
            await self.store_and_notify(camera_id, detection)
            self.debounce.mark_notified(camera_id, detection.class_name)
```

#### Frame Extraction from HLS Segment

```python
async def extract_frame_from_segment(self, segment_path: Path) -> Optional[np.ndarray]:
    """Extract a single frame from an HLS .ts segment."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(segment_path),
        "-vf", f"select='eq(n,{FRAME_INDEX})',scale={MAX_DIM}:-1",
        "-frames:v", "1",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    # Run and read stdout as numpy array
```

### 2. Database Model (`LiveDetection`)

New model separate from historical `Detection`:

```python
class LiveDetection(Base):
    """Real-time detection from live stream."""

    __tablename__ = "live_detections"

    id: int                    # Primary key
    camera_id: int             # FK to cameras
    class_name: str            # "person", "car", etc.
    confidence: float          # 0.0 - 1.0
    bbox_x: float              # Normalized bounding box
    bbox_y: float
    bbox_width: float
    bbox_height: float
    model_name: str            # "yolov8l"
    detected_at: datetime      # When detection occurred (UTC)
    notified: bool             # Whether notification was sent
    snapshot_path: str         # Path to JPG snapshot (nullable)
    llm_description: str       # Vision LLM description (nullable, future)

    # Indexes for efficient queries
    __table_args__ = (
        Index("ix_live_detections_camera_detected", "camera_id", "detected_at"),
        Index("ix_live_detections_class_name", "class_name"),
    )
```

### 3. Snapshot Service

Captures and stores JPG snapshots when detections occur:

```python
class SnapshotService:
    """Save detection snapshots with bounding boxes."""

    def __init__(self, storage_root: Path):
        self.storage_root = storage_root
        self.snapshots_dir = storage_root / ".snapshots"
        self.jpeg_quality = 85  # Balance quality vs size

    def save_snapshot(
        self,
        frame: np.ndarray,
        camera_id: int,
        detections: list[DetectionResult],
        timestamp: datetime,
    ) -> Path:
        """Save frame with detection boxes drawn.

        Args:
            frame: BGR image (original resolution)
            camera_id: Camera that captured the frame
            detections: List of detections to draw
            timestamp: Detection timestamp

        Returns:
            Path to saved JPG file
        """
        # Draw bounding boxes on frame
        annotated = self._draw_boxes(frame.copy(), detections)

        # Generate path: .snapshots/{camera_id}/{YYYY-MM-DD}/{HH-MM-SS-fff}.jpg
        date_dir = self.snapshots_dir / str(camera_id) / timestamp.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        filename = timestamp.strftime("%H-%M-%S-%f")[:-3] + ".jpg"
        snapshot_path = date_dir / filename

        # Save as JPEG
        cv2.imwrite(str(snapshot_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

        return snapshot_path

    def _draw_boxes(self, frame: np.ndarray, detections: list[DetectionResult]) -> np.ndarray:
        """Draw detection boxes and labels on frame."""
        h, w = frame.shape[:2]
        for det in detections:
            # Convert normalized coords to pixels
            x1 = int(det.x * w)
            y1 = int(det.y * h)
            x2 = int((det.x + det.width) * w)
            y2 = int((det.y + det.height) * h)

            # Draw box (green for person, blue for vehicle, etc.)
            color = self._get_class_color(det.class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det.class_name} {det.confidence:.0%}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def _get_class_color(self, class_name: str) -> tuple[int, int, int]:
        """Get BGR color for detection class."""
        colors = {
            "person": (0, 255, 0),      # Green
            "car": (255, 0, 0),          # Blue
            "truck": (255, 0, 0),
            "dog": (0, 165, 255),        # Orange
            "cat": (0, 165, 255),
        }
        return colors.get(class_name, (0, 255, 255))  # Yellow default
```

**Storage Structure:**
```
.snapshots/
├── 1/                          # camera_id
│   ├── 2025-01-15/
│   │   ├── 08-30-45-123.jpg   # HH-MM-SS-milliseconds
│   │   ├── 08-31-22-456.jpg
│   │   └── ...
│   └── 2025-01-14/
├── 2/
│   └── ...
```

**Storage Estimates:**
- ~50-100KB per snapshot (1080p, JPEG quality 85)
- 1 detection/minute/camera = ~150KB/hour/camera
- 8 cameras × 24 hours × 150KB = ~28MB/day
- 48-hour retention = ~56MB per camera

### 4. Debounce Tracker

Prevents notification spam:

```python
@dataclass
class DebounceTracker:
    """Track detection cooldowns per camera/class."""

    cooldown_seconds: float = 30.0
    _last_notified: dict[tuple[int, str], datetime] = field(default_factory=dict)

    def should_notify(self, camera_id: int, class_name: str) -> bool:
        key = (camera_id, class_name)
        last = self._last_notified.get(key)
        if last is None:
            return True
        return (datetime.utcnow() - last).total_seconds() >= self.cooldown_seconds

    def mark_notified(self, camera_id: int, class_name: str):
        self._last_notified[(camera_id, class_name)] = datetime.utcnow()
```

### 4. WebSocket/SSE Notifications

Extend existing `MLEventService` with new event types:

```python
class EventType(str, Enum):
    # ... existing types ...
    LIVE_DETECTION = "live_detection"       # Real-time detection event
    LIVE_DETECTION_SUMMARY = "live_summary" # Periodic summary

async def emit_live_detection(
    self,
    camera_id: int,
    camera_name: str,
    class_name: str,
    confidence: float,
    thumbnail_url: Optional[str] = None,
) -> None:
    """Emit live detection event for real-time alerts."""
    await self.emit(MLEvent(
        event_type=EventType.LIVE_DETECTION,
        data={
            "camera_id": camera_id,
            "camera_name": camera_name,
            "class_name": class_name,
            "confidence": confidence,
            "thumbnail_url": thumbnail_url,
            "detected_at": datetime.utcnow().isoformat(),
        },
    ))
```

### 5. Configuration

Add to `Settings` in `config.py`:

```python
# Live Detection settings
live_detection_enabled: bool = True
live_detection_fps: float = 1.0           # Frames per second per camera
live_detection_cooldown: float = 30.0     # Seconds between same-class notifications
live_detection_confidence: float = 0.6    # Higher threshold for alerts
live_detection_classes: str = "person,car,truck"  # Classes that trigger alerts
live_detection_retention_hours: int = 48  # Keep live detections for 48 hours
```

### 6. API Endpoints

New endpoints in `app/api/ml.py`:

```python
@router.get("/live-detections")
async def get_live_detections(
    camera_id: Optional[int] = None,
    class_name: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = 100,
):
    """Get recent live detections."""
    ...

@router.get("/live-detections/stream")
async def stream_live_detections():
    """SSE endpoint for real-time detection events."""
    ...

@router.get("/live-detections/status")
async def get_live_detection_status():
    """Get live detection worker status."""
    return {
        "enabled": settings.live_detection_enabled,
        "cameras_monitored": len(worker.cameras),
        "detections_last_hour": count,
        "model_loaded": worker.model_loaded,
    }
```

## Data Flow

### Startup Sequence

```
1. LiveDetectionWorker starts
2. Loads YOLO model into memory
3. Queries database for active cameras
4. Scans .streams/ directory for camera subdirectories
5. Enters main loop
```

### Per-Camera Processing (every 0.5-1 second)

```
1. List .ts files in .streams/{camera_id}/
2. Find newest segment not yet processed
3. If new segment found:
   a. Extract single frame from middle of segment
   b. Scale to 640px max dimension
   c. Run YOLO inference
   d. For each detection above confidence threshold:
      - Check debounce tracker
      - If should notify:
        * Insert into live_detections table
        * Emit WebSocket event
        * Update debounce tracker
4. Sleep briefly before next camera
```

### Graceful Degradation

If the worker falls behind (slow GPU, many cameras):
- Skip old segments, only process newest
- Log warning about falling behind
- Never block camera recording

## Performance Estimates

| Cameras | FPS/Camera | Total FPS | GPU Time (20ms/frame) | Utilization |
|---------|------------|-----------|----------------------|-------------|
| 4       | 2          | 8         | 160ms/sec            | 16%         |
| 8       | 1          | 8         | 160ms/sec            | 16%         |
| 16      | 1          | 16        | 320ms/sec            | 32%         |
| 32      | 0.5        | 16        | 320ms/sec            | 32%         |

**Memory usage:**
- YOLO model: ~500MB GPU RAM
- Per-camera state: negligible
- Frame buffer: ~2MB per camera (1080p BGR)

## Docker Integration

Add new service to `docker-compose.yml`:

```yaml
live-detection:
  build:
    context: ./backend
    dockerfile: Dockerfile
  command: python live_detection_worker.py
  environment:
    - DATABASE_URL=${DATABASE_URL}
    - STORAGE_ROOT=/data/storage
    - LIVE_DETECTION_ENABLED=true
  volumes:
    - storage:/data/storage
  depends_on:
    - postgres
    - backend
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

## Migration

Alembic migration file: `backend/alembic/versions/20241230_add_live_detections.py`

Creates the `live_detections` table with indexes for:
- `(camera_id, detected_at)` - primary query pattern
- `class_name` - filtering by detection type
- `detected_at` - retention cleanup

Run migration:
```bash
cd backend
alembic upgrade head
```

## Testing Strategy

1. **Unit Tests**
   - DebounceTracker logic
   - Frame extraction from .ts files
   - Detection filtering

2. **Integration Tests**
   - Worker startup/shutdown
   - Database persistence
   - WebSocket event emission

3. **Manual Testing**
   - Walk in front of camera
   - Verify notification within 2-5 seconds
   - Check debounce prevents spam

## Rollout Plan

### Phase 1: Core Worker
- [ ] Create `live_detection_worker.py`
- [ ] Add `LiveDetection` model and migration
- [ ] Implement HLS segment tapping
- [ ] Basic detection loop (no notifications yet)

### Phase 2: Notifications
- [ ] Add debounce tracker
- [ ] Extend `MLEventService` with live detection events
- [ ] Add SSE endpoint for live detection stream
- [ ] Frontend: display real-time detection alerts

### Phase 3: Configuration & Polish
- [ ] Add configuration options to Settings
- [ ] Per-camera enable/disable
- [ ] Per-class alert configuration
- [ ] Retention cleanup job for old live_detections

### Phase 4: Snapshots
- [ ] Implement SnapshotService
- [ ] Save JPG with bounding boxes on detection
- [ ] Add snapshot_path to LiveDetection model
- [ ] Serve snapshots via API endpoint
- [ ] Add snapshot cleanup to retention job

### Phase 5: Vision LLM Integration (Future)
- [ ] Design LLM service abstraction (support multiple providers)
- [ ] Implement async LLM processing queue
- [ ] Send snapshots to Vision LLM for scene description
- [ ] Store descriptions in llm_description field
- [ ] Add natural language search over descriptions

### Phase 6: Advanced Features (Future)
- [ ] Push notifications (mobile)
- [ ] Detection zones (only alert if person in specific area)
- [ ] Object tracking (same person across frames)
- [ ] Alert rules engine (if person + car + nighttime → high priority)

## Future: Vision LLM Integration

The snapshot capture enables a powerful future feature: using Vision LLMs to describe
what's happening in the scene, enabling natural language search and smarter alerts.

### Architecture

```
LiveDetection (with snapshot)
         ↓
    LLM Job Queue (PostgreSQL)
         ↓
    LLMWorker (separate process)
         ↓
    Vision LLM API (Claude, GPT-4V, etc.)
         ↓
    Store description in live_detections.llm_description
```

### Example Flow

1. Live detection: "person" detected at front door
2. Snapshot saved: `.snapshots/1/2025-01-15/08-30-45-123.jpg`
3. LLM job queued with snapshot path
4. LLMWorker sends image to Vision API with prompt:
   ```
   Describe what is happening in this security camera image.
   Focus on: people (appearance, actions), vehicles, packages, unusual activity.
   Be concise (1-2 sentences).
   ```
5. Response stored: "A delivery driver in a brown uniform is placing a package
   on the front porch. A white delivery van is visible in the driveway."
6. User can later search: "when was a package delivered?" → finds this event

### LLM Provider Abstraction

```python
class VisionLLMProvider(Protocol):
    """Abstract interface for Vision LLM providers."""

    async def describe_image(
        self,
        image_path: Path,
        prompt: str,
        max_tokens: int = 150,
    ) -> str:
        """Send image to LLM and get description."""
        ...

class ClaudeVisionProvider(VisionLLMProvider):
    """Claude API implementation."""
    ...

class OpenAIVisionProvider(VisionLLMProvider):
    """GPT-4V implementation."""
    ...

class OllamaVisionProvider(VisionLLMProvider):
    """Local Ollama with LLaVA or similar."""
    ...
```

### Configuration

```python
# Vision LLM settings (future)
vision_llm_enabled: bool = False
vision_llm_provider: str = "claude"  # claude, openai, ollama
vision_llm_model: str = "claude-3-haiku-20240307"  # Cost-effective for descriptions
vision_llm_api_key: str = ""
vision_llm_prompt: str = "Describe this security camera image concisely..."
vision_llm_max_tokens: int = 150
vision_llm_rate_limit: float = 1.0  # Max requests per second
```

### Cost Considerations

At ~$0.001 per image (Claude Haiku):
- 100 detections/day = $0.10/day = $3/month
- 1000 detections/day = $1/day = $30/month

Can be optimized by:
- Only processing high-confidence detections
- Batching similar detections
- Using local models (Ollama) for cost-free operation

## Open Questions

1. **Recording Correlation**: Should live detections link to recordings?
   - Current design: No, they're separate
   - Could add `approximate_recording_id` based on timestamp match
   - Recommendation: Keep separate for now, correlate in queries if needed

2. **Multi-GPU Support**: Should worker support multiple GPUs?
   - Current design: Single GPU
   - Could spawn worker per GPU with camera sharding
   - Recommendation: Single GPU handles 16+ cameras, revisit if needed

3. **Snapshot Resolution**: Should snapshots be full resolution or scaled?
   - Full resolution: Better for LLM analysis, larger files
   - Scaled (1280px): Smaller files, still good for LLM
   - Recommendation: Keep full resolution, optimize JPEG quality

4. **LLM Processing Priority**: Should all detections get LLM analysis?
   - Could prioritize: person > vehicle > animal
   - Could skip if same scene described recently
   - Recommendation: Start with all, add filtering based on cost/value

## Appendix: Existing Code References

| Component | File | Purpose |
|-----------|------|---------|
| Camera streaming | `app/services/camera_stream.py` | HLS segment generation |
| Frame extraction | `app/services/ml/frame_extractor.py` | FFmpeg frame extraction |
| Detection service | `app/services/ml/detection_service.py` | YOLO inference |
| ML events | `app/services/ml/events.py` | SSE event emission |
| ML worker | `ml_worker.py` | File-based ML processing |
| Configuration | `app/config.py` | Settings management |
| Detection model | `app/models/detection.py` | Historical detections |
