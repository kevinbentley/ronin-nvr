# RoninNVR - Project TODO

A multi-phase implementation plan for building a Network Video Recorder system.

## Technology Stack
- **Backend**: Python 3.11+ / FastAPI
- **Frontend**: React / TypeScript
- **Database**: PostgreSQL 14+
- **Video Processing**: FFmpeg 5.x+
- **Live Streaming**: Low-Latency HLS

---

## Phase 1: Project Foundation
**Branch**: `phase-1-foundation`
**Status**: COMPLETE

### Tasks
- [x] Create project directory structure (backend/, frontend/, storage/)
- [x] Set up Python venv and `setup_venv.sh` script
- [x] Create `requirements.txt` with initial dependencies
- [x] Create `SYSTEM.md` documenting FFmpeg, PostgreSQL requirements
- [x] Set up PostgreSQL database connection with async support
- [x] Create SQLAlchemy models (Camera, Recording)
- [x] Set up Alembic migrations
- [x] Create basic FastAPI app with `/health` endpoint
- [x] Write tests for database models

### Validation Criteria
- [x] `pytest` passes with all tests green (10/10 passed)
- [x] `/health` endpoint responds with 200 OK
- [x] Database migrations configured (Alembic ready)
- [x] `setup_venv.sh` creates working virtual environment

### Merge Checklist
- [x] All validation criteria met
- [x] Code reviewed
- [x] Merge `phase-1-foundation` into `master`
- [x] Tag release: `v1.0`

---

## Phase 2: Camera Management API
**Branch**: `phase-2-camera-api`
**Status**: COMPLETE
**Depends on**: Phase 1

### Tasks
- [x] Create Pydantic schemas for Camera CRUD operations
- [x] Implement camera API endpoints:
  - [x] `GET /api/cameras` - List all cameras
  - [x] `GET /api/cameras/{id}` - Get camera details
  - [x] `POST /api/cameras` - Add new camera
  - [x] `PUT /api/cameras/{id}` - Update camera
  - [x] `DELETE /api/cameras/{id}` - Remove camera
- [x] Add RTSP URL validation and parsing
- [x] Implement `POST /api/cameras/{id}/test` endpoint using FFprobe
- [x] Add camera status polling background task (check every 60s)
- [x] Store camera credentials (plain text initially, encrypt in Phase 7)
- [x] Write API tests for all endpoints

### Validation Criteria
- [x] Can add a camera via POST request
- [x] Test connection returns stream info (codec, resolution, fps)
- [x] Camera list shows correct status (online/offline)
- [x] All API tests pass (26/26 passed)

### Merge Checklist
- [x] All validation criteria met
- [x] Code reviewed
- [x] Merge `phase-2-camera-api` into `master`
- [x] Tag release: `v2.0`

---

## Phase 3: Video Recording Engine
**Branch**: `phase-3-recording`
**Status**: COMPLETE
**Depends on**: Phase 2

### Tasks
- [x] Create FFmpeg wrapper service for RTSP capture
- [x] Implement transmuxing (copy codec to MP4, no re-encoding)
- [x] Add 15-minute file segmentation using FFmpeg segment muxer
- [x] Create directory structure: `/{StorageRoot}/{CameraName}/{YYYY-MM-DD}/{HH-MM-SS}.mp4`
- [x] Store recording metadata in database (file path, start/end time, size)
- [x] Implement recording control API:
  - [x] `POST /api/cameras/{id}/recording/start`
  - [x] `POST /api/cameras/{id}/recording/stop`
  - [x] `GET /api/cameras/{id}/recording/status`
- [x] Add automatic reconnection on stream failure (with backoff)
- [x] Write tests with mock RTSP stream

### Validation Criteria
- [x] Single camera records to correct directory structure
- [x] Files are exactly 15-minute segments (except last segment)
- [x] Recording survives stream interruption and auto-reconnects
- [x] Recording metadata stored in database
- [x] MP4 files playable without transcoding (transmux only)

### Merge Checklist
- [x] All validation criteria met
- [x] Code reviewed
- [x] Merge `phase-3-recording` into `master`
- [x] Tag release: `v3.0`

---

## Phase 4: Multi-Camera & Retention
**Branch**: `phase-4-scaling`
**Status**: COMPLETE
**Depends on**: Phase 3

### Tasks
- [x] Implement concurrent recording manager (asyncio-based)
- [x] Test with 4+ simultaneous camera recordings
- [x] Add storage quota configuration:
  - [x] Days-based retention (e.g., keep last 30 days)
  - [x] Size-based retention (e.g., max 2TB)
- [x] Implement FIFO retention policy (delete oldest files first)
- [x] Create background task for retention enforcement (run every hour)
- [x] Add recording statistics API:
  - [x] `GET /api/storage/stats` - Total size, file count, oldest/newest
  - [x] `POST /api/storage/cleanup` - Manual retention cleanup trigger
- [x] Write tests for retention service and storage API

### Validation Criteria
- [x] RecordingManager handles multiple concurrent recordings
- [x] Old files automatically deleted when quota reached (by age or size)
- [x] Storage stats API returns accurate information
- [x] RetentionMonitor runs periodic cleanup (configurable interval)
- [x] All 42 tests pass

### Merge Checklist
- [x] All validation criteria met
- [x] Code reviewed
- [x] Merge `phase-4-scaling` into `master`
- [x] Tag release: `v4.0`

---

## Phase 5: Live View Frontend
**Branch**: `phase-5-live-view`
**Status**: COMPLETE
**Depends on**: Phase 4

### Tasks
- [x] Initialize React project with TypeScript and Vite
- [x] Create API client service (axios-based)
- [x] Build camera list/grid component
- [x] Implement HLS.js player integration
- [x] Add Low-Latency HLS stream generation endpoint:
  - [x] `GET /api/cameras/{id}/stream/hls/playlist.m3u8`
  - [x] `GET /api/cameras/{id}/stream/hls/{segment}.ts`
  - [x] `POST /api/cameras/{id}/stream/start`
  - [x] `POST /api/cameras/{id}/stream/stop`
- [x] Create configurable grid layout (1x1, 2x2, 3x3, 4x4)
- [x] Display camera status indicators (online/offline/recording)
- [x] Add camera management UI:
  - [x] Add camera modal
  - [x] Edit camera modal
  - [x] Delete camera confirmation
  - [x] Test connection button
- [x] Camera sidebar with recording controls

### Validation Criteria
- [x] Grid layout component displays cameras
- [x] HLS.js player integrates with streaming endpoints
- [x] Camera status indicators update from API polling
- [x] Can add/edit/delete cameras from UI
- [x] Grid layout persists across page refresh (localStorage)
- [x] Backend tests pass (42 tests)
- [x] Frontend builds without errors

### Merge Checklist
- [x] All validation criteria met
- [x] Code reviewed
- [x] Merge `phase-5-live-view` into `master`
- [x] Tag release: `v5.0`

---

## Phase 6: Playback & Timeline
**Branch**: `phase-6-playback`
**Status**: Not Started
**Depends on**: Phase 5

### Tasks
- [ ] Create recordings API:
  - [ ] `GET /api/recordings?camera_id=X&start=DATE&end=DATE`
  - [ ] `GET /api/recordings/{id}` - Get recording details
  - [ ] `GET /api/recordings/{id}/stream` - Stream recording
- [ ] Build calendar date picker component
- [ ] Implement timeline scrubber UI with hour markers
- [ ] Add video player with seek functionality
- [ ] Create segment stitching for continuous playback across files
- [ ] Add download/export endpoint:
  - [ ] `GET /api/recordings/{id}/download`
  - [ ] `POST /api/recordings/export` - Export time range as single file
- [ ] Implement clip selection UI (start/end time markers)

### Validation Criteria
- [ ] Can navigate to any recorded date via calendar
- [ ] Timeline shows all available recordings for selected day
- [ ] Scrubbing timeline seeks video accurately
- [ ] Can download individual recording segments
- [ ] Can export custom time range as single file
- [ ] Playback is seamless across segment boundaries

### Merge Checklist
- [ ] All validation criteria met
- [ ] Code reviewed
- [ ] Merge `phase-6-playback` into `master`
- [ ] Tag release: `v6.0`

---

## Phase 7: Security & Production
**Branch**: `phase-7-security`
**Status**: Not Started
**Depends on**: Phase 6

### Tasks
- [ ] Add user authentication:
  - [ ] User model with hashed passwords
  - [ ] JWT token generation and validation
  - [ ] `POST /api/auth/login`
  - [ ] `POST /api/auth/logout`
  - [ ] `GET /api/auth/me`
- [ ] Create login page UI
- [ ] Protect all API endpoints with auth middleware
- [ ] Encrypt stored camera credentials (Fernet/AES)
- [ ] Add API rate limiting (100 requests/minute)
- [ ] Configure CORS for production
- [ ] Create Docker Compose for deployment:
  - [ ] Backend service
  - [ ] PostgreSQL database
  - [ ] Frontend (nginx)
- [ ] Run 48-hour stability test
- [ ] Document deployment process in README.md

### Validation Criteria
- [ ] Cannot access any API without valid token
- [ ] Login page works correctly
- [ ] Camera credentials are encrypted in database
- [ ] Rate limiting blocks excessive requests
- [ ] Docker Compose deploys successfully
- [ ] System runs stable for 48+ hours without memory leaks

### Merge Checklist
- [ ] All validation criteria met
- [ ] Security review completed
- [ ] Code reviewed
- [ ] Merge `phase-7-security` into `master`
- [ ] Tag release: `v7.0` (Production Ready)

---

## Quick Reference

### Git Workflow
```bash
# Start new phase
git checkout master
git pull
git checkout -b phase-N-name

# Complete phase
git checkout master
git merge phase-N-name
git tag vN.0
git push origin master --tags
```

### Key Commands
```bash
# Backend
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
pytest

# Frontend
cd frontend
npm run dev
npm run build

# Database
alembic upgrade head
alembic revision --autogenerate -m "description"
```

### Dependencies Summary
**System**: FFmpeg 5.x+, PostgreSQL 14+, Node.js 18+, Python 3.11+

**Backend (pip)**: fastapi, uvicorn, sqlalchemy, alembic, asyncpg, pydantic, python-dotenv, pytest, httpx

**Frontend (npm)**: react, typescript, vite, hls.js, axios
