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
**Status**: Not Started
**Depends on**: Phase 1

### Tasks
- [ ] Create Pydantic schemas for Camera CRUD operations
- [ ] Implement camera API endpoints:
  - [ ] `GET /api/cameras` - List all cameras
  - [ ] `GET /api/cameras/{id}` - Get camera details
  - [ ] `POST /api/cameras` - Add new camera
  - [ ] `PUT /api/cameras/{id}` - Update camera
  - [ ] `DELETE /api/cameras/{id}` - Remove camera
- [ ] Add RTSP URL validation and parsing
- [ ] Implement `POST /api/cameras/{id}/test` endpoint using FFprobe
- [ ] Add camera status polling background task (check every 60s)
- [ ] Store camera credentials (plain text initially, encrypt in Phase 7)
- [ ] Write API tests for all endpoints

### Validation Criteria
- [ ] Can add a camera via POST request
- [ ] Test connection returns stream info (codec, resolution, fps)
- [ ] Camera list shows correct status (online/offline)
- [ ] All API tests pass

### Merge Checklist
- [ ] All validation criteria met
- [ ] Code reviewed
- [ ] Merge `phase-2-camera-api` into `master`
- [ ] Tag release: `v2.0`

---

## Phase 3: Video Recording Engine
**Branch**: `phase-3-recording`
**Status**: Not Started
**Depends on**: Phase 2

### Tasks
- [ ] Create FFmpeg wrapper service for RTSP capture
- [ ] Implement transmuxing (copy codec to MP4, no re-encoding)
- [ ] Add 15-minute file segmentation using FFmpeg segment muxer
- [ ] Create directory structure: `/{StorageRoot}/{CameraName}/{YYYY-MM-DD}/{HH-MM-SS}.mp4`
- [ ] Store recording metadata in database (file path, start/end time, size)
- [ ] Implement recording control API:
  - [ ] `POST /api/cameras/{id}/recording/start`
  - [ ] `POST /api/cameras/{id}/recording/stop`
  - [ ] `GET /api/cameras/{id}/recording/status`
- [ ] Add automatic reconnection on stream failure (with backoff)
- [ ] Write tests with mock RTSP stream

### Validation Criteria
- [ ] Single camera records to correct directory structure
- [ ] Files are exactly 15-minute segments (except last segment)
- [ ] Recording survives stream interruption and auto-reconnects
- [ ] Recording metadata stored in database
- [ ] MP4 files playable without transcoding

### Merge Checklist
- [ ] All validation criteria met
- [ ] Code reviewed
- [ ] Merge `phase-3-recording` into `master`
- [ ] Tag release: `v3.0`

---

## Phase 4: Multi-Camera & Retention
**Branch**: `phase-4-scaling`
**Status**: Not Started
**Depends on**: Phase 3

### Tasks
- [ ] Implement concurrent recording manager (asyncio-based)
- [ ] Test with 4+ simultaneous camera recordings
- [ ] Add storage quota configuration:
  - [ ] Days-based retention (e.g., keep last 30 days)
  - [ ] Size-based retention (e.g., max 2TB)
- [ ] Implement FIFO retention policy (delete oldest files first)
- [ ] Create background task for retention enforcement (run every hour)
- [ ] Add recording statistics API:
  - [ ] `GET /api/storage/stats` - Total size, file count, oldest/newest
- [ ] Stress test with simulated 16-camera load

### Validation Criteria
- [ ] 4+ cameras record concurrently without frame drops
- [ ] Old files automatically deleted when quota reached
- [ ] Storage stats API returns accurate information
- [ ] No memory leaks during extended recording (4+ hours)

### Merge Checklist
- [ ] All validation criteria met
- [ ] Code reviewed
- [ ] Merge `phase-4-scaling` into `master`
- [ ] Tag release: `v4.0`

---

## Phase 5: Live View Frontend
**Branch**: `phase-5-live-view`
**Status**: Not Started
**Depends on**: Phase 4

### Tasks
- [ ] Initialize React project with TypeScript and Vite
- [ ] Create API client service (axios-based)
- [ ] Build camera list/grid component
- [ ] Implement HLS.js player integration
- [ ] Add Low-Latency HLS stream generation endpoint:
  - [ ] `GET /api/cameras/{id}/stream/hls/playlist.m3u8`
  - [ ] `GET /api/cameras/{id}/stream/hls/{segment}.ts`
- [ ] Create configurable grid layout (2x2, 3x3, 4x4)
- [ ] Display camera status indicators (online/offline/recording)
- [ ] Add camera management UI:
  - [ ] Add camera modal
  - [ ] Edit camera modal
  - [ ] Delete camera confirmation
  - [ ] Test connection button

### Validation Criteria
- [ ] Can view 4 live camera feeds in grid layout
- [ ] Latency is under 5 seconds from source to browser
- [ ] Camera status updates in real-time
- [ ] Can add/edit/delete cameras from UI
- [ ] Grid layout persists across page refresh

### Merge Checklist
- [ ] All validation criteria met
- [ ] Code reviewed
- [ ] Merge `phase-5-live-view` into `master`
- [ ] Tag release: `v5.0`

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
