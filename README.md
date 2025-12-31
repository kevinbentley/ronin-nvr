# RoninNVR

![RoninNVR Logo](logo.png)

A modern, self-hosted Network Video Recorder (NVR) solution built with Python FastAPI and React. RoninNVR provides reliable IP camera recording, live streaming, and playback with a clean web interface.

## Features

- **Multi-Camera Support** - Connect to IP cameras via RTSP (TCP/UDP)
- **Live View Grid** - View multiple cameras simultaneously in configurable layouts (1x1, 2x2, 3x3, 4x4)
- **Click-to-Zoom** - Click any camera feed to view full-screen
- **Continuous Recording** - 24/7 recording with automatic file segmentation
- **Timeline Playback** - Browse and play back recordings by date with visual timeline
- **Storage Management** - Automatic retention policy (by days or storage limit)
- **Low Latency Streaming** - HLS-based live view with 3-5 second latency
- **Connection Resilience** - Automatic reconnection on camera or network failures
- **ML Object Detection** - YOLO-based detection of people, vehicles, animals
- **Motion Detection** - Background subtraction-based motion alerts
- **Storage Optimization** - H.265 transcoding reduces storage by 40-70%
- **GPU Acceleration** - NVENC support for 10-20x faster transcoding
- **Docker Deployment** - Production-ready containerized deployment

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- FFmpeg 5.x+
- Node.js 18+

See [SYSTEM.md](SYSTEM.md) for detailed installation instructions.

### Backend Setup

```bash
# Create Python environment
./setup_venv.sh

# Configure environment
cp backend/.env.example backend/.env
# Edit .env with your database credentials

# Run database migrations
cd backend
source venv/bin/activate
alembic upgrade head

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Access the web interface at `http://localhost:5173`

---

## Docker Deployment

The recommended way to deploy RoninNVR in production is using Docker Compose.

### Prerequisites

- Docker Engine 24+
- Docker Compose v2
- (Optional) NVIDIA GPU with Container Toolkit for hardware-accelerated transcoding

### Quick Deploy (CPU)

```bash
# Clone the repository
git clone <repo-url>
cd dvr

# Create environment file
cp .env.example .env
# Edit .env with your secrets (JWT_SECRET_KEY, POSTGRES_PASSWORD)

# Build and start
docker compose -f docker-compose.cpu.yml up -d --build

# Check status
docker compose ps
```

### Deploy with NVIDIA GPU

For hardware-accelerated ML inference and video transcoding:

```bash
# Install NVIDIA Container Toolkit (one-time)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Deploy with GPU support
docker compose up -d --build
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| `frontend` | 80 | Web UI (Nginx + React) |
| `backend` | 8000 | FastAPI server |
| `postgres` | 5432 | PostgreSQL database |
| `ml-worker` | - | Object detection (YOLO) |
| `transcode-worker` | - | H.265 re-encoding |

### Scaling Workers

```bash
# Scale ML workers (for more concurrent video analysis)
docker compose up -d --scale ml-worker=4

# Scale transcode workers (for faster re-encoding)
docker compose up -d --scale transcode-worker=2
```

### Storage Configuration

By default, recordings are stored in a Docker volume. To use a host directory:

```yaml
# In docker-compose.yml, change:
volumes:
  - storage_data:/data/storage
# To:
volumes:
  - /path/to/your/storage:/data/storage
```

### Useful Commands

```bash
# View logs
docker compose logs -f backend
docker compose logs -f ml-worker

# Check transcode statistics
docker compose exec transcode-worker python transcode_worker.py --stats

# Restart a service
docker compose restart backend

# Stop everything
docker compose down

# Stop and remove volumes (DELETES DATA)
docker compose down -v
```

---

## Concept of Operations

### Overview

RoninNVR uses a unified streaming architecture where each camera is managed by a single FFmpeg process. This design ensures efficient resource usage and eliminates conflicts from multiple connections to the same camera.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              RoninNVR                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────┐     ┌──────────────────┐     ┌─────────────────────┐    │
│   │ IP Camera│────▶│  FFmpeg Process  │────▶│  HLS Segments (.ts) │    │
│   │  (RTSP)  │     │  (per camera)    │     │  for live view      │    │
│   └──────────┘     │                  │     └─────────────────────┘    │
│                    │  Single RTSP     │                                 │
│                    │  connection      │     ┌─────────────────────┐    │
│                    │                  │────▶│  MP4 Recordings     │    │
│                    └──────────────────┘     │  (15-min segments)  │    │
│                                             └─────────────────────┘    │
│                                                                          │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                     FastAPI Backend                             │   │
│   │  - Camera management API                                        │   │
│   │  - Stream control (start/stop)                                  │   │
│   │  - HLS playlist/segment serving                                 │   │
│   │  - Playback API for recordings                                  │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                     React Frontend                              │   │
│   │  - HLS.js player with auto-reconnect                           │   │
│   │  - Multi-camera grid view                                       │   │
│   │  - Timeline-based playback                                      │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Stream Management

Each camera stream is handled by a dedicated `CameraStream` instance that manages a single FFmpeg process with dual outputs:

1. **RTSP Input** - FFmpeg connects to the camera using RTSP over TCP (default) or UDP
2. **HLS Output** - 2-second segments for live streaming to browsers
3. **MP4 Output** - 15-minute segments for archival recording

**Key Features:**
- **Single Connection** - Only one RTSP connection per camera, avoiding authentication conflicts
- **Transmuxing** - Video is copied without re-encoding (H.264/H.265 passthrough)
- **Audio Transcoding** - Audio is transcoded to AAC for browser compatibility
- **Automatic Reconnection** - Up to 10 retry attempts with exponential backoff on connection loss
- **Graceful Shutdown** - Proper stream termination preserves in-progress recordings

**FFmpeg Command Structure:**
```bash
ffmpeg -rtsp_transport tcp -i rtsp://camera/stream \
    # Output 1: HLS for live view
    -c:v copy -c:a aac -f hls -hls_time 2 -hls_list_size 10 \
    -hls_flags delete_segments+append_list stream.m3u8 \
    # Output 2: MP4 for recording
    -c:v copy -c:a aac -f segment -segment_time 900 \
    -strftime 1 %H-%M-%S.mp4
```

### File Storage Structure

Recordings are organized in a hierarchical directory structure:

```
storage/
├── .streams/                    # Temporary HLS segments (live view)
│   ├── 1/                       # Camera ID
│   │   ├── playlist.m3u8
│   │   ├── segment000.ts
│   │   ├── segment001.ts
│   │   └── ...
│   └── 2/
│       └── ...
│
├── .logs/                       # FFmpeg log files
│   ├── Front_Door_ffmpeg.log
│   └── Backyard_ffmpeg.log
│
├── .exports/                    # Exported video clips
│
├── Front_Door/                  # Camera name (filesystem-safe)
│   ├── 2025-01-15/             # Date directory
│   │   ├── 08-00-00.mp4        # 15-minute segments
│   │   ├── 08-15-00.mp4
│   │   └── ...
│   └── 2025-01-14/
│       └── ...
│
└── Backyard/
    └── ...
```

**Storage Characteristics:**
- **Segment Duration**: 15 minutes (configurable)
- **File Format**: MP4 with H.264/H.265 video and AAC audio
- **Naming Convention**: `HH-MM-SS.mp4` based on recording start time
- **Retention Policy**: Automatic cleanup by age (days) or total size (GB)

### Client Video Streaming

The frontend uses HLS.js for video playback with several optimizations for reliability:

**Live View Flow:**
```
Browser                    Backend                     Camera
   │                          │                          │
   │  GET /stream/start       │                          │
   ├─────────────────────────▶│                          │
   │                          │  RTSP Connect            │
   │                          ├─────────────────────────▶│
   │                          │  ◀───Video Stream────    │
   │                          │                          │
   │  GET playlist.m3u8       │                          │
   ├─────────────────────────▶│                          │
   │  ◀──────HLS Manifest─────┤                          │
   │                          │                          │
   │  GET segment001.ts       │                          │
   ├─────────────────────────▶│                          │
   │  ◀──────Video Chunk──────┤                          │
   │                          │                          │
   │  (continuous segment     │                          │
   │   requests...)           │                          │
```

**HLS Configuration:**
- **Segment Duration**: 2 seconds
- **Playlist Size**: 10 segments (20-second window)
- **Target Latency**: 3-5 seconds
- **Buffer Settings**: Optimized for low latency vs stability balance

**Error Recovery:**
- **Auto-Retry**: Up to 5 automatic reconnection attempts on network errors
- **Media Recovery**: HLS.js built-in media error recovery
- **Stall Detection**: Automatic reload when playback stalls
- **Manual Reconnect**: User-triggered full stream restart

### Recording Playback

Recorded video playback uses a different path optimized for seeking and downloading:

**Playback API:**
- `GET /playback/cameras` - List cameras with recordings
- `GET /playback/cameras/{name}/dates` - Available recording dates
- `GET /playback/cameras/{name}/recordings?date=YYYY-MM-DD` - Day's recordings
- `GET /playback/recordings/{id}/stream` - Stream video file
- `GET /playback/recordings/{id}/download` - Download video file
- `POST /playback/export` - Export time range as single file

**Recording ID Format:**
Recording IDs encode the file path: `CameraName::2025-01-15::08-00-00.mp4`

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://localhost/ronin_nvr` | PostgreSQL connection string |
| `STORAGE_ROOT` | `./storage` | Root directory for recordings |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `SEGMENT_DURATION_MINUTES` | `15` | Recording segment length |
| `RETENTION_DAYS` | `30` | Days to keep recordings |
| `RETENTION_MAX_GB` | `null` | Max storage size (optional) |

### Camera Configuration

Each camera requires:
- **Name**: Display name for the camera
- **Host**: IP address or hostname
- **Port**: RTSP port (default: 554)
- **Path**: RTSP stream path (e.g., `/cam/realmonitor`)
- **Username/Password**: Camera credentials
- **Transport**: TCP (default, more reliable) or UDP (lower latency)

---

## Project Structure

```
dvr/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI route handlers
│   │   ├── models/           # SQLAlchemy models
│   │   ├── schemas/          # Pydantic schemas
│   │   ├── services/         # Business logic
│   │   │   ├── camera_stream.py  # Stream management
│   │   │   ├── playback.py       # Recording playback
│   │   │   └── retention.py      # Storage cleanup
│   │   ├── config.py         # Settings
│   │   └── main.py           # FastAPI app
│   ├── tests/
│   ├── alembic/              # Database migrations
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/            # Page components
│   │   ├── services/         # API client
│   │   └── types/            # TypeScript types
│   └── package.json
│
├── storage/                  # Video storage (gitignored)
├── setup_venv.sh            # Python environment setup
├── camera_cli.sh            # CLI for camera testing
└── README.md
```

---

## Tested Cameras

- Amcrest IP cameras (RTSP)
- Synology IP cameras (RTSP)

The system should work with any RTSP-compatible IP camera using H.264 or H.265 video encoding.

---

## License

TBD

---

## Contributing

- Kevin Bentley
- Claude Code

---

## Future Features

- Selectable camera grid - Choose which cameras to display
- Cold storage - Archive to remote server with playback integration
- Detection zones - Mask areas to ignore (clocks, etc.)
- Push notifications - Alerts for detected objects/motion
- Mobile app - Android/iOS companion app
- Distributed processing - Stream handling across multiple servers