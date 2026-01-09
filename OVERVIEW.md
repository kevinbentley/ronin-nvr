# RONIN NVR - Project Overview

*Generated: 2026-01-07*

## Lines of Code by Language

| Language | Lines | Files | % |
|----------|------:|------:|--:|
| Python | 19,995 | 95 | 61% |
| TypeScript/TSX | 7,479 | 34 | 23% |
| CSS | 4,398 | 23 | 13% |
| YAML | 705 | 29 | 2% |
| Shell | 439 | 4 | 1% |
| **Total** | **32,701** | **192** | |

## Architecture - 7 Docker Services

| Service | Purpose |
|---------|---------|
| **postgres** | PostgreSQL 16 database |
| **backend** | FastAPI REST API |
| **stream-manager** | FFmpeg stream coordinator |
| **frontend** | React + Nginx |
| **live-detection** | Real-time ML detection |
| **ml-worker** | Historical recording analysis |
| **transcode-worker** | H.265 encoding (3 replicas) |

Plus optional **onvif-events** worker for camera event subscription.

## Key Features

### Backend (74 API endpoints)

- Camera management & ONVIF discovery
- HLS streaming & recording playback
- ML object detection (YOLO via ONNX Runtime)
- Storage retention policies
- JWT authentication with encrypted credentials

### Frontend (5 pages, 15 components)

- Live camera grid view
- Timeline-based playback with thumbnails
- ML detection status & job monitoring
- Camera configuration modal with ONVIF support

### ML Pipeline (14 modules)

- Real-time detection from live streams
- Historical recording analysis
- Motion gating to reduce processing
- Job queue with configurable workers

## Database

- **10 models**: User, Camera, Recording, Detection, MLJob, MLSettings, etc.
- **7 migrations** tracking schema evolution

## Dependencies

| Stack | Count |
|-------|------:|
| Python packages | 23 |
| npm packages | 16 |

**Key tech**: FastAPI, SQLAlchemy, ONNX Runtime, React 19, HLS.js, FFmpeg

## Testing

- **14 test files** with **136 test functions**
- Unit + integration test coverage

## Presentation Highlights

1. **32K+ lines of code** across a full-stack application
2. **Microservices architecture** with 7+ coordinated containers
3. **GPU-accelerated ML** with YOLO object detection
4. **Real-time streaming** via HLS with 3-5 second latency
5. **ONVIF integration** for camera discovery & native events
6. **Production-ready** with auth, encryption, migrations, and testing
