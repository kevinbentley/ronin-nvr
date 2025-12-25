# Application requirements

**Application name**: RoninNVR

**Goal**: Develop a custom Network Video Recorder (NVR) solution to replace Synology Surveillance Station.

## 1. Executive Summary
The objective is to build a scalable, self-hosted NVR application. The system will decouple the video ingestion engine from the user interface, allowing for flexible configuration of IP cameras (specifically Amcrest and Synology models via RTSP), efficient storage management, and a user-friendly interface for live monitoring and historical playback.

## 2. System Architecture Overview
The system will follow a client-server architecture.

### Backend Service (The Engine)
- Responsible for persistent connections to cameras, stream processing, file writing, and storage rotation.

### Fronend Client (The Dashboard)
- A web-based or desktop interface for user interaction, configuration, and media consumption.

### Database
 - Storage (PostgreSQL) for configuration data and file indexing.

### Storage Layer
- The physical or network file system where video segments are saved.

## 3. Functional Requirements
### 3.1. Camera Management (Frontend & Backend)
Protocol Support: The system must support RTSP (Real-Time Streaming Protocol) over TCP and UDP.

Camera Onboarding: Users must be able to add cameras by providing:

IP Address / Hostname

Port (Default: 554)

RTSP Stream Path (e.g., /cam/realmonitor)

Authentication Credentials (Username/Password)

Connection Testing: The UI should provide a "Test Connection" button to validate RTSP streams before saving.

Status Monitoring: Dashboard must display connectivity status (Online/Offline/Error) for all configured cameras.

### 3.2. Storage & Recording (Backend)
Stream Ingestion: The backend must maintain persistent RTSP connections to all active cameras.

Transmuxing/Transcoding: Video should ideally be transmuxed (container change only, e.g., H.264 raw -> MP4) to minimize CPU usage, rather than transcoded, unless format incompatibility exists.

File Segmentation: Video streams must be chunked into discrete files (e.g., 15-minute segments) to prevent data loss in case of crashes.

Directory Structure: Files should be organized logically on the disk.

Proposed Structure: /{StorageRoot}/{CameraName}/{YYYY-MM-DD}/{HH-MM-SS}.mp4

Retention Policy (FIFO): Users must be able to define storage limits (e.g., "Keep last 30 days" or "Max 2TB"). The backend must automatically delete the oldest footage when limits are reached.

### 3.3. Live View (Frontend)
Multi-Camera Grid: Users can view multiple camera feeds simultaneously in a grid layout (e.g., 2x2, 3x3).

Low Latency: The specific requirement is to minimize latency between the RTSP source and the browser display (targeting <2 seconds).

Technical Note: Consider using WebRTC or MSE (Media Source Extensions) for browser playback.

### 3.4. Playback & Review (Frontend)
Timeline Navigation: A visual timeline allowing users to "scrub" through recorded footage.

Calendar View: Select specific dates to filter available footage.

Export/Download: Ability to download specific video segments to the local client device.

## 4. Non-Functional Requirements
### 4.1. Performance

#### Throughput: 
- The backend must handle simultaneous writing of at least [16] cameras at 1080p/30fps without frame drops.
- Multi-node processing across multiple computers and file servers should be considered. Separation of procesing with a centralized database for orchestration is recommended.


#### Resource Efficiency: 
Memory leaks in the stream ingestion service are critical failures; the service must be stable for 24/7 operation.


### 4.2. Compatibility
- Camera Hardware: Primary validation targets are Synology and Amcrest IP cameras.

### Codecs: Support for H.264 and H.265 (HEVC) video compression standards.

### 4.3. Security
- Credentials: Camera passwords must be stored securely (encrypted at rest).

- Access Control: Simple authentication mechanism for accessing the Frontend UI.
