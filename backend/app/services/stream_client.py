"""
Client for communicating with the Stream Manager service.

The Stream Manager runs as a separate process and owns all ffmpeg streams.
This client allows backend workers to start/stop streams without each
worker spawning its own ffmpeg processes.
"""

import logging
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import httpx
from pydantic import BaseModel

if TYPE_CHECKING:
    from app.models import Camera

logger = logging.getLogger(__name__)

STREAM_MANAGER_URL = os.environ.get("STREAM_MANAGER_URL", "http://stream-manager:8001")
STORAGE_ROOT = Path(os.environ.get("STORAGE_ROOT", "/data/storage"))


class CameraConfig(BaseModel):
    """Camera configuration for starting a stream."""
    id: int
    name: str
    rtsp_url: str
    transport: str = "tcp"
    recording_enabled: bool = True


class StreamStatus(BaseModel):
    """Status of a camera stream."""
    camera_id: int
    camera_name: str
    state: str
    is_running: bool
    is_recording: bool
    start_time: Optional[str] = None
    error_message: Optional[str] = None
    pid: Optional[int] = None


class StreamManagerClient:
    """Client for the Stream Manager service."""

    def __init__(self, base_url: str = STREAM_MANAGER_URL):
        self.base_url = base_url
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if stream manager is healthy."""
        try:
            client = await self._get_client()
            resp = await client.get("/health")
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Stream manager health check failed: {e}")
            return False

    async def start_stream(self, config: CameraConfig) -> Optional[StreamStatus]:
        """Start streaming for a camera."""
        try:
            client = await self._get_client()
            resp = await client.post("/streams/start", json=config.model_dump())
            if resp.status_code == 200:
                return StreamStatus(**resp.json())
            logger.error(f"Failed to start stream: {resp.status_code} {resp.text}")
            return None
        except Exception as e:
            logger.error(f"Error starting stream for camera {config.id}: {e}")
            return None

    async def stop_stream(self, camera_id: int) -> bool:
        """Stop streaming for a camera."""
        try:
            client = await self._get_client()
            resp = await client.post(f"/streams/{camera_id}/stop")
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Error stopping stream for camera {camera_id}: {e}")
            return False

    async def get_stream_status(self, camera_id: int) -> Optional[StreamStatus]:
        """Get status of a camera stream."""
        try:
            client = await self._get_client()
            resp = await client.get(f"/streams/{camera_id}")
            if resp.status_code == 200:
                return StreamStatus(**resp.json())
            return None
        except Exception as e:
            logger.debug(f"Error getting stream status for camera {camera_id}: {e}")
            return None

    async def list_streams(self) -> list[StreamStatus]:
        """List all streams."""
        try:
            client = await self._get_client()
            resp = await client.get("/streams")
            if resp.status_code == 200:
                return [StreamStatus(**s) for s in resp.json()]
            return []
        except Exception as e:
            logger.error(f"Error listing streams: {e}")
            return []


# Global client instance
stream_client = StreamManagerClient()


class StreamManagerFacade:
    """
    Facade that provides the same interface as the old CameraStreamManager
    but delegates to the remote stream manager service.

    This allows minimal changes to existing code that uses stream_manager.
    """

    def __init__(self, client: StreamManagerClient):
        self._client = client
        self._streams: dict[int, StreamStatus] = {}  # Local cache

    async def start_stream(self, camera: "Camera", recording_enabled: bool = True) -> bool:
        """Start streaming for a camera."""
        config = CameraConfig(
            id=camera.id,
            name=camera.name,
            rtsp_url=camera.rtsp_url,
            transport=camera.transport,
            recording_enabled=recording_enabled,
        )
        status = await self._client.start_stream(config)
        if status:
            self._streams[camera.id] = status
            return status.is_running
        return False

    async def stop_stream(self, camera_id: int) -> bool:
        """Stop streaming for a camera."""
        result = await self._client.stop_stream(camera_id)
        if camera_id in self._streams:
            del self._streams[camera_id]
        return result

    async def restart_stream(self, camera_id: int) -> bool:
        """Restart stream - not directly supported, return False to trigger fresh start."""
        # The caller will start fresh if this returns False
        await self.stop_stream(camera_id)
        return False

    def is_running(self, camera_id: int) -> bool:
        """Check if camera stream is running (uses cached state)."""
        if camera_id in self._streams:
            return self._streams[camera_id].is_running
        return False

    def is_recording(self, camera_id: int) -> bool:
        """Check if camera is recording (uses cached state)."""
        if camera_id in self._streams:
            return self._streams[camera_id].is_recording
        return False

    def get_status(self, camera_id: int) -> Optional[dict]:
        """Get stream status for a camera."""
        if camera_id in self._streams:
            s = self._streams[camera_id]
            return {
                "camera_id": s.camera_id,
                "camera_name": s.camera_name,
                "state": s.state,
                "is_running": s.is_running,
                "is_recording": s.is_recording,
                "start_time": s.start_time,
                "error_message": s.error_message,
            }
        return None

    def get_all_status(self) -> list[dict]:
        """Get status of all streams."""
        return [
            {
                "camera_id": s.camera_id,
                "camera_name": s.camera_name,
                "state": s.state,
                "is_running": s.is_running,
                "is_recording": s.is_recording,
                "start_time": s.start_time,
                "error_message": s.error_message,
            }
            for s in self._streams.values()
        ]

    def get_playlist_path(self, camera_id: int) -> Optional[Path]:
        """Get HLS playlist path for a camera."""
        playlist = STORAGE_ROOT / ".streams" / str(camera_id) / "playlist.m3u8"
        if playlist.exists():
            return playlist
        return None

    def get_segment_path(self, camera_id: int, segment_name: str) -> Optional[Path]:
        """Get HLS segment path."""
        segment = STORAGE_ROOT / ".streams" / str(camera_id) / segment_name
        if segment.exists():
            return segment
        return None

    async def stop_all(self) -> None:
        """Stop all streams."""
        for camera_id in list(self._streams.keys()):
            await self.stop_stream(camera_id)


# Global facade instance - provides same interface as old stream_manager
stream_manager = StreamManagerFacade(stream_client)
