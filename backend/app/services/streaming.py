"""HLS streaming service for live camera feeds."""

import asyncio
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.models.camera import Camera

logger = logging.getLogger(__name__)
settings = get_settings()


class HLSStream:
    """Manages HLS streaming for a single camera."""

    def __init__(self, camera: Camera, stream_root: Path):
        self.camera = camera
        self.stream_dir = stream_root / str(camera.id)
        self.process: Optional[subprocess.Popen] = None
        self._running = False

    @property
    def playlist_path(self) -> Path:
        """Get path to the HLS playlist."""
        return self.stream_dir / "playlist.m3u8"

    @property
    def is_running(self) -> bool:
        """Check if stream is running."""
        return self._running and self.process is not None and self.process.poll() is None

    def _get_rtsp_url(self) -> str:
        """Build RTSP URL from camera config."""
        if self.camera.username and self.camera.password:
            auth = f"{self.camera.username}:{self.camera.password}@"
        else:
            auth = ""
        return f"rtsp://{auth}{self.camera.host}:{self.camera.port}{self.camera.path}"

    def start(self) -> bool:
        """Start HLS streaming."""
        if self.is_running:
            return True

        # Ensure stream directory exists
        self.stream_dir.mkdir(parents=True, exist_ok=True)

        # Clean up old segments
        for f in self.stream_dir.glob("*.ts"):
            f.unlink()
        if self.playlist_path.exists():
            self.playlist_path.unlink()

        rtsp_url = self._get_rtsp_url()
        ffmpeg_path = shutil.which("ffmpeg")

        if not ffmpeg_path:
            logger.error("FFmpeg not found in PATH")
            return False

        # FFmpeg command for Low-Latency HLS
        cmd = [
            ffmpeg_path,
            "-rtsp_transport", self.camera.transport,
            "-fflags", "+genpts+discardcorrupt",
            "-i", rtsp_url,
            # Video: copy codec (no transcoding for low latency)
            "-c:v", "copy",
            # Audio: copy or transcode to AAC
            "-c:a", "aac",
            "-ar", "44100",
            # HLS options for low latency
            "-f", "hls",
            "-hls_time", "2",  # 2-second segments
            "-hls_list_size", "5",  # Keep only 5 segments in playlist
            "-hls_flags", "delete_segments+append_list",
            "-hls_segment_filename", str(self.stream_dir / "segment%03d.ts"),
            str(self.playlist_path),
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._running = True
            logger.info(f"Started HLS stream for camera {self.camera.id}: {self.camera.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to start HLS stream for camera {self.camera.id}: {e}")
            return False

    def stop(self) -> None:
        """Stop HLS streaming."""
        self._running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            logger.info(f"Stopped HLS stream for camera {self.camera.id}")

        # Clean up stream files
        if self.stream_dir.exists():
            for f in self.stream_dir.glob("*.ts"):
                try:
                    f.unlink()
                except OSError:
                    pass
            if self.playlist_path.exists():
                try:
                    self.playlist_path.unlink()
                except OSError:
                    pass


class StreamingManager:
    """Manages HLS streams for all cameras."""

    def __init__(self):
        self.streams: dict[int, HLSStream] = {}
        self.stream_root = Path(settings.storage_root) / ".streams"

    def get_stream(self, camera: Camera) -> HLSStream:
        """Get or create HLS stream for a camera."""
        if camera.id not in self.streams:
            self.streams[camera.id] = HLSStream(camera, self.stream_root)
        return self.streams[camera.id]

    def start_stream(self, camera: Camera) -> bool:
        """Start streaming for a camera."""
        stream = self.get_stream(camera)
        return stream.start()

    def stop_stream(self, camera_id: int) -> None:
        """Stop streaming for a camera."""
        if camera_id in self.streams:
            self.streams[camera_id].stop()
            del self.streams[camera_id]

    def get_playlist_path(self, camera_id: int) -> Optional[Path]:
        """Get path to camera's HLS playlist."""
        if camera_id in self.streams:
            stream = self.streams[camera_id]
            if stream.is_running and stream.playlist_path.exists():
                return stream.playlist_path
        return None

    def get_segment_path(self, camera_id: int, segment_name: str) -> Optional[Path]:
        """Get path to a specific HLS segment."""
        if camera_id in self.streams:
            stream = self.streams[camera_id]
            segment_path = stream.stream_dir / segment_name
            if stream.is_running and segment_path.exists():
                return segment_path
        return None

    async def stop_all(self) -> None:
        """Stop all streams."""
        for camera_id in list(self.streams.keys()):
            self.stop_stream(camera_id)


# Global streaming manager
streaming_manager = StreamingManager()
