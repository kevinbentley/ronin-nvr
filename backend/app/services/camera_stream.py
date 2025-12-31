"""
Unified camera streaming service.

Single FFmpeg process per camera that handles both:
- HLS live streaming
- MP4 recording

This uses only ONE RTSP connection per camera, avoiding conflicts.
"""

import asyncio
import logging
import os
import shutil
import signal
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.models import Camera
from app.utils import utc_now

logger = logging.getLogger(__name__)
settings = get_settings()


class StreamState(str, Enum):
    """Stream state."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class CameraStream:
    """
    Manages a single FFmpeg process for one camera.

    Outputs both HLS (for live view) and MP4 segments (for recording).
    """

    def __init__(
        self,
        camera: Camera,
        storage_root: Optional[Path] = None,
        segment_duration_minutes: int = 15,
    ):
        # Extract camera data upfront to avoid SQLAlchemy detached session issues
        self.camera_id = camera.id
        self.camera_name = camera.name
        self.camera_host = camera.host
        self.camera_port = camera.port
        self.camera_path = camera.path
        self.camera_username = camera.username
        self.camera_password = camera.password
        self.camera_transport = camera.transport
        self._rtsp_url = camera.rtsp_url

        self.storage_root = storage_root or Path(settings.storage_root)
        self.segment_duration = segment_duration_minutes * 60

        self._process: Optional[asyncio.subprocess.Process] = None
        self._state = StreamState.STOPPED
        self._error_message: Optional[str] = None
        self._start_time: Optional[datetime] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 5
        self._recording_enabled = True
        self._monitor_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._last_stderr_lines: list[str] = []

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state in (StreamState.RUNNING, StreamState.RECONNECTING)

    @property
    def is_recording(self) -> bool:
        return self.is_running and self._recording_enabled

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message

    @property
    def safe_camera_name(self) -> str:
        """Filesystem-safe camera name."""
        return "".join(
            c if c.isalnum() or c in ("-", "_") else "_"
            for c in self.camera_name
        )

    @property
    def recording_directory(self) -> Path:
        """Directory for MP4 recordings."""
        return self.storage_root / self.safe_camera_name

    @property
    def hls_directory(self) -> Path:
        """Directory for HLS segments."""
        return self.storage_root / ".streams" / str(self.camera_id)

    @property
    def log_directory(self) -> Path:
        """Directory for FFmpeg logs."""
        return self.storage_root / ".logs"

    @property
    def ffmpeg_log_path(self) -> Path:
        """Path to FFmpeg log file for this camera."""
        return self.log_directory / f"{self.safe_camera_name}_ffmpeg.log"

    @property
    def playlist_path(self) -> Path:
        """Path to HLS playlist."""
        return self.hls_directory / "playlist.m3u8"

    @property
    def rtsp_url(self) -> str:
        """Get RTSP URL."""
        return self._rtsp_url

    def _get_recording_pattern(self) -> str:
        """Get MP4 segment filename pattern.

        Uses UTC time for both directory and filename to ensure consistent
        timezone handling across Docker and native deployments.
        """
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output_dir = self.recording_directory / today_utc
        output_dir.mkdir(parents=True, exist_ok=True)
        # FFmpeg strftime uses system time, so we set TZ=UTC in the environment
        # when starting FFmpeg. For now, we use the pattern and document this.
        return str(output_dir / "%H-%M-%S.mp4")

    def _build_ffmpeg_command(self, clean_hls: bool = True) -> list[str]:
        """Build FFmpeg command with dual output.

        Args:
            clean_hls: Whether to clean old HLS segments. Set False on reconnect
                      to avoid 503 errors while FFmpeg restarts.
        """
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("FFmpeg not found in PATH")

        # Ensure HLS directory exists
        self.hls_directory.mkdir(parents=True, exist_ok=True)

        # Clean old HLS segments only on fresh start (not reconnect)
        if clean_hls:
            for f in self.hls_directory.glob("*.ts"):
                try:
                    f.unlink()
                except OSError:
                    pass
            if self.playlist_path.exists():
                try:
                    self.playlist_path.unlink()
                except OSError:
                    pass

        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel", "warning",
            # Input options - handle broken timestamps from cameras
            "-rtsp_transport", self.camera_transport,
            "-fflags", "+genpts+discardcorrupt+igndts",
            "-flags", "low_delay",
            "-i", self.rtsp_url,
            # Global output options for timestamp issues
            "-max_muxing_queue_size", "1024",
            "-avoid_negative_ts", "make_zero",
        ]

        # Output 1: HLS for live streaming
        # Use start_at_zero and reset_timestamps for cleaner segments
        cmd.extend([
            "-c:v", "copy",
            "-c:a", "aac",
            "-ar", "44100",
            "-f", "hls",
            "-hls_time", "2",
            "-hls_list_size", "10",
            "-hls_flags", "delete_segments+append_list+omit_endlist",
            "-start_at_zero",
            "-hls_segment_filename", str(self.hls_directory / "segment%03d.ts"),
            str(self.playlist_path),
        ])

        # Output 2: MP4 segments for recording (if enabled)
        if self._recording_enabled:
            recording_pattern = self._get_recording_pattern()
            cmd.extend([
                "-c:v", "copy",
                "-c:a", "aac",  # Transcode audio - some cameras use pcm_mulaw which MP4 doesn't support
                "-ar", "44100",
                "-f", "segment",
                "-segment_time", str(self.segment_duration),
                "-segment_format", "mp4",
                "-reset_timestamps", "1",
                "-strftime", "1",
                recording_pattern,
            ])

        return cmd

    async def start(self, recording_enabled: bool = True) -> bool:
        """Start the camera stream."""
        if self._state == StreamState.RUNNING:
            logger.warning(f"Camera '{self.camera_name}' stream already running")
            return True

        self._recording_enabled = recording_enabled
        self._state = StreamState.STARTING
        self._error_message = None
        self._reconnect_attempts = 0

        try:
            await self._start_ffmpeg()
            return True
        except Exception as e:
            logger.error(f"Failed to start stream for '{self.camera_name}': {e}")
            self._state = StreamState.ERROR
            self._error_message = str(e)
            return False

    async def _start_ffmpeg(self, is_reconnect: bool = False) -> None:
        """Start the FFmpeg process.

        Args:
            is_reconnect: If True, don't clean HLS segments to avoid 503 errors.
        """
        try:
            cmd = self._build_ffmpeg_command(clean_hls=not is_reconnect)
        except Exception as e:
            logger.error(f"Failed to build FFmpeg command for '{self.camera_name}': {e}")
            raise

        action = "Reconnecting" if is_reconnect else "Starting"
        logger.info(f"{action} stream for '{self.camera_name}' (recording={self._recording_enabled})")
        logger.info(f"RTSP URL: {self.rtsp_url}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        # Set TZ=UTC so FFmpeg's strftime outputs UTC time for filenames
        env = os.environ.copy()
        env["TZ"] = "UTC"

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        self._state = StreamState.RUNNING
        self._start_time = utc_now()
        self._last_stderr_lines: list[str] = []

        # Start monitoring tasks
        self._monitor_task = asyncio.create_task(self._monitor_process())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    async def _read_stderr(self) -> None:
        """Read FFmpeg stderr in real-time for debugging."""
        if not self._process or not self._process.stderr:
            return

        # Ensure log directory exists and open log file
        self.log_directory.mkdir(parents=True, exist_ok=True)
        log_file = None

        try:
            # Open log file in append mode with timestamp header
            log_file = open(self.ffmpeg_log_path, "a")
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"FFmpeg started at {datetime.now().isoformat()}\n")
            log_file.write(f"Camera: {self.camera_name} (ID: {self.camera_id})\n")
            log_file.write(f"RTSP URL: {self.rtsp_url}\n")
            log_file.write(f"{'='*60}\n")
            log_file.flush()

            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                decoded = line.decode().strip()
                if decoded:
                    # Write to log file with timestamp
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    log_file.write(f"[{timestamp}] {decoded}\n")
                    log_file.flush()

                    # Keep last 20 lines for diagnostics
                    self._last_stderr_lines.append(decoded)
                    if len(self._last_stderr_lines) > 20:
                        self._last_stderr_lines.pop(0)

                    # Log important messages to console
                    if any(kw in decoded.lower() for kw in [
                        'error', 'failed', 'refused', 'timeout', 'invalid',
                        'unauthorized', '401', '403', '404', 'connection'
                    ]):
                        logger.warning(f"FFmpeg [{self.camera_name}]: {decoded}")
                    else:
                        logger.debug(f"FFmpeg [{self.camera_name}]: {decoded}")

            # Write exit info
            if self._process:
                log_file.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] FFmpeg exited with code: {self._process.returncode}\n")
                log_file.flush()

        except asyncio.CancelledError:
            if log_file:
                log_file.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] Stream stopped by user\n")
        except Exception as e:
            logger.debug(f"Stderr reader ended for '{self.camera_name}': {e}")
            if log_file:
                log_file.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] Stderr reader error: {e}\n")
        finally:
            if log_file:
                log_file.close()

    async def _monitor_process(self) -> None:
        """Monitor FFmpeg process and handle failures."""
        if not self._process:
            return

        try:
            # Wait for process to exit
            await self._process.wait()

            if self._state == StreamState.STOPPED:
                return

            exit_code = self._process.returncode
            if exit_code != 0:
                error_msg = "\n".join(self._last_stderr_lines[-5:]) if self._last_stderr_lines else f"Exit code {exit_code}"
                logger.warning(f"FFmpeg stopped for '{self.camera_name}' (exit={exit_code}): {error_msg}")
                self._error_message = error_msg
                await self._handle_disconnect()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Error monitoring FFmpeg for '{self.camera_name}'")
            self._error_message = str(e)
            await self._handle_disconnect()

    async def _handle_disconnect(self) -> None:
        """Handle stream disconnection with reconnection."""
        if self._state == StreamState.STOPPED:
            return

        self._reconnect_attempts += 1

        if self._reconnect_attempts > self._max_reconnect_attempts:
            logger.error(f"Max reconnect attempts reached for '{self.camera_name}'")
            self._state = StreamState.ERROR
            self._error_message = "Max reconnection attempts exceeded"
            return

        self._state = StreamState.RECONNECTING
        delay = min(self._reconnect_delay * self._reconnect_attempts, 60)

        logger.info(
            f"Reconnecting to '{self.camera_name}' in {delay}s "
            f"(attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})"
        )

        await asyncio.sleep(delay)

        if self._state == StreamState.STOPPED:
            return

        try:
            await self._start_ffmpeg(is_reconnect=True)
            self._reconnect_attempts = 0
            logger.info(f"Reconnected to '{self.camera_name}'")
        except Exception as e:
            logger.error(f"Reconnection failed for '{self.camera_name}': {e}")
            await self._handle_disconnect()

    async def stop(self) -> None:
        """Stop the camera stream."""
        if self._state == StreamState.STOPPED:
            return

        logger.info(f"Stopping stream for '{self.camera_name}'")
        self._state = StreamState.STOPPED

        # Cancel monitoring tasks
        for task in [self._monitor_task, self._stderr_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._monitor_task = None
        self._stderr_task = None

        if self._process:
            try:
                self._process.send_signal(signal.SIGINT)
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                pass
            finally:
                self._process = None

        self._start_time = None

    async def restart(self) -> bool:
        """Restart the stream."""
        await self.stop()
        await asyncio.sleep(0.5)
        return await self.start(self._recording_enabled)

    def get_playlist_path(self) -> Optional[Path]:
        """Get HLS playlist path if available."""
        if self.is_running and self.playlist_path.exists():
            return self.playlist_path
        return None

    def get_segment_path(self, segment_name: str) -> Optional[Path]:
        """Get path to a specific HLS segment."""
        segment_path = self.hls_directory / segment_name
        if self.is_running and segment_path.exists():
            return segment_path
        return None

    def get_status(self) -> dict:
        """Get stream status."""
        return {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "state": self._state.value,
            "is_running": self.is_running,
            "is_recording": self.is_recording,
            "error_message": self._error_message,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "recording_directory": str(self.recording_directory),
            "reconnect_attempts": self._reconnect_attempts,
            "ffmpeg_output": self._last_stderr_lines[-10:] if self._last_stderr_lines else [],
            "ffmpeg_log_path": str(self.ffmpeg_log_path),
        }


class CameraStreamManager:
    """Manages streams for all cameras."""

    def __init__(self):
        self._streams: dict[int, CameraStream] = {}

    def get_stream(self, camera: Camera) -> CameraStream:
        """Get or create stream for a camera."""
        if camera.id not in self._streams:
            self._streams[camera.id] = CameraStream(camera)
        return self._streams[camera.id]

    async def start_stream(self, camera: Camera, recording_enabled: bool = True) -> bool:
        """Start streaming (and optionally recording) for a camera."""
        stream = self.get_stream(camera)
        return await stream.start(recording_enabled)

    async def stop_stream(self, camera_id: int) -> bool:
        """Stop streaming for a camera."""
        if camera_id not in self._streams:
            return False
        await self._streams[camera_id].stop()
        return True

    async def restart_stream(self, camera_id: int) -> bool:
        """Restart stream for a camera."""
        if camera_id not in self._streams:
            return False
        return await self._streams[camera_id].restart()

    def get_status(self, camera_id: int) -> Optional[dict]:
        """Get stream status for a camera."""
        if camera_id not in self._streams:
            return None
        return self._streams[camera_id].get_status()

    def get_all_status(self) -> list[dict]:
        """Get status of all streams."""
        return [s.get_status() for s in self._streams.values()]

    def is_running(self, camera_id: int) -> bool:
        """Check if camera stream is running."""
        if camera_id not in self._streams:
            return False
        return self._streams[camera_id].is_running

    def is_recording(self, camera_id: int) -> bool:
        """Check if camera is recording."""
        if camera_id not in self._streams:
            return False
        return self._streams[camera_id].is_recording

    def get_playlist_path(self, camera_id: int) -> Optional[Path]:
        """Get HLS playlist path for a camera."""
        if camera_id not in self._streams:
            return None
        return self._streams[camera_id].get_playlist_path()

    def get_segment_path(self, camera_id: int, segment_name: str) -> Optional[Path]:
        """Get HLS segment path."""
        if camera_id not in self._streams:
            return None
        return self._streams[camera_id].get_segment_path(segment_name)

    async def stop_all(self) -> None:
        """Stop all streams."""
        for stream in self._streams.values():
            await stream.stop()


# Global stream manager
stream_manager = CameraStreamManager()
