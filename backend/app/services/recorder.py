"""FFmpeg-based video recording service."""

import asyncio
import logging
import os
import shutil
import signal
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.models import Camera

logger = logging.getLogger(__name__)
settings = get_settings()


class RecordingState(str, Enum):
    """Recording state."""

    STOPPED = "stopped"
    STARTING = "starting"
    RECORDING = "recording"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class CameraRecorder:
    """Manages FFmpeg recording for a single camera."""

    def __init__(
        self,
        camera: Camera,
        storage_root: Optional[Path] = None,
        segment_duration_minutes: int = 15,
    ):
        self.camera = camera
        self.storage_root = storage_root or Path(settings.storage_root)
        self.segment_duration = segment_duration_minutes * 60  # Convert to seconds

        self._process: Optional[asyncio.subprocess.Process] = None
        self._state = RecordingState.STOPPED
        self._error_message: Optional[str] = None
        self._start_time: Optional[datetime] = None
        self._current_segment_path: Optional[Path] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 5  # seconds

    @property
    def state(self) -> RecordingState:
        """Get current recording state."""
        return self._state

    @property
    def error_message(self) -> Optional[str]:
        """Get error message if in error state."""
        return self._error_message

    @property
    def is_recording(self) -> bool:
        """Check if actively recording."""
        return self._state in (RecordingState.RECORDING, RecordingState.RECONNECTING)

    @property
    def output_directory(self) -> Path:
        """Get output directory for this camera."""
        # Sanitize camera name for filesystem
        safe_name = "".join(
            c if c.isalnum() or c in ("-", "_") else "_"
            for c in self.camera.name
        )
        return self.storage_root / safe_name

    def _get_segment_pattern(self) -> str:
        """Get the segment filename pattern for FFmpeg."""
        today = datetime.now().strftime("%Y-%m-%d")
        output_dir = self.output_directory / today
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / "%H-%M-%S.mp4")

    def _build_ffmpeg_command(self) -> list[str]:
        """Build FFmpeg command for recording."""
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("FFmpeg not found in PATH")

        segment_pattern = self._get_segment_pattern()

        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel", "warning",
            # Input options
            "-rtsp_transport", self.camera.transport,
            "-i", self.camera.rtsp_url,
            # Output options - transmux only (no re-encoding)
            "-c", "copy",
            # Segment options
            "-f", "segment",
            "-segment_time", str(self.segment_duration),
            "-segment_format", "mp4",
            "-reset_timestamps", "1",
            "-strftime", "1",
            # Output pattern
            segment_pattern,
        ]

        return cmd

    async def start(self) -> bool:
        """Start recording."""
        if self._state == RecordingState.RECORDING:
            logger.warning(f"Camera '{self.camera.name}' is already recording")
            return True

        self._state = RecordingState.STARTING
        self._error_message = None
        self._reconnect_attempts = 0

        try:
            await self._start_ffmpeg()
            return True
        except Exception as e:
            logger.error(f"Failed to start recording for '{self.camera.name}': {e}")
            self._state = RecordingState.ERROR
            self._error_message = str(e)
            return False

    async def _start_ffmpeg(self) -> None:
        """Start the FFmpeg process."""
        cmd = self._build_ffmpeg_command()
        logger.info(f"Starting recording for '{self.camera.name}'")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._state = RecordingState.RECORDING
        self._start_time = datetime.now()

        # Start monitoring task
        asyncio.create_task(self._monitor_process())

    async def _monitor_process(self) -> None:
        """Monitor FFmpeg process and handle failures."""
        if not self._process:
            return

        try:
            stdout, stderr = await self._process.communicate()

            if self._state == RecordingState.STOPPED:
                # Normal stop, don't reconnect
                return

            exit_code = self._process.returncode
            if exit_code != 0:
                error_msg = stderr.decode().strip() if stderr else f"Exit code {exit_code}"
                logger.warning(
                    f"FFmpeg stopped for '{self.camera.name}': {error_msg}"
                )

                # Try to reconnect
                await self._handle_disconnect()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Error monitoring FFmpeg for '{self.camera.name}'")
            self._error_message = str(e)
            await self._handle_disconnect()

    async def _handle_disconnect(self) -> None:
        """Handle stream disconnection with reconnection logic."""
        if self._state == RecordingState.STOPPED:
            return

        self._reconnect_attempts += 1

        if self._reconnect_attempts > self._max_reconnect_attempts:
            logger.error(
                f"Max reconnect attempts reached for '{self.camera.name}'"
            )
            self._state = RecordingState.ERROR
            self._error_message = "Max reconnection attempts exceeded"
            return

        self._state = RecordingState.RECONNECTING
        delay = self._reconnect_delay * self._reconnect_attempts  # Exponential backoff

        logger.info(
            f"Reconnecting to '{self.camera.name}' in {delay}s "
            f"(attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})"
        )

        await asyncio.sleep(delay)

        if self._state == RecordingState.STOPPED:
            return

        try:
            await self._start_ffmpeg()
            self._reconnect_attempts = 0  # Reset on success
            logger.info(f"Reconnected to '{self.camera.name}'")
        except Exception as e:
            logger.error(f"Reconnection failed for '{self.camera.name}': {e}")
            await self._handle_disconnect()

    async def stop(self) -> None:
        """Stop recording."""
        if self._state == RecordingState.STOPPED:
            return

        logger.info(f"Stopping recording for '{self.camera.name}'")
        self._state = RecordingState.STOPPED

        if self._process:
            try:
                # Send SIGINT for graceful shutdown
                self._process.send_signal(signal.SIGINT)
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Force kill if graceful shutdown fails
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                pass  # Process already terminated
            finally:
                self._process = None

        self._start_time = None

    def get_status(self) -> dict:
        """Get recording status."""
        return {
            "camera_id": self.camera.id,
            "camera_name": self.camera.name,
            "state": self._state.value,
            "error_message": self._error_message,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "output_directory": str(self.output_directory),
            "reconnect_attempts": self._reconnect_attempts,
        }


class RecordingManager:
    """Manages recording for multiple cameras."""

    def __init__(self):
        self._recorders: dict[int, CameraRecorder] = {}

    def get_recorder(self, camera: Camera) -> CameraRecorder:
        """Get or create recorder for a camera."""
        if camera.id not in self._recorders:
            self._recorders[camera.id] = CameraRecorder(camera)
        return self._recorders[camera.id]

    async def start_recording(self, camera: Camera) -> bool:
        """Start recording for a camera."""
        recorder = self.get_recorder(camera)
        return await recorder.start()

    async def stop_recording(self, camera_id: int) -> bool:
        """Stop recording for a camera."""
        if camera_id not in self._recorders:
            return False
        await self._recorders[camera_id].stop()
        return True

    def get_status(self, camera_id: int) -> Optional[dict]:
        """Get recording status for a camera."""
        if camera_id not in self._recorders:
            return None
        return self._recorders[camera_id].get_status()

    def get_all_status(self) -> list[dict]:
        """Get status of all recorders."""
        return [r.get_status() for r in self._recorders.values()]

    async def stop_all(self) -> None:
        """Stop all recordings."""
        for recorder in self._recorders.values():
            await recorder.stop()

    def is_recording(self, camera_id: int) -> bool:
        """Check if camera is recording."""
        if camera_id not in self._recorders:
            return False
        return self._recorders[camera_id].is_recording


# Global recording manager
recording_manager = RecordingManager()
