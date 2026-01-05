#!/usr/bin/env python3
"""
Dedicated Stream Manager Service.

Single process that owns all camera ffmpeg streams. Backend workers
communicate with this service via HTTP to start/stop streams.

This solves the multi-worker problem where each uvicorn worker would
start its own ffmpeg processes.
"""

import asyncio
import logging
import os
import shutil
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("stream-manager")

# Settings
STORAGE_ROOT = Path(os.environ.get("STORAGE_ROOT", "/data/storage"))
STREAM_MANAGER_PORT = int(os.environ.get("STREAM_MANAGER_PORT", "8001"))
SEGMENT_DURATION_MINUTES = int(os.environ.get("SEGMENT_DURATION_MINUTES", "15"))


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


class CameraStream:
    """Manages a single camera's ffmpeg process."""

    def __init__(self, config: CameraConfig, storage_root: Path):
        self.config = config
        self.storage_root = storage_root
        self._process: Optional[asyncio.subprocess.Process] = None
        self._state = "stopped"
        self._error_message: Optional[str] = None
        self._start_time: Optional[datetime] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10

    @property
    def safe_name(self) -> str:
        return "".join(
            c if c.isalnum() or c in ("-", "_") else "_"
            for c in self.config.name
        )

    @property
    def hls_directory(self) -> Path:
        return self.storage_root / ".streams" / str(self.config.id)

    @property
    def recording_directory(self) -> Path:
        return self.storage_root / self.safe_name

    @property
    def playlist_path(self) -> Path:
        return self.hls_directory / "playlist.m3u8"

    @property
    def is_running(self) -> bool:
        return self._state in ("running", "reconnecting")

    def _get_recording_pattern(self) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output_dir = self.recording_directory / today
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / "%H-%M-%S.mp4")

    def _build_command(self) -> list[str]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg not found")

        self.hls_directory.mkdir(parents=True, exist_ok=True)

        # Clean old HLS segments
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

        segment_duration = SEGMENT_DURATION_MINUTES * 60

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "warning",
            "-rtsp_transport", self.config.transport,
            "-fflags", "+genpts+discardcorrupt+igndts",
            "-flags", "low_delay",
            "-i", self.config.rtsp_url,
            "-max_muxing_queue_size", "1024",
            "-avoid_negative_ts", "make_zero",
            # HLS output
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
        ]

        # Recording output
        if self.config.recording_enabled:
            pattern = self._get_recording_pattern()
            cmd.extend([
                "-c:v", "copy",
                "-c:a", "aac",
                "-ar", "44100",
                "-f", "segment",
                "-segment_time", str(segment_duration),
                "-segment_format", "mp4",
                "-segment_format_options", "movflags=frag_keyframe+empty_moov+default_base_moof",
                "-reset_timestamps", "1",
                "-strftime", "1",
                pattern,
            ])

        return cmd

    async def start(self) -> bool:
        if self._state == "running":
            return True

        self._state = "starting"
        self._error_message = None
        self._reconnect_attempts = 0

        try:
            await self._start_ffmpeg()
            return True
        except Exception as e:
            logger.error(f"Failed to start stream for '{self.config.name}': {e}")
            self._state = "error"
            self._error_message = str(e)
            return False

    async def _start_ffmpeg(self) -> None:
        cmd = self._build_command()
        logger.info(f"Starting stream for '{self.config.name}'")

        env = os.environ.copy()
        env["TZ"] = "UTC"

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        self._state = "running"
        self._start_time = datetime.now(timezone.utc)
        self._monitor_task = asyncio.create_task(self._monitor())
        logger.info(f"Stream started for '{self.config.name}' (pid={self._process.pid})")

    async def _monitor(self) -> None:
        if not self._process:
            return

        try:
            stdout, stderr = await self._process.communicate()

            if self._state == "stopped":
                return

            exit_code = self._process.returncode
            if exit_code != 0:
                error = stderr.decode()[-500:] if stderr else f"Exit code {exit_code}"
                logger.warning(f"FFmpeg stopped for '{self.config.name}': {error}")
                self._error_message = error
                await self._reconnect()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Monitor error for '{self.config.name}'")
            self._error_message = str(e)
            await self._reconnect()

    async def _reconnect(self) -> None:
        if self._state == "stopped":
            return

        self._reconnect_attempts += 1
        if self._reconnect_attempts > self._max_reconnect_attempts:
            logger.error(f"Max reconnects for '{self.config.name}'")
            self._state = "error"
            return

        self._state = "reconnecting"
        delay = min(5 * self._reconnect_attempts, 60)
        logger.info(f"Reconnecting '{self.config.name}' in {delay}s")

        await asyncio.sleep(delay)

        if self._state == "stopped":
            return

        try:
            await self._start_ffmpeg()
            self._reconnect_attempts = 0
        except Exception as e:
            logger.error(f"Reconnect failed for '{self.config.name}': {e}")
            await self._reconnect()

    async def stop(self) -> None:
        if self._state == "stopped":
            return

        logger.info(f"Stopping stream for '{self.config.name}'")
        self._state = "stopped"

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        if self._process:
            try:
                self._process.send_signal(signal.SIGINT)
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    self._process.kill()
                    await self._process.wait()
                except ProcessLookupError:
                    pass
            self._process = None

    def get_status(self) -> StreamStatus:
        return StreamStatus(
            camera_id=self.config.id,
            camera_name=self.config.name,
            state=self._state,
            is_running=self.is_running,
            is_recording=self.is_running and self.config.recording_enabled,
            start_time=self._start_time.isoformat() if self._start_time else None,
            error_message=self._error_message,
            pid=self._process.pid if self._process else None,
        )


class StreamManager:
    """Manages all camera streams."""

    def __init__(self):
        self._streams: dict[int, CameraStream] = {}
        self._locks: dict[int, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    def _get_lock(self, camera_id: int) -> asyncio.Lock:
        """Get or create a lock for a camera (thread-safe)."""
        if camera_id not in self._locks:
            self._locks[camera_id] = asyncio.Lock()
        return self._locks[camera_id]

    async def start_stream(self, config: CameraConfig) -> StreamStatus:
        # Get camera-specific lock to prevent race conditions
        async with self._global_lock:
            lock = self._get_lock(config.id)

        async with lock:
            if config.id in self._streams:
                stream = self._streams[config.id]
                if stream.is_running:
                    logger.debug(f"Stream for '{config.name}' already running")
                    return stream.get_status()
                # Config may have changed, recreate
                await stream.stop()

            stream = CameraStream(config, STORAGE_ROOT)
            self._streams[config.id] = stream
            await stream.start()
            return stream.get_status()

    async def stop_stream(self, camera_id: int) -> bool:
        if camera_id not in self._streams:
            return False
        await self._streams[camera_id].stop()
        return True

    def get_status(self, camera_id: int) -> Optional[StreamStatus]:
        if camera_id not in self._streams:
            return None
        return self._streams[camera_id].get_status()

    def get_all_status(self) -> list[StreamStatus]:
        return [s.get_status() for s in self._streams.values()]

    async def stop_all(self) -> None:
        for stream in self._streams.values():
            await stream.stop()


# FastAPI app
app = FastAPI(title="Stream Manager")
manager = StreamManager()


@app.get("/health")
async def health():
    return {"status": "ok", "streams": len(manager._streams)}


@app.post("/streams/start", response_model=StreamStatus)
async def start_stream(config: CameraConfig):
    """Start streaming for a camera."""
    return await manager.start_stream(config)


@app.post("/streams/{camera_id}/stop")
async def stop_stream(camera_id: int):
    """Stop streaming for a camera."""
    if not await manager.stop_stream(camera_id):
        raise HTTPException(404, "Stream not found")
    return {"status": "stopped"}


@app.get("/streams/{camera_id}", response_model=StreamStatus)
async def get_stream_status(camera_id: int):
    """Get status of a camera stream."""
    status = manager.get_status(camera_id)
    if not status:
        raise HTTPException(404, "Stream not found")
    return status


@app.get("/streams", response_model=list[StreamStatus])
async def list_streams():
    """List all streams."""
    return manager.get_all_status()


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down stream manager")
    await manager.stop_all()


if __name__ == "__main__":
    logger.info(f"Starting Stream Manager on port {STREAM_MANAGER_PORT}")
    logger.info(f"Storage root: {STORAGE_ROOT}")
    uvicorn.run(app, host="0.0.0.0", port=STREAM_MANAGER_PORT, log_level="info")
