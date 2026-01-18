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
import time
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

# Watchdog settings - detect and restart stuck streams
WATCHDOG_CHECK_INTERVAL = int(os.environ.get("WATCHDOG_CHECK_INTERVAL", "15"))
WATCHDOG_STALE_THRESHOLD = int(os.environ.get("WATCHDOG_STALE_THRESHOLD", "30"))
WATCHDOG_GRACE_PERIOD = int(os.environ.get("WATCHDOG_GRACE_PERIOD", "30"))


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
        self._recording_date: Optional[str] = None  # Track the date used for recording path

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
        self._recording_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output_dir = self.recording_directory / self._recording_date
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / "%H-%M-%S.mp4")

    def needs_date_rollover(self) -> bool:
        """Check if the stream needs to be restarted due to date change."""
        if not self._recording_date or not self.is_running:
            return False
        if not self.config.recording_enabled:
            return False
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return current_date != self._recording_date

    def is_stale(self) -> bool:
        """Check if stream is stuck (running but not producing output).

        Returns True if the playlist file hasn't been updated within the
        stale threshold, indicating FFmpeg may be alive but not receiving data.
        """
        if self._state != "running":
            return False

        # Grace period after start - don't check streams that just started
        if self._start_time:
            age = (datetime.now(timezone.utc) - self._start_time).total_seconds()
            if age < WATCHDOG_GRACE_PERIOD:
                return False

        # Check if playlist exists and when it was last modified
        if not self.playlist_path.exists():
            return True  # No playlist means definitely stale

        try:
            mtime = self.playlist_path.stat().st_mtime
            age = time.time() - mtime
            return age > WATCHDOG_STALE_THRESHOLD
        except OSError:
            return True  # Can't stat file, assume stale

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
        self._date_rollover_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None

    async def start_date_rollover_monitor(self) -> None:
        """Start the background task that monitors for date changes."""
        if self._date_rollover_task is None:
            self._date_rollover_task = asyncio.create_task(self._date_rollover_loop())
            logger.info("Date rollover monitor started")

    async def stop_date_rollover_monitor(self) -> None:
        """Stop the date rollover monitor task."""
        if self._date_rollover_task:
            self._date_rollover_task.cancel()
            try:
                await self._date_rollover_task
            except asyncio.CancelledError:
                pass
            self._date_rollover_task = None

    async def _date_rollover_loop(self) -> None:
        """Periodically check for date changes and restart streams as needed."""
        while True:
            try:
                # Check every 30 seconds
                await asyncio.sleep(30)
                await self._check_date_rollover()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in date rollover check: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _check_date_rollover(self) -> None:
        """Check all streams and restart any that need date rollover."""
        streams_to_restart = []

        for camera_id, stream in self._streams.items():
            if stream.needs_date_rollover():
                streams_to_restart.append((camera_id, stream))

        if streams_to_restart:
            logger.info(f"Date rollover: restarting {len(streams_to_restart)} streams")

        for camera_id, stream in streams_to_restart:
            try:
                config = stream.config
                logger.info(f"Restarting stream '{config.name}' for date rollover")
                await stream.stop()
                # Small delay to ensure clean shutdown
                await asyncio.sleep(1)
                await stream.start()
                logger.info(f"Stream '{config.name}' restarted for new date")
            except Exception as e:
                logger.error(f"Failed to restart stream {camera_id} for date rollover: {e}")

    async def start_watchdog(self) -> None:
        """Start the background task that detects stuck streams."""
        if self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())
            logger.info("Stream watchdog started")

    async def stop_watchdog(self) -> None:
        """Stop the watchdog task."""
        if self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None

    async def _watchdog_loop(self) -> None:
        """Periodically check for stuck streams and restart them."""
        while True:
            try:
                await asyncio.sleep(WATCHDOG_CHECK_INTERVAL)
                await self._check_stale_streams()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in watchdog check: {e}")
                await asyncio.sleep(60)

    async def _check_stale_streams(self) -> None:
        """Check all streams and restart any that are stuck."""
        stale_streams = []

        for camera_id, stream in self._streams.items():
            if stream.is_stale():
                stale_streams.append((camera_id, stream))

        for camera_id, stream in stale_streams:
            try:
                config = stream.config
                logger.warning(
                    f"Watchdog: stream '{config.name}' appears stuck, restarting"
                )
                await stream.stop()
                await asyncio.sleep(1)
                await stream.start()
                logger.info(f"Watchdog: stream '{config.name}' restarted successfully")
            except Exception as e:
                logger.error(f"Watchdog: failed to restart stream {camera_id}: {e}")

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


@app.on_event("startup")
async def startup():
    logger.info("Starting date rollover monitor")
    await manager.start_date_rollover_monitor()
    logger.info("Starting stream watchdog")
    await manager.start_watchdog()


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down stream manager")
    await manager.stop_watchdog()
    await manager.stop_date_rollover_monitor()
    await manager.stop_all()


if __name__ == "__main__":
    logger.info(f"Starting Stream Manager on port {STREAM_MANAGER_PORT}")
    logger.info(f"Storage root: {STORAGE_ROOT}")
    uvicorn.run(app, host="0.0.0.0", port=STREAM_MANAGER_PORT, log_level="info")
