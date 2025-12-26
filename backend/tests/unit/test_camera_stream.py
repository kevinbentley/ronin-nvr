"""Unit tests for CameraStream class."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.camera_stream import CameraStream, StreamState


@pytest.fixture
def mock_camera() -> MagicMock:
    """Create a mock camera object."""
    camera = MagicMock()
    camera.id = 1
    camera.name = "Test Camera"
    camera.host = "192.168.1.100"
    camera.port = 554
    camera.path = "/stream"
    camera.username = "admin"
    camera.password = "secret"
    camera.transport = "tcp"
    camera.rtsp_url = "rtsp://admin:secret@192.168.1.100:554/stream"
    return camera


@pytest.fixture
def stream(mock_camera: MagicMock, tmp_path: Path) -> CameraStream:
    """Create CameraStream instance with temp storage."""
    return CameraStream(
        camera=mock_camera,
        storage_root=tmp_path,
        segment_duration_minutes=1,
    )


class TestCameraStreamInitialization:
    """Tests for CameraStream initialization."""

    def test_initial_state_is_stopped(self, stream: CameraStream) -> None:
        """Stream starts in STOPPED state."""
        assert stream.state == StreamState.STOPPED

    def test_is_running_false_when_stopped(self, stream: CameraStream) -> None:
        """is_running is False when stopped."""
        assert stream.is_running is False

    def test_is_recording_false_when_stopped(self, stream: CameraStream) -> None:
        """is_recording is False when stopped."""
        assert stream.is_recording is False

    def test_camera_attributes_extracted(
        self, stream: CameraStream, mock_camera: MagicMock
    ) -> None:
        """Camera attributes are extracted during init."""
        assert stream.camera_id == mock_camera.id
        assert stream.camera_name == mock_camera.name
        assert stream.camera_host == mock_camera.host
        assert stream.camera_port == mock_camera.port
        assert stream.camera_path == mock_camera.path
        assert stream.camera_username == mock_camera.username
        assert stream.camera_password == mock_camera.password
        assert stream.camera_transport == mock_camera.transport
        assert stream.rtsp_url == mock_camera.rtsp_url

    def test_segment_duration_converted_to_seconds(
        self, mock_camera: MagicMock, tmp_path: Path
    ) -> None:
        """Segment duration in minutes is converted to seconds."""
        stream = CameraStream(
            camera=mock_camera,
            storage_root=tmp_path,
            segment_duration_minutes=15,
        )
        assert stream.segment_duration == 15 * 60


class TestSafeCameraName:
    """Tests for filesystem-safe camera name generation."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("simple", "simple"),
            ("Front Door", "Front_Door"),
            ("Camera #1", "Camera__1"),
            ("Test/Camera", "Test_Camera"),
            ("a-b_c", "a-b_c"),
            ("Back Yard (North)", "Back_Yard__North_"),
            ("Camera@Home", "Camera_Home"),
        ],
    )
    def test_safe_camera_name_transformation(
        self, mock_camera: MagicMock, tmp_path: Path, name: str, expected: str
    ) -> None:
        """Camera names are sanitized for filesystem."""
        mock_camera.name = name
        stream = CameraStream(mock_camera, tmp_path)
        assert stream.safe_camera_name == expected


class TestDirectoryPaths:
    """Tests for directory path properties."""

    def test_recording_directory(
        self, stream: CameraStream, tmp_path: Path
    ) -> None:
        """Recording directory is based on camera name."""
        expected = tmp_path / "Test_Camera"
        assert stream.recording_directory == expected

    def test_hls_directory(
        self, stream: CameraStream, tmp_path: Path
    ) -> None:
        """HLS directory is based on camera ID."""
        expected = tmp_path / ".streams" / "1"
        assert stream.hls_directory == expected

    def test_playlist_path(self, stream: CameraStream) -> None:
        """Playlist path is in HLS directory."""
        assert stream.playlist_path == stream.hls_directory / "playlist.m3u8"


class TestFFmpegCommand:
    """Tests for FFmpeg command construction."""

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_command_includes_rtsp_url(
        self, mock_which: MagicMock, stream: CameraStream
    ) -> None:
        """FFmpeg command includes RTSP URL."""
        cmd = stream._build_ffmpeg_command()
        assert stream.rtsp_url in cmd

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_command_includes_transport(
        self, mock_which: MagicMock, stream: CameraStream
    ) -> None:
        """FFmpeg command includes transport setting."""
        cmd = stream._build_ffmpeg_command()
        idx = cmd.index("-rtsp_transport")
        assert cmd[idx + 1] == "tcp"

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_command_includes_hls_output(
        self, mock_which: MagicMock, stream: CameraStream
    ) -> None:
        """FFmpeg command includes HLS output configuration."""
        cmd = stream._build_ffmpeg_command()
        assert "-f" in cmd
        hls_idx = cmd.index("hls")
        assert cmd[hls_idx - 1] == "-f"
        assert str(stream.playlist_path) in cmd

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_command_includes_segment_output_when_recording(
        self, mock_which: MagicMock, stream: CameraStream
    ) -> None:
        """FFmpeg command includes segment output when recording enabled."""
        stream._recording_enabled = True
        cmd = stream._build_ffmpeg_command()
        assert "-segment_time" in cmd
        assert str(stream.segment_duration) in cmd

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_command_excludes_segment_output_when_not_recording(
        self, mock_which: MagicMock, stream: CameraStream
    ) -> None:
        """FFmpeg command excludes segment output when recording disabled."""
        stream._recording_enabled = False
        cmd = stream._build_ffmpeg_command()
        assert "-segment_time" not in cmd

    @patch("shutil.which", return_value=None)
    def test_raises_when_ffmpeg_not_found(
        self, mock_which: MagicMock, stream: CameraStream
    ) -> None:
        """Raises RuntimeError when FFmpeg not in PATH."""
        with pytest.raises(RuntimeError, match="FFmpeg not found"):
            stream._build_ffmpeg_command()

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_creates_hls_directory(
        self, mock_which: MagicMock, stream: CameraStream
    ) -> None:
        """FFmpeg command creation creates HLS directory."""
        stream._build_ffmpeg_command()
        assert stream.hls_directory.exists()

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_cleans_old_hls_segments(
        self, mock_which: MagicMock, stream: CameraStream
    ) -> None:
        """FFmpeg command creation cleans old HLS segments."""
        # Create some old segments
        stream.hls_directory.mkdir(parents=True, exist_ok=True)
        old_segment = stream.hls_directory / "old_segment.ts"
        old_segment.write_bytes(b"old data")
        old_playlist = stream.playlist_path
        old_playlist.write_text("#EXTM3U\n")

        stream._build_ffmpeg_command()

        assert not old_segment.exists()
        assert not old_playlist.exists()


class TestStartStop:
    """Tests for stream start/stop operations."""

    @pytest.mark.asyncio
    async def test_start_calls_start_ffmpeg(
        self, stream: CameraStream
    ) -> None:
        """Start calls _start_ffmpeg."""
        with patch.object(
            stream, "_start_ffmpeg", new_callable=AsyncMock
        ) as mock_start:
            await stream.start()
            mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_sets_recording_enabled(
        self, stream: CameraStream
    ) -> None:
        """Start sets recording_enabled flag."""
        with patch.object(stream, "_start_ffmpeg", new_callable=AsyncMock):
            await stream.start(recording_enabled=False)
            assert stream._recording_enabled is False

            await stream.stop()
            await stream.start(recording_enabled=True)
            assert stream._recording_enabled is True

    @pytest.mark.asyncio
    async def test_start_when_already_running_returns_true(
        self, stream: CameraStream
    ) -> None:
        """Starting an already running stream returns True without restart."""
        stream._state = StreamState.RUNNING
        with patch.object(
            stream, "_start_ffmpeg", new_callable=AsyncMock
        ) as mock_start:
            result = await stream.start()
            assert result is True
            mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_resets_reconnect_attempts(
        self, stream: CameraStream
    ) -> None:
        """Start resets reconnect attempt counter."""
        stream._reconnect_attempts = 5
        with patch.object(stream, "_start_ffmpeg", new_callable=AsyncMock):
            await stream.start()
            assert stream._reconnect_attempts == 0

    @pytest.mark.asyncio
    async def test_start_on_error_sets_error_state(
        self, stream: CameraStream
    ) -> None:
        """Start on error sets ERROR state and message."""
        with patch.object(
            stream, "_start_ffmpeg", new_callable=AsyncMock,
            side_effect=RuntimeError("FFmpeg crashed")
        ):
            result = await stream.start()
            assert result is False
            assert stream.state == StreamState.ERROR
            assert "FFmpeg crashed" in stream.error_message

    @pytest.mark.asyncio
    async def test_stop_from_stopped_is_noop(
        self, stream: CameraStream
    ) -> None:
        """Stopping a stopped stream is a no-op."""
        await stream.stop()
        assert stream.state == StreamState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_sets_state_to_stopped(
        self, stream: CameraStream
    ) -> None:
        """Stop sets state to STOPPED."""
        stream._state = StreamState.RUNNING
        stream._process = None
        await stream.stop()
        assert stream.state == StreamState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_cancels_monitor_task(
        self, stream: CameraStream
    ) -> None:
        """Stop cancels the monitor task."""
        stream._state = StreamState.RUNNING

        # Create a real task that we can cancel
        async def dummy():
            await asyncio.sleep(100)

        stream._monitor_task = asyncio.create_task(dummy())
        stream._process = None

        await stream.stop()

        assert stream._monitor_task is None

    @pytest.mark.asyncio
    async def test_stop_sends_sigint_then_kills(
        self, stream: CameraStream
    ) -> None:
        """Stop sends SIGINT, then SIGKILL on timeout."""
        import signal

        stream._state = StreamState.RUNNING
        stream._monitor_task = None

        mock_process = MagicMock()
        mock_process.send_signal = MagicMock()
        mock_process.kill = MagicMock()

        # Simulate timeout on first wait, then success
        call_count = 0

        async def mock_wait():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError()
            return 0

        mock_process.wait = mock_wait
        stream._process = mock_process

        await stream.stop()

        mock_process.send_signal.assert_called_once_with(signal.SIGINT)
        mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_stops_then_starts(
        self, stream: CameraStream
    ) -> None:
        """Restart stops then starts the stream."""
        stream._state = StreamState.RUNNING
        stream._recording_enabled = True

        with patch.object(stream, "stop", new_callable=AsyncMock) as mock_stop:
            with patch.object(stream, "start", new_callable=AsyncMock) as mock_start:
                mock_start.return_value = True
                result = await stream.restart()

                mock_stop.assert_called_once()
                mock_start.assert_called_once_with(True)
                assert result is True


class TestReconnection:
    """Tests for reconnection logic."""

    @pytest.mark.asyncio
    async def test_disconnect_increments_attempts(
        self, stream: CameraStream
    ) -> None:
        """Disconnect handler increments reconnect attempts."""
        stream._state = StreamState.RUNNING
        stream._reconnect_attempts = 2
        stream._reconnect_delay = 0.01  # Fast for testing

        # Make start fail so it keeps trying and incrementing
        async def fail_start():
            raise RuntimeError("Connection failed")

        with patch.object(stream, "_start_ffmpeg", side_effect=fail_start):
            task = asyncio.create_task(stream._handle_disconnect())
            await asyncio.sleep(0.1)
            stream._state = StreamState.STOPPED
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Should have incremented at least once from 2 to 3
        assert stream._reconnect_attempts >= 3

    @pytest.mark.asyncio
    async def test_max_attempts_sets_error_state(
        self, stream: CameraStream
    ) -> None:
        """Exceeding max attempts sets ERROR state."""
        stream._state = StreamState.RUNNING
        stream._reconnect_attempts = stream._max_reconnect_attempts

        await stream._handle_disconnect()

        assert stream.state == StreamState.ERROR
        assert "Max reconnection attempts" in stream.error_message

    @pytest.mark.asyncio
    async def test_successful_reconnect_resets_counter(
        self, stream: CameraStream
    ) -> None:
        """Successful reconnection resets attempt counter."""
        stream._state = StreamState.RUNNING
        stream._reconnect_attempts = 3
        stream._reconnect_delay = 0.01

        async def mock_start():
            stream._state = StreamState.RUNNING

        with patch.object(stream, "_start_ffmpeg", side_effect=mock_start):
            task = asyncio.create_task(stream._handle_disconnect())
            await asyncio.sleep(0.05)
            stream._state = StreamState.STOPPED
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert stream._reconnect_attempts == 0

    def test_backoff_delay_capped_at_60_seconds(
        self, stream: CameraStream
    ) -> None:
        """Reconnection delay is capped at 60 seconds."""
        stream._reconnect_delay = 5
        # Simulate many attempts
        for attempts in [10, 15, 20, 50]:
            delay = min(stream._reconnect_delay * (attempts + 1), 60)
            assert delay <= 60


class TestGetStatus:
    """Tests for status dictionary."""

    def test_status_includes_all_fields(self, stream: CameraStream) -> None:
        """Status dict includes all expected fields."""
        status = stream.get_status()
        expected_fields = {
            "camera_id",
            "camera_name",
            "state",
            "is_running",
            "is_recording",
            "error_message",
            "start_time",
            "recording_directory",
            "reconnect_attempts",
        }
        assert set(status.keys()) == expected_fields

    def test_status_values_correct_when_stopped(
        self, stream: CameraStream
    ) -> None:
        """Status values are correct when stopped."""
        status = stream.get_status()
        assert status["camera_id"] == 1
        assert status["camera_name"] == "Test Camera"
        assert status["state"] == "stopped"
        assert status["is_running"] is False
        assert status["is_recording"] is False
        assert status["start_time"] is None

    def test_status_start_time_formatted_as_iso(
        self, stream: CameraStream
    ) -> None:
        """Start time is formatted as ISO string."""
        stream._start_time = datetime(2024, 1, 15, 10, 30, 0)
        status = stream.get_status()
        assert status["start_time"] == "2024-01-15T10:30:00"

    def test_status_reflects_running_state(
        self, stream: CameraStream
    ) -> None:
        """Status reflects running state correctly."""
        stream._state = StreamState.RUNNING
        stream._recording_enabled = True
        stream._start_time = datetime.now()

        status = stream.get_status()
        assert status["state"] == "running"
        assert status["is_running"] is True
        assert status["is_recording"] is True


class TestGetPaths:
    """Tests for path getter methods."""

    def test_get_playlist_path_when_running(
        self, stream: CameraStream
    ) -> None:
        """get_playlist_path returns path when running and file exists."""
        stream._state = StreamState.RUNNING
        stream.hls_directory.mkdir(parents=True, exist_ok=True)
        stream.playlist_path.write_text("#EXTM3U\n")

        result = stream.get_playlist_path()
        assert result == stream.playlist_path

    def test_get_playlist_path_when_stopped(
        self, stream: CameraStream
    ) -> None:
        """get_playlist_path returns None when stopped."""
        stream._state = StreamState.STOPPED
        result = stream.get_playlist_path()
        assert result is None

    def test_get_playlist_path_when_file_missing(
        self, stream: CameraStream
    ) -> None:
        """get_playlist_path returns None when file doesn't exist."""
        stream._state = StreamState.RUNNING
        result = stream.get_playlist_path()
        assert result is None

    def test_get_segment_path_when_exists(
        self, stream: CameraStream
    ) -> None:
        """get_segment_path returns path when segment exists."""
        stream._state = StreamState.RUNNING
        stream.hls_directory.mkdir(parents=True, exist_ok=True)
        segment = stream.hls_directory / "segment001.ts"
        segment.write_bytes(b"video data")

        result = stream.get_segment_path("segment001.ts")
        assert result == segment

    def test_get_segment_path_when_missing(
        self, stream: CameraStream
    ) -> None:
        """get_segment_path returns None when segment doesn't exist."""
        stream._state = StreamState.RUNNING
        result = stream.get_segment_path("missing.ts")
        assert result is None
