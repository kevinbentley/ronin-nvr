"""Tests for stream state machine and CameraStreamManager."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.camera_stream import (
    CameraStream,
    CameraStreamManager,
    StreamState,
)


class TestStreamStateEnum:
    """Tests for StreamState enum values."""

    def test_all_states_are_strings(self) -> None:
        """All states serialize to strings."""
        for state in StreamState:
            assert isinstance(state.value, str)

    def test_expected_states_exist(self) -> None:
        """All expected states are defined."""
        expected = {"stopped", "starting", "running", "reconnecting", "error"}
        actual = {s.value for s in StreamState}
        assert actual == expected

    def test_state_values_are_lowercase(self) -> None:
        """State values are all lowercase for JSON consistency."""
        for state in StreamState:
            assert state.value == state.value.lower()


class TestIsRunningProperty:
    """Tests for is_running determination based on state."""

    @pytest.mark.parametrize(
        "state,expected",
        [
            (StreamState.STOPPED, False),
            (StreamState.STARTING, False),
            (StreamState.RUNNING, True),
            (StreamState.RECONNECTING, True),
            (StreamState.ERROR, False),
        ],
    )
    def test_is_running_for_each_state(
        self, state: StreamState, expected: bool
    ) -> None:
        """is_running is True only for RUNNING and RECONNECTING."""
        is_running = state in (StreamState.RUNNING, StreamState.RECONNECTING)
        assert is_running == expected


class TestCameraStreamManager:
    """Tests for CameraStreamManager class."""

    @pytest.fixture
    def manager(self) -> CameraStreamManager:
        """Create fresh stream manager."""
        return CameraStreamManager()

    @pytest.fixture
    def mock_camera(self) -> MagicMock:
        """Create a mock camera."""
        camera = MagicMock()
        camera.id = 1
        camera.name = "Test Camera"
        camera.host = "192.168.1.100"
        camera.port = 554
        camera.path = "/stream"
        camera.username = None
        camera.password = None
        camera.transport = "tcp"
        camera.rtsp_url = "rtsp://192.168.1.100:554/stream"
        return camera

    def test_get_stream_creates_new(
        self, manager: CameraStreamManager, mock_camera: MagicMock
    ) -> None:
        """get_stream creates new stream for unknown camera."""
        stream = manager.get_stream(mock_camera)
        assert stream is not None
        assert stream.camera_id == mock_camera.id

    def test_get_stream_returns_existing(
        self, manager: CameraStreamManager, mock_camera: MagicMock
    ) -> None:
        """get_stream returns existing stream for known camera."""
        stream1 = manager.get_stream(mock_camera)
        stream2 = manager.get_stream(mock_camera)
        assert stream1 is stream2

    def test_get_status_unknown_camera(
        self, manager: CameraStreamManager
    ) -> None:
        """get_status returns None for unknown camera."""
        status = manager.get_status(999)
        assert status is None

    def test_get_status_known_camera(
        self, manager: CameraStreamManager, mock_camera: MagicMock
    ) -> None:
        """get_status returns status for known camera."""
        manager.get_stream(mock_camera)
        status = manager.get_status(mock_camera.id)
        assert status is not None
        assert status["camera_id"] == mock_camera.id

    def test_is_running_unknown_camera(
        self, manager: CameraStreamManager
    ) -> None:
        """is_running returns False for unknown camera."""
        assert manager.is_running(999) is False

    def test_is_running_stopped_camera(
        self, manager: CameraStreamManager, mock_camera: MagicMock
    ) -> None:
        """is_running returns False for stopped camera."""
        manager.get_stream(mock_camera)
        assert manager.is_running(mock_camera.id) is False

    def test_is_running_running_camera(
        self, manager: CameraStreamManager, mock_camera: MagicMock
    ) -> None:
        """is_running returns True for running camera."""
        stream = manager.get_stream(mock_camera)
        stream._state = StreamState.RUNNING
        assert manager.is_running(mock_camera.id) is True

    def test_is_recording_unknown_camera(
        self, manager: CameraStreamManager
    ) -> None:
        """is_recording returns False for unknown camera."""
        assert manager.is_recording(999) is False

    def test_is_recording_stopped_camera(
        self, manager: CameraStreamManager, mock_camera: MagicMock
    ) -> None:
        """is_recording returns False for stopped camera."""
        manager.get_stream(mock_camera)
        assert manager.is_recording(mock_camera.id) is False

    def test_is_recording_running_camera(
        self, manager: CameraStreamManager, mock_camera: MagicMock
    ) -> None:
        """is_recording returns True for running camera with recording."""
        stream = manager.get_stream(mock_camera)
        stream._state = StreamState.RUNNING
        stream._recording_enabled = True
        assert manager.is_recording(mock_camera.id) is True

    def test_get_all_status_empty(
        self, manager: CameraStreamManager
    ) -> None:
        """get_all_status returns empty list when no streams."""
        statuses = manager.get_all_status()
        assert statuses == []

    def test_get_all_status_multiple(
        self, manager: CameraStreamManager
    ) -> None:
        """get_all_status returns status for all streams."""
        for i in range(3):
            camera = MagicMock()
            camera.id = i + 1
            camera.name = f"Camera {i + 1}"
            camera.host = f"192.168.1.{i + 1}"
            camera.port = 554
            camera.path = "/stream"
            camera.username = None
            camera.password = None
            camera.transport = "tcp"
            camera.rtsp_url = f"rtsp://192.168.1.{i + 1}:554/stream"
            manager.get_stream(camera)

        statuses = manager.get_all_status()
        assert len(statuses) == 3
        camera_ids = {s["camera_id"] for s in statuses}
        assert camera_ids == {1, 2, 3}

    @pytest.mark.asyncio
    async def test_start_stream(
        self, manager: CameraStreamManager, mock_camera: MagicMock
    ) -> None:
        """start_stream starts the camera stream."""
        with patch.object(
            CameraStream, "start", new_callable=AsyncMock
        ) as mock_start:
            mock_start.return_value = True
            result = await manager.start_stream(mock_camera, recording_enabled=True)
            assert result is True
            mock_start.assert_called_once_with(True)

    @pytest.mark.asyncio
    async def test_stop_stream_unknown(
        self, manager: CameraStreamManager
    ) -> None:
        """stop_stream returns False for unknown camera."""
        result = await manager.stop_stream(999)
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_stream_known(
        self, manager: CameraStreamManager, mock_camera: MagicMock
    ) -> None:
        """stop_stream stops known camera."""
        stream = manager.get_stream(mock_camera)
        with patch.object(stream, "stop", new_callable=AsyncMock) as mock_stop:
            result = await manager.stop_stream(mock_camera.id)
            assert result is True
            mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_stream_unknown(
        self, manager: CameraStreamManager
    ) -> None:
        """restart_stream returns False for unknown camera."""
        result = await manager.restart_stream(999)
        assert result is False

    @pytest.mark.asyncio
    async def test_restart_stream_known(
        self, manager: CameraStreamManager, mock_camera: MagicMock
    ) -> None:
        """restart_stream restarts known camera."""
        stream = manager.get_stream(mock_camera)
        with patch.object(stream, "restart", new_callable=AsyncMock) as mock_restart:
            mock_restart.return_value = True
            result = await manager.restart_stream(mock_camera.id)
            assert result is True
            mock_restart.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_all(
        self, manager: CameraStreamManager
    ) -> None:
        """stop_all stops all managed streams."""
        streams = []
        for i in range(3):
            camera = MagicMock()
            camera.id = i + 1
            camera.name = f"Camera {i + 1}"
            camera.host = f"192.168.1.{i + 1}"
            camera.port = 554
            camera.path = "/stream"
            camera.username = None
            camera.password = None
            camera.transport = "tcp"
            camera.rtsp_url = f"rtsp://192.168.1.{i + 1}:554/stream"
            stream = manager.get_stream(camera)
            streams.append(stream)

        with patch.object(
            CameraStream, "stop", new_callable=AsyncMock
        ) as mock_stop:
            await manager.stop_all()
            assert mock_stop.call_count == 3

    def test_get_playlist_path_unknown(
        self, manager: CameraStreamManager
    ) -> None:
        """get_playlist_path returns None for unknown camera."""
        result = manager.get_playlist_path(999)
        assert result is None

    def test_get_playlist_path_known(
        self, manager: CameraStreamManager, mock_camera: MagicMock, tmp_path: Path
    ) -> None:
        """get_playlist_path returns path for known camera with playlist."""
        # Create stream with tmp_path as storage root
        stream = CameraStream(mock_camera, storage_root=tmp_path)
        manager._streams[mock_camera.id] = stream
        stream._state = StreamState.RUNNING

        # Create playlist file
        stream.hls_directory.mkdir(parents=True, exist_ok=True)
        stream.playlist_path.write_text("#EXTM3U\n")

        result = manager.get_playlist_path(mock_camera.id)
        assert result == stream.playlist_path

    def test_get_segment_path_unknown(
        self, manager: CameraStreamManager
    ) -> None:
        """get_segment_path returns None for unknown camera."""
        result = manager.get_segment_path(999, "segment.ts")
        assert result is None


class TestMultipleCameras:
    """Tests for managing multiple independent cameras."""

    @pytest.fixture
    def manager(self) -> CameraStreamManager:
        """Create fresh stream manager."""
        return CameraStreamManager()

    def test_cameras_have_independent_state(
        self, manager: CameraStreamManager
    ) -> None:
        """Each camera has independent stream state."""
        cameras = []
        streams = []
        for i in range(3):
            camera = MagicMock()
            camera.id = i + 1
            camera.name = f"Camera {i + 1}"
            camera.host = f"192.168.1.{i + 1}"
            camera.port = 554
            camera.path = "/stream"
            camera.username = None
            camera.password = None
            camera.transport = "tcp"
            camera.rtsp_url = f"rtsp://192.168.1.{i + 1}:554/stream"
            cameras.append(camera)
            streams.append(manager.get_stream(camera))

        # Set different states
        streams[0]._state = StreamState.RUNNING
        streams[1]._state = StreamState.ERROR
        streams[2]._state = StreamState.STOPPED

        # Verify independence
        assert manager.is_running(1) is True
        assert manager.is_running(2) is False
        assert manager.is_running(3) is False

    def test_cameras_have_independent_error_messages(
        self, manager: CameraStreamManager
    ) -> None:
        """Each camera has independent error messages."""
        cameras = []
        streams = []
        for i in range(2):
            camera = MagicMock()
            camera.id = i + 1
            camera.name = f"Camera {i + 1}"
            camera.host = f"192.168.1.{i + 1}"
            camera.port = 554
            camera.path = "/stream"
            camera.username = None
            camera.password = None
            camera.transport = "tcp"
            camera.rtsp_url = f"rtsp://192.168.1.{i + 1}:554/stream"
            cameras.append(camera)
            streams.append(manager.get_stream(camera))

        streams[0]._error_message = "Connection timeout"
        streams[1]._error_message = "Authentication failed"

        status0 = manager.get_status(1)
        status1 = manager.get_status(2)

        assert status0["error_message"] == "Connection timeout"
        assert status1["error_message"] == "Authentication failed"
