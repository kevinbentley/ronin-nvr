"""Tests for streams health endpoint."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient

from app.services.camera_stream import CameraStream, StreamState, stream_manager


@pytest.fixture(autouse=True)
def clear_streams():
    """Clear stream manager before and after each test."""
    stream_manager._streams.clear()
    yield
    stream_manager._streams.clear()


class TestStreamsHealth:
    """Tests for /streams/health endpoint."""

    @pytest.mark.asyncio
    async def test_health_empty_returns_zeros(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Health endpoint returns zeros when no streams exist."""
        response = await client.get(
            "/api/cameras/streams/health", headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_streams"] == 0
        assert data["healthy"] == 0
        assert data["reconnecting"] == 0
        assert data["errored"] == 0
        assert data["stopped"] == 0
        assert data["streams"] == []

    @pytest.mark.asyncio
    async def test_health_counts_running_as_healthy(
        self, client: AsyncClient, tmp_path: Path, auth_headers: dict[str, str]
    ) -> None:
        """Running streams are counted as healthy."""
        # Create mock camera and stream
        mock_cam = MagicMock()
        mock_cam.id = 1
        mock_cam.name = "Test Camera"
        mock_cam.host = "192.168.1.100"
        mock_cam.port = 554
        mock_cam.path = "/stream"
        mock_cam.username = None
        mock_cam.password = None
        mock_cam.transport = "tcp"
        mock_cam.rtsp_url = "rtsp://192.168.1.100:554/stream"

        stream = CameraStream(mock_cam, storage_root=tmp_path)
        stream._state = StreamState.RUNNING
        stream._recording_enabled = True
        stream_manager._streams[1] = stream

        response = await client.get(
            "/api/cameras/streams/health", headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_streams"] == 1
        assert data["healthy"] == 1
        assert data["reconnecting"] == 0
        assert data["errored"] == 0

    @pytest.mark.asyncio
    async def test_health_counts_reconnecting(
        self, client: AsyncClient, tmp_path: Path, auth_headers: dict[str, str]
    ) -> None:
        """Reconnecting streams are counted separately."""
        mock_cam = MagicMock()
        mock_cam.id = 1
        mock_cam.name = "Reconnecting Camera"
        mock_cam.host = "192.168.1.100"
        mock_cam.port = 554
        mock_cam.path = "/stream"
        mock_cam.username = None
        mock_cam.password = None
        mock_cam.transport = "tcp"
        mock_cam.rtsp_url = "rtsp://192.168.1.100:554/stream"

        stream = CameraStream(mock_cam, storage_root=tmp_path)
        stream._state = StreamState.RECONNECTING
        stream._reconnect_attempts = 3
        stream_manager._streams[1] = stream

        response = await client.get(
            "/api/cameras/streams/health", headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_streams"] == 1
        assert data["healthy"] == 0
        assert data["reconnecting"] == 1
        assert data["streams"][0]["reconnect_attempts"] == 3

    @pytest.mark.asyncio
    async def test_health_counts_errored(
        self, client: AsyncClient, tmp_path: Path, auth_headers: dict[str, str]
    ) -> None:
        """Errored streams are counted separately."""
        mock_cam = MagicMock()
        mock_cam.id = 1
        mock_cam.name = "Error Camera"
        mock_cam.host = "192.168.1.100"
        mock_cam.port = 554
        mock_cam.path = "/stream"
        mock_cam.username = None
        mock_cam.password = None
        mock_cam.transport = "tcp"
        mock_cam.rtsp_url = "rtsp://192.168.1.100:554/stream"

        stream = CameraStream(mock_cam, storage_root=tmp_path)
        stream._state = StreamState.ERROR
        stream._error_message = "Connection refused"
        stream_manager._streams[1] = stream

        response = await client.get(
            "/api/cameras/streams/health", headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_streams"] == 1
        assert data["healthy"] == 0
        assert data["errored"] == 1
        assert data["streams"][0]["error_message"] == "Connection refused"

    @pytest.mark.asyncio
    async def test_health_multiple_streams_mixed_states(
        self, client: AsyncClient, tmp_path: Path, auth_headers: dict[str, str]
    ) -> None:
        """Health correctly counts multiple streams in different states."""
        # Create streams in different states
        states = [
            (1, "Running Camera", StreamState.RUNNING),
            (2, "Also Running", StreamState.RUNNING),
            (3, "Reconnecting", StreamState.RECONNECTING),
            (4, "Error Camera", StreamState.ERROR),
            (5, "Stopped Camera", StreamState.STOPPED),
        ]

        for cam_id, name, state in states:
            mock_cam = MagicMock()
            mock_cam.id = cam_id
            mock_cam.name = name
            mock_cam.host = f"192.168.1.{cam_id}"
            mock_cam.port = 554
            mock_cam.path = "/stream"
            mock_cam.username = None
            mock_cam.password = None
            mock_cam.transport = "tcp"
            mock_cam.rtsp_url = f"rtsp://192.168.1.{cam_id}:554/stream"

            stream = CameraStream(mock_cam, storage_root=tmp_path)
            stream._state = state
            stream_manager._streams[cam_id] = stream

        response = await client.get(
            "/api/cameras/streams/health", headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_streams"] == 5
        assert data["healthy"] == 2
        assert data["reconnecting"] == 1
        assert data["errored"] == 1
        assert data["stopped"] == 1
        assert len(data["streams"]) == 5

    @pytest.mark.asyncio
    async def test_health_stream_details_complete(
        self, client: AsyncClient, tmp_path: Path, auth_headers: dict[str, str]
    ) -> None:
        """Health endpoint returns complete stream details."""
        from datetime import datetime

        mock_cam = MagicMock()
        mock_cam.id = 1
        mock_cam.name = "Detailed Camera"
        mock_cam.host = "192.168.1.100"
        mock_cam.port = 554
        mock_cam.path = "/stream"
        mock_cam.username = None
        mock_cam.password = None
        mock_cam.transport = "tcp"
        mock_cam.rtsp_url = "rtsp://192.168.1.100:554/stream"

        stream = CameraStream(mock_cam, storage_root=tmp_path)
        stream._state = StreamState.RUNNING
        stream._recording_enabled = True
        stream._start_time = datetime(2024, 1, 15, 10, 30, 0)
        stream._reconnect_attempts = 0
        stream_manager._streams[1] = stream

        response = await client.get(
            "/api/cameras/streams/health", headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        stream_info = data["streams"][0]

        assert stream_info["camera_id"] == 1
        assert stream_info["camera_name"] == "Detailed Camera"
        assert stream_info["state"] == "running"
        assert stream_info["is_running"] is True
        assert stream_info["is_recording"] is True
        assert stream_info["error_message"] is None
        assert stream_info["reconnect_attempts"] == 0
        assert stream_info["start_time"] == "2024-01-15T10:30:00"

    @pytest.mark.asyncio
    async def test_health_unauthorized(self, client: AsyncClient) -> None:
        """Health endpoint requires authentication."""
        response = await client.get("/api/cameras/streams/health")
        assert response.status_code == 401
