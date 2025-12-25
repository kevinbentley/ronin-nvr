"""Tests for recording API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.fixture
async def test_camera(client: AsyncClient) -> dict:
    """Create a test camera for recording tests."""
    camera_data = {
        "name": "Recording Test Camera",
        "host": "192.168.1.100",
        "port": 554,
        "path": "/cam/realmonitor",
    }
    response = await client.post("/api/cameras", json=camera_data)
    return response.json()


@pytest.mark.asyncio
async def test_get_recording_status_no_recorder(
    client: AsyncClient, test_camera: dict
) -> None:
    """Test getting recording status when no recorder exists."""
    camera_id = test_camera["id"]
    response = await client.get(f"/api/cameras/{camera_id}/recording/status")
    assert response.status_code == 200

    data = response.json()
    assert data["camera_id"] == camera_id
    assert data["state"] == "stopped"
    assert data["error_message"] is None


@pytest.mark.asyncio
async def test_get_recording_status_not_found(client: AsyncClient) -> None:
    """Test getting recording status for non-existent camera."""
    response = await client.get("/api/cameras/99999/recording/status")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_start_recording_not_found(client: AsyncClient) -> None:
    """Test starting recording for non-existent camera."""
    response = await client.post("/api/cameras/99999/recording/start")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_stop_recording_not_found(client: AsyncClient) -> None:
    """Test stopping recording for non-existent camera."""
    response = await client.post("/api/cameras/99999/recording/stop")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_stop_recording_not_started(
    client: AsyncClient, test_camera: dict
) -> None:
    """Test stopping recording when not started."""
    camera_id = test_camera["id"]
    response = await client.post(f"/api/cameras/{camera_id}/recording/stop")
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["message"] == "Recording not in progress"


@pytest.mark.asyncio
async def test_get_all_recording_status_empty(client: AsyncClient) -> None:
    """Test getting all recording statuses when none exist."""
    response = await client.get("/api/cameras/recording/status")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_start_recording_no_ffmpeg(
    client: AsyncClient, test_camera: dict, monkeypatch
) -> None:
    """Test starting recording when FFmpeg is not available."""
    import shutil

    # Mock shutil.which to return None (FFmpeg not found)
    monkeypatch.setattr(shutil, "which", lambda x: None)

    camera_id = test_camera["id"]
    response = await client.post(f"/api/cameras/{camera_id}/recording/start")
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is False
    assert "FFmpeg not found" in data["message"] or "not found" in data["message"].lower()
