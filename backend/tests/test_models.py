"""Tests for SQLAlchemy models."""

from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Camera, Recording
from app.models.camera import CameraStatus
from app.models.recording import RecordingStatus


@pytest.mark.asyncio
async def test_create_camera(db_session: AsyncSession) -> None:
    """Test creating a camera."""
    camera = Camera(
        name="Test Camera",
        host="192.168.1.100",
        port=554,
        path="/cam/realmonitor",
        username="admin",
        password="password123",
    )
    db_session.add(camera)
    await db_session.commit()
    await db_session.refresh(camera)

    assert camera.id is not None
    assert camera.name == "Test Camera"
    assert camera.host == "192.168.1.100"
    assert camera.port == 554
    assert camera.status == CameraStatus.UNKNOWN.value


@pytest.mark.asyncio
async def test_camera_rtsp_url(db_session: AsyncSession) -> None:
    """Test RTSP URL generation."""
    camera = Camera(
        name="URL Test Camera",
        host="192.168.1.100",
        port=554,
        path="/live/main",
        username="admin",
        password="secret",
    )
    db_session.add(camera)
    await db_session.commit()

    assert camera.rtsp_url == "rtsp://admin:secret@192.168.1.100:554/live/main"


@pytest.mark.asyncio
async def test_camera_rtsp_url_no_auth(db_session: AsyncSession) -> None:
    """Test RTSP URL without authentication."""
    camera = Camera(
        name="No Auth Camera",
        host="192.168.1.100",
        port=554,
        path="/stream",
    )
    db_session.add(camera)
    await db_session.commit()

    assert camera.rtsp_url == "rtsp://192.168.1.100:554/stream"


@pytest.mark.asyncio
async def test_camera_unique_name(db_session: AsyncSession) -> None:
    """Test that camera names must be unique."""
    camera1 = Camera(name="Unique Camera", host="192.168.1.100")
    db_session.add(camera1)
    await db_session.commit()

    camera2 = Camera(name="Unique Camera", host="192.168.1.101")
    db_session.add(camera2)

    with pytest.raises(Exception):
        await db_session.commit()


@pytest.mark.asyncio
async def test_create_recording(db_session: AsyncSession) -> None:
    """Test creating a recording."""
    camera = Camera(name="Recording Test Camera", host="192.168.1.100")
    db_session.add(camera)
    await db_session.commit()

    recording = Recording(
        camera_id=camera.id,
        file_path="/storage/test/2024-01-15/12-00-00.mp4",
        start_time=datetime(2024, 1, 15, 12, 0, 0),
        status=RecordingStatus.RECORDING.value,
    )
    db_session.add(recording)
    await db_session.commit()
    await db_session.refresh(recording)

    assert recording.id is not None
    assert recording.camera_id == camera.id
    assert recording.status == RecordingStatus.RECORDING.value


@pytest.mark.asyncio
async def test_camera_recording_relationship(db_session: AsyncSession) -> None:
    """Test camera-recording relationship."""
    camera = Camera(name="Relationship Test Camera", host="192.168.1.100")
    db_session.add(camera)
    await db_session.commit()

    recording1 = Recording(
        camera_id=camera.id,
        file_path="/storage/test/2024-01-15/12-00-00.mp4",
        start_time=datetime(2024, 1, 15, 12, 0, 0),
    )
    recording2 = Recording(
        camera_id=camera.id,
        file_path="/storage/test/2024-01-15/12-15-00.mp4",
        start_time=datetime(2024, 1, 15, 12, 15, 0),
    )
    db_session.add_all([recording1, recording2])
    await db_session.commit()

    # Refresh to load relationships
    result = await db_session.execute(
        select(Camera).where(Camera.id == camera.id)
    )
    loaded_camera = result.scalar_one()
    await db_session.refresh(loaded_camera, ["recordings"])

    assert len(loaded_camera.recordings) == 2


@pytest.mark.asyncio
async def test_cascade_delete_recordings(db_session: AsyncSession) -> None:
    """Test that deleting a camera cascades to recordings."""
    camera = Camera(name="Cascade Test Camera", host="192.168.1.100")
    db_session.add(camera)
    await db_session.commit()

    recording = Recording(
        camera_id=camera.id,
        file_path="/storage/cascade/test.mp4",
        start_time=datetime(2024, 1, 15, 12, 0, 0),
    )
    db_session.add(recording)
    await db_session.commit()
    recording_id = recording.id

    # Delete camera
    await db_session.delete(camera)
    await db_session.commit()

    # Recording should be deleted
    result = await db_session.execute(
        select(Recording).where(Recording.id == recording_id)
    )
    assert result.scalar_one_or_none() is None
