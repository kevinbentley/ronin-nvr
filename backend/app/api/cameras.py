"""Camera management API endpoints."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.camera import CameraStatus
from app.schemas import (
    CameraCreate,
    CameraListResponse,
    CameraResponse,
    CameraTestResult,
    CameraUpdate,
)
from app.services.camera import CameraService, test_camera_connection
from app.services.camera_stream import stream_manager

router = APIRouter(prefix="/cameras", tags=["cameras"])

# Per-camera locks to prevent concurrent stream start operations
_stream_start_locks: dict[int, asyncio.Lock] = {}


def _get_stream_lock(camera_id: int) -> asyncio.Lock:
    """Get or create a lock for a camera's stream operations."""
    if camera_id not in _stream_start_locks:
        _stream_start_locks[camera_id] = asyncio.Lock()
    return _stream_start_locks[camera_id]


@router.get("", response_model=CameraListResponse)
async def list_cameras(db: AsyncSession = Depends(get_db)) -> CameraListResponse:
    """Get all cameras."""
    service = CameraService(db)
    cameras = await service.get_all()
    return CameraListResponse(
        cameras=[CameraResponse.model_validate(c) for c in cameras],
        total=len(cameras),
    )


@router.get("/recording/status")
async def get_all_recording_status() -> list[dict]:
    """Get recording status for all cameras."""
    statuses = stream_manager.get_all_status()
    return [
        {
            "camera_id": s["camera_id"],
            "camera_name": s["camera_name"],
            "is_recording": s["is_recording"],
            "state": s["state"],
            "start_time": s.get("start_time"),
        }
        for s in statuses
    ]


@router.get("/streams/health")
async def get_streams_health() -> dict:
    """Get health status of all camera streams.

    Provides a summary of stream states for monitoring and debugging.
    """
    statuses = stream_manager.get_all_status()

    healthy = sum(1 for s in statuses if s["state"] == "running")
    reconnecting = sum(1 for s in statuses if s["state"] == "reconnecting")
    errored = sum(1 for s in statuses if s["state"] == "error")
    stopped = sum(1 for s in statuses if s["state"] == "stopped")

    return {
        "total_streams": len(statuses),
        "healthy": healthy,
        "reconnecting": reconnecting,
        "errored": errored,
        "stopped": stopped,
        "streams": [
            {
                "camera_id": s["camera_id"],
                "camera_name": s["camera_name"],
                "state": s["state"],
                "is_running": s["is_running"],
                "is_recording": s["is_recording"],
                "error_message": s.get("error_message"),
                "reconnect_attempts": s.get("reconnect_attempts", 0),
                "start_time": s.get("start_time"),
            }
            for s in statuses
        ],
    }


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
) -> CameraResponse:
    """Get a specific camera by ID."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )
    return CameraResponse.model_validate(camera)


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    camera_data: CameraCreate,
    db: AsyncSession = Depends(get_db),
) -> CameraResponse:
    """Create a new camera."""
    service = CameraService(db)

    # Check for duplicate name
    existing = await service.get_by_name(camera_data.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Camera with name '{camera_data.name}' already exists",
        )

    camera = await service.create(camera_data)
    return CameraResponse.model_validate(camera)


@router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: int,
    camera_data: CameraUpdate,
    db: AsyncSession = Depends(get_db),
) -> CameraResponse:
    """Update an existing camera."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    # Check for duplicate name if name is being changed
    if camera_data.name and camera_data.name != camera.name:
        existing = await service.get_by_name(camera_data.name)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Camera with name '{camera_data.name}' already exists",
            )

    camera = await service.update(camera, camera_data)
    return CameraResponse.model_validate(camera)


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a camera."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    # Stop stream if running
    await stream_manager.stop_stream(camera_id)

    await service.delete(camera)


@router.post("/{camera_id}/test", response_model=CameraTestResult)
async def test_camera(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
) -> CameraTestResult:
    """Test camera RTSP connection."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    result = await test_camera_connection(camera)

    # Update camera status based on test result
    if result.success:
        await service.update_status(camera, CameraStatus.ONLINE)
    else:
        await service.update_status(camera, CameraStatus.ERROR, result.message)

    return result


# Streaming and Recording endpoints (unified)

@router.post("/{camera_id}/stream/start")
async def start_stream(
    camera_id: int,
    recording: bool = True,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Start streaming (and optionally recording) for a camera."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    success = await stream_manager.start_stream(camera, recording_enabled=recording)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start stream",
        )

    return {
        "camera_id": camera_id,
        "streaming": True,
        "recording": recording,
        "playlist_url": f"/api/cameras/{camera_id}/stream/hls/playlist.m3u8",
    }


@router.post("/{camera_id}/stream/stop")
async def stop_stream(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Stop streaming for a camera."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    await stream_manager.stop_stream(camera_id)
    return {"camera_id": camera_id, "streaming": False, "recording": False}


@router.post("/{camera_id}/stream/restart")
async def restart_stream(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Restart streaming for a camera."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    success = await stream_manager.restart_stream(camera_id)
    if not success:
        # Camera wasn't streaming, start it fresh
        success = await stream_manager.start_stream(camera, recording_enabled=True)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to restart stream",
        )

    return {
        "camera_id": camera_id,
        "streaming": True,
        "playlist_url": f"/api/cameras/{camera_id}/stream/hls/playlist.m3u8",
    }


@router.get("/{camera_id}/stream/status")
async def get_stream_status(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get streaming status for a camera."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    status_info = stream_manager.get_status(camera_id)
    if not status_info:
        return {
            "camera_id": camera_id,
            "streaming": False,
            "recording": False,
            "state": "stopped",
        }

    return status_info


@router.get("/{camera_id}/stream/hls/playlist.m3u8")
async def get_hls_playlist(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """Get HLS playlist for a camera stream."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    # Use per-camera lock to prevent race conditions on concurrent requests
    async with _get_stream_lock(camera_id):
        # Auto-start stream if not running
        if not stream_manager.is_running(camera_id):
            success = await stream_manager.start_stream(camera, recording_enabled=True)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Failed to start stream - check camera connection",
                )
            # Wait briefly for FFmpeg to generate playlist
            for _ in range(20):  # Wait up to 4 seconds
                await asyncio.sleep(0.2)
                playlist_path = stream_manager.get_playlist_path(camera_id)
                if playlist_path and Path(playlist_path).exists():
                    break

    playlist_path = stream_manager.get_playlist_path(camera_id)
    if not playlist_path or not Path(playlist_path).exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stream not ready - waiting for camera response",
        )

    return FileResponse(
        playlist_path,
        media_type="application/vnd.apple.mpegurl",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/{camera_id}/stream/hls/{segment}")
async def get_hls_segment(
    camera_id: int,
    segment: str,
) -> FileResponse:
    """Get HLS segment file."""
    if not segment.endswith(".ts"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid segment name",
        )

    segment_path = stream_manager.get_segment_path(camera_id, segment)
    if not segment_path or not Path(segment_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Segment not found",
        )

    return FileResponse(
        segment_path,
        media_type="video/mp2t",
        headers={
            "Cache-Control": "max-age=3600",
            "Access-Control-Allow-Origin": "*",
        },
    )


# Legacy recording endpoints (now use unified stream)

@router.post("/{camera_id}/recording/start")
async def start_recording(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Start recording for a camera (starts stream with recording enabled)."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    success = await stream_manager.start_stream(camera, recording_enabled=True)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start recording",
        )

    return {
        "camera_id": camera_id,
        "is_recording": True,
        "streaming": True,
    }


@router.post("/{camera_id}/recording/stop")
async def stop_recording(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Stop recording for a camera (stops the stream entirely)."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    await stream_manager.stop_stream(camera_id)
    return {
        "camera_id": camera_id,
        "is_recording": False,
        "streaming": False,
    }


@router.get("/{camera_id}/recording/status")
async def get_recording_status(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get recording status for a camera."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    status_info = stream_manager.get_status(camera_id)
    if not status_info:
        return {
            "camera_id": camera_id,
            "is_recording": False,
        }

    return {
        "camera_id": camera_id,
        "is_recording": status_info.get("is_recording", False),
        "state": status_info.get("state", "stopped"),
        "start_time": status_info.get("start_time"),
        "recording_directory": status_info.get("recording_directory"),
    }
