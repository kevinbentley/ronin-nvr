"""Camera management API endpoints."""

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
from app.services.streaming import streaming_manager

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("", response_model=CameraListResponse)
async def list_cameras(db: AsyncSession = Depends(get_db)) -> CameraListResponse:
    """Get all cameras."""
    service = CameraService(db)
    cameras = await service.get_all()
    return CameraListResponse(
        cameras=[CameraResponse.model_validate(c) for c in cameras],
        total=len(cameras),
    )


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


# HLS Streaming endpoints

@router.post("/{camera_id}/stream/start")
async def start_stream(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Start HLS stream for a camera."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    success = streaming_manager.start_stream(camera)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start stream",
        )

    return {
        "camera_id": camera_id,
        "streaming": True,
        "playlist_url": f"/api/cameras/{camera_id}/stream/hls/playlist.m3u8",
    }


@router.post("/{camera_id}/stream/stop")
async def stop_stream(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Stop HLS stream for a camera."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    streaming_manager.stop_stream(camera_id)
    return {"camera_id": camera_id, "streaming": False}


@router.get("/{camera_id}/stream/hls/playlist.m3u8")
async def get_hls_playlist(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """Get HLS playlist for a camera stream."""
    import asyncio
    from pathlib import Path

    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    # Auto-start stream if not running
    if camera_id not in streaming_manager.streams or not streaming_manager.streams[camera_id].is_running:
        success = streaming_manager.start_stream(camera)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to start stream - check camera connection",
            )
        # Wait briefly for FFmpeg to generate playlist
        for _ in range(15):  # Wait up to 3 seconds
            await asyncio.sleep(0.2)
            playlist_path = streaming_manager.get_playlist_path(camera_id)
            if playlist_path and Path(playlist_path).exists():
                break

    playlist_path = streaming_manager.get_playlist_path(camera_id)
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
    from pathlib import Path

    if not segment.endswith(".ts"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid segment name",
        )

    segment_path = streaming_manager.get_segment_path(camera_id, segment)
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
