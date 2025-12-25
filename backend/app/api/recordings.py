"""Recording management API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import (
    RecordingActionResponse,
    RecordingStatusResponse,
)
from app.services.camera import CameraService
from app.services.recorder import recording_manager

router = APIRouter(prefix="/cameras", tags=["recording"])


@router.post("/{camera_id}/recording/start", response_model=RecordingActionResponse)
async def start_recording(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
) -> RecordingActionResponse:
    """Start recording for a camera."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    if recording_manager.is_recording(camera_id):
        return RecordingActionResponse(
            success=True,
            message="Recording already in progress",
            status=RecordingStatusResponse(**recording_manager.get_status(camera_id)),
        )

    success = await recording_manager.start_recording(camera)
    status_info = recording_manager.get_status(camera_id)

    if success:
        return RecordingActionResponse(
            success=True,
            message="Recording started",
            status=RecordingStatusResponse(**status_info) if status_info else None,
        )
    else:
        return RecordingActionResponse(
            success=False,
            message=status_info.get("error_message", "Failed to start recording")
            if status_info
            else "Failed to start recording",
            status=RecordingStatusResponse(**status_info) if status_info else None,
        )


@router.post("/{camera_id}/recording/stop", response_model=RecordingActionResponse)
async def stop_recording(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
) -> RecordingActionResponse:
    """Stop recording for a camera."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    if not recording_manager.is_recording(camera_id):
        status_info = recording_manager.get_status(camera_id)
        return RecordingActionResponse(
            success=True,
            message="Recording not in progress",
            status=RecordingStatusResponse(**status_info) if status_info else None,
        )

    await recording_manager.stop_recording(camera_id)
    status_info = recording_manager.get_status(camera_id)

    return RecordingActionResponse(
        success=True,
        message="Recording stopped",
        status=RecordingStatusResponse(**status_info) if status_info else None,
    )


@router.get("/{camera_id}/recording/status", response_model=RecordingStatusResponse)
async def get_recording_status(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
) -> RecordingStatusResponse:
    """Get recording status for a camera."""
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} not found",
        )

    status_info = recording_manager.get_status(camera_id)
    if not status_info:
        # Camera exists but no recorder initialized yet
        return RecordingStatusResponse(
            camera_id=camera_id,
            camera_name=camera.name,
            state="stopped",
            error_message=None,
            start_time=None,
            output_directory="",
            reconnect_attempts=0,
        )

    return RecordingStatusResponse(**status_info)


@router.get("/recording/status", response_model=list[RecordingStatusResponse])
async def get_all_recording_status() -> list[RecordingStatusResponse]:
    """Get recording status for all cameras."""
    statuses = recording_manager.get_all_status()
    return [RecordingStatusResponse(**s) for s in statuses]
