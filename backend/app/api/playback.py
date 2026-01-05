"""Playback API endpoints for viewing recorded videos."""

from datetime import date, datetime
from typing import Optional, Union

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.dependencies import get_current_user
from app.models.user import User
from app.rate_limiter import limiter
from app.services.playback import playback_service

router = APIRouter(prefix="/playback", tags=["playback"])


class RecordingFileResponse(BaseModel):
    """Response for a single recording file."""

    id: str
    camera_name: str
    date: str
    start_time: str
    duration_seconds: Optional[int]
    size_bytes: int
    filename: str
    is_in_progress: bool = False  # True if recording is currently being written


class DayRecordingsResponse(BaseModel):
    """Response for recordings on a single day."""

    camera_name: str
    date: str
    files: list[RecordingFileResponse]
    total_duration_seconds: int
    total_size_bytes: int
    start_time: Optional[str]
    end_time: Optional[str]


class AvailableDatesResponse(BaseModel):
    """Response for available recording dates."""

    camera_name: str
    dates: list[str]


class CamerasWithRecordingsResponse(BaseModel):
    """Response for cameras that have recordings."""

    cameras: list[str]


class ExportRequest(BaseModel):
    """Request to export a clip."""

    camera_name: str
    start_time: datetime
    end_time: datetime


class ExportResponse(BaseModel):
    """Response for export request."""

    success: bool
    message: str
    download_url: Optional[str] = None


@router.get("/cameras", response_model=CamerasWithRecordingsResponse)
async def get_cameras_with_recordings(
    current_user: User = Depends(get_current_user),
) -> CamerasWithRecordingsResponse:
    """Get list of cameras that have recordings."""
    cameras = playback_service.get_cameras_with_recordings()
    return CamerasWithRecordingsResponse(cameras=cameras)


@router.get("/cameras/{camera_name}/dates", response_model=AvailableDatesResponse)
async def get_available_dates(
    camera_name: str,
    current_user: User = Depends(get_current_user),
) -> AvailableDatesResponse:
    """Get available recording dates for a camera."""
    dates = playback_service.get_available_dates(camera_name)
    return AvailableDatesResponse(
        camera_name=camera_name,
        dates=[d.isoformat() for d in dates],
    )


@router.get("/cameras/{camera_name}/recordings", response_model=DayRecordingsResponse)
async def get_day_recordings(
    camera_name: str,
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    current_user: User = Depends(get_current_user),
) -> DayRecordingsResponse:
    """Get all recordings for a camera on a specific date."""
    try:
        rec_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD",
        )

    day_recs = playback_service.get_day_recordings(camera_name, rec_date)

    if not day_recs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No recordings found for {camera_name} on {date}",
        )

    files = [
        RecordingFileResponse(
            id=f.id,
            camera_name=f.camera_name,
            date=f.date.isoformat(),
            start_time=f.start_time.isoformat(),
            duration_seconds=f.duration_seconds,
            size_bytes=f.size,
            filename=f.filename,
            is_in_progress=f.is_in_progress,
        )
        for f in day_recs.files
    ]

    return DayRecordingsResponse(
        camera_name=day_recs.camera_name,
        date=day_recs.date.isoformat(),
        files=files,
        total_duration_seconds=day_recs.total_duration_seconds,
        total_size_bytes=day_recs.total_size_bytes,
        start_time=day_recs.start_time.isoformat() if day_recs.start_time else None,
        end_time=day_recs.end_time.isoformat() if day_recs.end_time else None,
    )


@router.get("/recordings")
async def list_recordings(
    camera_name: Optional[str] = Query(None, description="Filter by camera name"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
) -> dict:
    """List recordings with optional filters."""
    # Parse dates
    start = None
    end = None

    if start_date:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid start_date format. Use YYYY-MM-DD",
            )

    if end_date:
        try:
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid end_date format. Use YYYY-MM-DD",
            )

    recordings = playback_service.scan_recordings(
        camera_name=camera_name,
        start_date=start,
        end_date=end,
    )

    # Apply pagination
    total = len(recordings)
    recordings = recordings[offset:offset + limit]

    files = [
        RecordingFileResponse(
            id=f.id,
            camera_name=f.camera_name,
            date=f.date.isoformat(),
            start_time=f.start_time.isoformat(),
            duration_seconds=f.duration_seconds,
            size_bytes=f.size,
            filename=f.filename,
            is_in_progress=f.is_in_progress,
        )
        for f in recordings
    ]

    return {
        "recordings": files,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/recordings/{recording_id}")
async def get_recording(
    recording_id: str,
    current_user: User = Depends(get_current_user),
) -> RecordingFileResponse:
    """Get details of a specific recording."""
    rec = playback_service.get_recording_by_id(recording_id)

    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording {recording_id} not found",
        )

    return RecordingFileResponse(
        id=rec.id,
        camera_name=rec.camera_name,
        date=rec.date.isoformat(),
        start_time=rec.start_time.isoformat(),
        duration_seconds=rec.duration_seconds,
        size_bytes=rec.size,
        filename=rec.filename,
        is_in_progress=rec.is_in_progress,
    )


@router.get("/recordings/{recording_id}/stream", response_model=None)
async def stream_recording(
    recording_id: str,
    request: Request,
) -> Union[FileResponse, StreamingResponse]:
    """Stream a recording file with Range request support.

    Note: No auth required - video players can't send Authorization headers.

    For in-progress recordings, uses StreamingResponse with Range support.
    For completed recordings, uses FileResponse for better efficiency.
    """
    rec = playback_service.get_recording_by_id(recording_id)

    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording {recording_id} not found",
        )

    # For in-progress recordings, handle Range requests manually
    if rec.is_in_progress:
        # Get current file size (may grow during streaming)
        file_size = rec.path.stat().st_size

        # Parse Range header
        range_header = request.headers.get("range")
        start = 0
        end = file_size - 1

        if range_header:
            # Parse "bytes=start-end" format
            range_match = range_header.replace("bytes=", "").split("-")
            if range_match[0]:
                start = int(range_match[0])
            if len(range_match) > 1 and range_match[1]:
                end = min(int(range_match[1]), file_size - 1)

        # Ensure valid range
        if start >= file_size:
            raise HTTPException(
                status_code=416,
                detail="Range not satisfiable",
                headers={"Content-Range": f"bytes */{file_size}"},
            )

        content_length = end - start + 1

        async def stream_range():
            async with aiofiles.open(rec.path, "rb") as f:
                await f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(64 * 1024, remaining)
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        # Return 206 Partial Content for Range requests, 200 for full file
        status_code = 206 if range_header else 200
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Disposition": f'inline; filename="{rec.filename}"',
        }
        if range_header:
            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"

        return StreamingResponse(
            stream_range(),
            status_code=status_code,
            media_type="video/mp4",
            headers=headers,
        )

    # For completed recordings, use FileResponse (more efficient, handles Range)
    return FileResponse(
        rec.path,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{rec.filename}"',
        },
    )


@router.get("/recordings/{recording_id}/download")
async def download_recording(
    recording_id: str,
) -> FileResponse:
    """Download a recording file.

    Note: No auth required - video players can't send Authorization headers.
    """
    rec = playback_service.get_recording_by_id(recording_id)

    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording {recording_id} not found",
        )

    return FileResponse(
        rec.path,
        media_type="video/mp4",
        filename=rec.filename,
        headers={
            "Content-Disposition": f'attachment; filename="{rec.filename}"',
        },
    )


@router.post("/export", response_model=ExportResponse)
@limiter.limit("5/minute")
async def export_clip(
    request: Request,
    export_request: ExportRequest,
    current_user: User = Depends(get_current_user),
) -> ExportResponse:
    """Export a time range as a single video file."""
    if export_request.end_time <= export_request.start_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="end_time must be after start_time",
        )

    # Limit export duration to 1 hour
    duration = (export_request.end_time - export_request.start_time).total_seconds()
    if duration > 3600:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Export duration cannot exceed 1 hour",
        )

    output_path = playback_service.export_clip(
        camera_name=export_request.camera_name,
        start_time=export_request.start_time,
        end_time=export_request.end_time,
    )

    if not output_path:
        return ExportResponse(
            success=False,
            message="Export failed. No recordings found for the specified time range.",
        )

    # Generate download URL (relative path from exports directory)
    export_id = output_path.name
    download_url = f"/api/playback/exports/{export_id}"

    return ExportResponse(
        success=True,
        message="Export completed successfully",
        download_url=download_url,
    )


@router.get("/exports/{export_id}")
async def download_export(
    export_id: str,
) -> FileResponse:
    """Download an exported clip.

    Note: No auth required - video players can't send Authorization headers.
    """
    from app.config import get_settings
    from pathlib import Path

    settings = get_settings()
    export_path = Path(settings.storage_root) / ".exports" / export_id

    if not export_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Export {export_id} not found",
        )

    return FileResponse(
        export_path,
        media_type="video/mp4",
        filename=export_id,
        headers={
            "Content-Disposition": f'attachment; filename="{export_id}"',
        },
    )
