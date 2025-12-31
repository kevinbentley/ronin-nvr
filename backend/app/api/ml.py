"""ML API endpoints for managing inference jobs and detections."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.dependencies import get_current_user
from app.models.camera import Camera
from app.models.detection import Detection
from app.models.ml_job import JobStatus, MLJob
from app.models.ml_model import MLModel
from app.models.recording import Recording
from app.models.user import User
from app.schemas.ml import (
    CameraMLSettingsRequest,
    CameraMLSettingsResponse,
    CreateJobRequest,
    DetectionListResponse,
    DetectionResponse,
    DetectionSummaryItem,
    DetectionSummaryResponse,
    JobListResponse,
    JobResponse,
    MLStatusResponse,
    ModelConfigRequest,
    ModelListResponse,
    ModelResponse,
    QueueStatusResponse,
    TimelineEvent,
    TimelineEventsResponse,
    WorkerStatusResponse,
)
from app.services.ml import ml_coordinator

router = APIRouter(prefix="/ml", tags=["ml"])


# === Jobs ===


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status_filter: Optional[str] = Query(None, alias="status"),
    recording_id: Optional[int] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> JobListResponse:
    """List ML processing jobs with optional filters."""
    query = select(MLJob)

    if status_filter:
        query = query.where(MLJob.status == status_filter)
    if recording_id:
        query = query.where(MLJob.recording_id == recording_id)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query) or 0

    # Get paginated results
    query = query.order_by(MLJob.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    jobs = result.scalars().all()

    return JobListResponse(
        jobs=[JobResponse.model_validate(job) for job in jobs],
        total=total,
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    """Get details of a specific job."""
    job = await db.get(MLJob, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    return JobResponse.model_validate(job)


@router.post("/jobs", response_model=JobResponse)
async def create_job(
    request: CreateJobRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    """Create a new ML processing job."""
    job = await ml_coordinator.process_recording(
        recording_id=request.recording_id,
        model_name=request.model_name,
        priority=request.priority,
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not create job. Recording may not exist or job already pending.",
        )

    # Refresh from database
    await db.refresh(job)
    return JobResponse.model_validate(job)


@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Cancel a pending job."""
    success = await ml_coordinator.cancel_job(job_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not cancel job. It may already be processing or completed.",
        )

    return {"success": True, "message": f"Job {job_id} cancelled"}


# === Detections ===


@router.get("/detections", response_model=DetectionListResponse)
async def list_detections(
    camera_id: Optional[int] = None,
    recording_id: Optional[int] = None,
    class_name: Optional[str] = None,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DetectionListResponse:
    """Query detections with filters."""
    query = select(Detection)

    if camera_id:
        query = query.where(Detection.camera_id == camera_id)
    if recording_id:
        query = query.where(Detection.recording_id == recording_id)
    if class_name:
        query = query.where(Detection.class_name == class_name)
    if min_confidence > 0:
        query = query.where(Detection.confidence >= min_confidence)
    if start_time:
        query = query.where(Detection.created_at >= start_time)
    if end_time:
        query = query.where(Detection.created_at <= end_time)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query) or 0

    # Get paginated results
    query = query.order_by(Detection.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    detections = result.scalars().all()

    return DetectionListResponse(
        detections=[DetectionResponse.model_validate(d) for d in detections],
        total=total,
    )


@router.get("/detections/summary", response_model=DetectionSummaryResponse)
async def get_detection_summary(
    camera_id: Optional[int] = None,
    recording_id: Optional[int] = None,
    group_by: str = Query("class_name", pattern="^(class_name|camera|hour)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DetectionSummaryResponse:
    """Get aggregated detection statistics."""
    base_query = select(Detection)

    if camera_id:
        base_query = base_query.where(Detection.camera_id == camera_id)
    if recording_id:
        base_query = base_query.where(Detection.recording_id == recording_id)

    # Get total count
    total_query = select(func.count()).select_from(base_query.subquery())
    total = await db.scalar(total_query) or 0

    # Group by class name (most common use case)
    if group_by == "class_name":
        query = (
            select(
                Detection.class_name,
                func.count().label("count"),
                func.avg(Detection.confidence).label("avg_confidence"),
            )
            .group_by(Detection.class_name)
            .order_by(func.count().desc())
        )

        if camera_id:
            query = query.where(Detection.camera_id == camera_id)
        if recording_id:
            query = query.where(Detection.recording_id == recording_id)

        result = await db.execute(query)
        rows = result.all()

        items = [
            DetectionSummaryItem(
                label=row.class_name,
                count=row.count,
                avg_confidence=float(row.avg_confidence or 0),
            )
            for row in rows
        ]
    else:
        items = []

    return DetectionSummaryResponse(
        total_detections=total,
        items=items,
    )


@router.delete("/detections")
async def delete_detections(
    recording_id: Optional[int] = None,
    before_date: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Delete detections for cleanup or reprocessing."""
    if not recording_id and not before_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must specify recording_id or before_date",
        )

    query = delete(Detection)

    if recording_id:
        query = query.where(Detection.recording_id == recording_id)
    if before_date:
        query = query.where(Detection.created_at < before_date)

    result = await db.execute(query)
    await db.commit()

    return {
        "success": True,
        "deleted": result.rowcount,
    }


def _safe_camera_name(name: str) -> str:
    """Convert camera name to filesystem-safe version."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


@router.get("/detections/timeline", response_model=TimelineEventsResponse)
async def get_timeline_events(
    camera_name: str,
    date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
    class_filter: Optional[str] = None,
    min_confidence: float = Query(0.3, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TimelineEventsResponse:
    """Get detection events for timeline display.

    Returns events for a specific camera and date, optimized for timeline visualization.
    Events are grouped into time buckets to reduce data volume.
    """
    # Get camera by name - try exact match first, then filesystem-safe match
    result = await db.execute(
        select(Camera).where(Camera.name == camera_name)
    )
    camera = result.scalar_one_or_none()

    # If not found, the camera_name might be filesystem-safe format (underscores)
    # Try to find a camera whose safe name matches
    if not camera:
        all_cameras = await db.execute(select(Camera))
        for cam in all_cameras.scalars().all():
            if _safe_camera_name(cam.name) == camera_name:
                camera = cam
                break

    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_name}' not found",
        )

    # Parse date and get recording IDs for that day
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD",
        )

    # Get recordings for this camera on this date
    start_of_day = datetime.combine(target_date, datetime.min.time())
    end_of_day = datetime.combine(target_date, datetime.max.time())

    recordings_result = await db.execute(
        select(Recording.id, Recording.start_time)
        .where(Recording.camera_id == camera.id)
        .where(Recording.start_time >= start_of_day)
        .where(Recording.start_time <= end_of_day)
    )
    recordings = recordings_result.all()

    if not recordings:
        return TimelineEventsResponse(events=[], total=0, class_counts={})

    recording_ids = [r.id for r in recordings]
    recording_times = {r.id: r.start_time for r in recordings}

    # Query detections for these recordings
    query = (
        select(Detection)
        .where(Detection.recording_id.in_(recording_ids))
        .where(Detection.confidence >= min_confidence)
    )

    if class_filter:
        query = query.where(Detection.class_name == class_filter)

    query = query.order_by(Detection.timestamp_ms)

    result = await db.execute(query)
    detections = result.scalars().all()

    # Group detections into time buckets (30-second intervals)
    bucket_size_ms = 30000  # 30 seconds
    buckets: dict[tuple[int, str], dict] = {}  # (bucket_time, class) -> data
    class_counts: dict[str, int] = {}

    for det in detections:
        # Calculate absolute time in day (ms from midnight)
        recording_start = recording_times.get(det.recording_id)
        if recording_start:
            recording_start_ms = (
                recording_start.hour * 3600000
                + recording_start.minute * 60000
                + recording_start.second * 1000
            )
            abs_time_ms = recording_start_ms + det.timestamp_ms
        else:
            abs_time_ms = det.timestamp_ms

        # Round to bucket
        bucket_time = (abs_time_ms // bucket_size_ms) * bucket_size_ms
        bucket_key = (bucket_time, det.class_name)

        if bucket_key not in buckets:
            buckets[bucket_key] = {
                "timestamp_ms": bucket_time,
                "class_name": det.class_name,
                "confidence": det.confidence,
                "recording_id": det.recording_id,
                "count": 0,
            }

        buckets[bucket_key]["count"] += 1
        buckets[bucket_key]["confidence"] = max(
            buckets[bucket_key]["confidence"], det.confidence
        )

        # Update class counts
        class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

    events = [
        TimelineEvent(**bucket_data)
        for bucket_data in sorted(buckets.values(), key=lambda x: x["timestamp_ms"])
    ]

    return TimelineEventsResponse(
        events=events,
        total=len(detections),
        class_counts=class_counts,
    )


# === Models ===


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ModelListResponse:
    """List available ML models."""
    result = await db.execute(
        select(MLModel).order_by(MLModel.name)
    )
    models = result.scalars().all()

    return ModelListResponse(
        models=[ModelResponse.model_validate(m) for m in models]
    )


@router.get("/models/{model_name}", response_model=ModelResponse)
async def get_model(
    model_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ModelResponse:
    """Get model details."""
    result = await db.execute(
        select(MLModel).where(MLModel.name == model_name)
    )
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_name} not found",
        )

    return ModelResponse.model_validate(model)


@router.put("/models/{model_name}", response_model=ModelResponse)
async def update_model(
    model_name: str,
    config: ModelConfigRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ModelResponse:
    """Update model configuration."""
    result = await db.execute(
        select(MLModel).where(MLModel.name == model_name)
    )
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_name} not found",
        )

    # Update fields
    update_data = config.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(model, key, value)

    # If setting as default, unset other defaults
    if config.is_default:
        await db.execute(
            select(MLModel)
            .where(MLModel.id != model.id, MLModel.is_default == True)
        )
        # Reset other defaults - need proper update statement
        from sqlalchemy import update
        await db.execute(
            update(MLModel)
            .where(MLModel.id != model.id)
            .values(is_default=False)
        )

    await db.commit()
    await db.refresh(model)

    return ModelResponse.model_validate(model)


@router.delete("/models/{model_name}")
async def delete_model(
    model_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Remove a model from the registry."""
    result = await db.execute(
        select(MLModel).where(MLModel.name == model_name)
    )
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_name} not found",
        )

    await db.delete(model)
    await db.commit()

    return {"success": True, "message": f"Model {model_name} deleted"}


# === Status ===


@router.get("/status", response_model=MLStatusResponse)
async def get_ml_status(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MLStatusResponse:
    """Get ML system status."""
    # Ensure coordinator has session factory
    from app.database import async_session_maker
    ml_coordinator.set_session_factory(async_session_maker)

    status_data = await ml_coordinator.get_status()

    return MLStatusResponse(
        running=status_data["running"],
        workers=status_data["workers"],
        worker_status=[
            WorkerStatusResponse(**w) for w in status_data["worker_status"]
        ],
        queue=QueueStatusResponse(**status_data["queue"]),
        models_loaded=status_data["models_loaded"],
    )


# === Events ===


@router.get("/events")
async def ml_events(
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Server-Sent Events stream for real-time ML updates.

    Events include:
    - job_created: New job queued
    - job_started: Job processing began
    - job_progress: Job progress update
    - job_completed: Job finished successfully
    - job_failed: Job failed with error
    - detection: High-confidence detection found
    - live_detection: Real-time detection from live stream
    """
    from app.services.ml import ml_event_service

    async def event_generator():
        """Generate SSE events."""
        async for event in ml_event_service.subscribe():
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# === Control ===


@router.post("/start")
async def start_ml_system(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Start the ML processing system."""
    if ml_coordinator.is_running:
        return {"success": True, "message": "ML system already running"}

    from app.database import async_session_maker
    from app.services.ml import recording_watcher

    ml_coordinator.set_session_factory(async_session_maker)
    recording_watcher.set_session_factory(async_session_maker)

    await ml_coordinator.start()
    await recording_watcher.start()

    return {"success": True, "message": "ML system started"}


@router.post("/stop")
async def stop_ml_system(
    current_user: User = Depends(get_current_user),
) -> dict:
    """Stop the ML processing system."""
    from app.services.ml import recording_watcher

    await recording_watcher.stop()
    await ml_coordinator.stop()

    return {"success": True, "message": "ML system stopped"}


@router.post("/process-all")
async def process_all_recordings(
    camera_name: Optional[str] = Query(None, description="Filter by camera name"),
    limit: int = Query(100, ge=1, le=1000, description="Max recordings to queue"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Bulk-queue existing recordings for ML processing.

    Scans the filesystem for recordings that haven't been processed yet
    and queues them for ML analysis.
    """
    from app.services.ml import recording_watcher

    if not ml_coordinator.is_running:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ML system is not running. Start it first with POST /api/ml/start",
        )

    queued = await recording_watcher.process_existing_recordings(
        camera_name=camera_name,
        limit=limit,
    )

    return {
        "success": True,
        "queued": queued,
        "message": f"Queued {queued} recordings for processing",
    }


@router.post("/retry-failed")
async def retry_failed_jobs(
    error_filter: Optional[str] = Query(
        None, description="Only retry jobs with this error message (partial match)"
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Reset failed jobs back to PENDING so they can be retried.

    This is useful when jobs failed due to transient issues like queue full.
    """
    from sqlalchemy import update

    # Build query for failed jobs
    query = (
        update(MLJob)
        .where(MLJob.status == JobStatus.FAILED.value)
    )

    if error_filter:
        query = query.where(MLJob.error_message.contains(error_filter))

    query = query.values(
        status=JobStatus.PENDING.value,
        error_message=None,
        started_at=None,
        completed_at=None,
    )

    result = await db.execute(query)
    await db.commit()

    return {
        "success": True,
        "reset_count": result.rowcount,
        "message": f"Reset {result.rowcount} failed jobs to PENDING",
    }


@router.post("/reset-stuck")
async def reset_stuck_jobs(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Reset stuck processing jobs back to PENDING.

    When the server restarts, jobs that were in PROCESSING state become orphaned
    because no worker is actually processing them. This endpoint resets those
    jobs so they can be picked up again.
    """
    from sqlalchemy import update

    # Reset PROCESSING jobs back to PENDING
    query = (
        update(MLJob)
        .where(MLJob.status == JobStatus.PROCESSING.value)
        .values(
            status=JobStatus.PENDING.value,
            started_at=None,
            progress_percent=0,
            frames_processed=0,
        )
    )

    result = await db.execute(query)
    await db.commit()

    return {
        "success": True,
        "reset_count": result.rowcount,
        "message": f"Reset {result.rowcount} stuck processing jobs to PENDING",
    }


# === Live Detection ===


@router.get("/live-detections")
async def get_live_detections(
    camera_id: Optional[int] = None,
    class_name: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Get recent live detections.

    Live detections are those with recording_id=NULL, meaning they came
    from real-time stream analysis rather than historical file processing.
    """
    query = select(Detection).where(Detection.recording_id.is_(None))

    if camera_id:
        query = query.where(Detection.camera_id == camera_id)
    if class_name:
        query = query.where(Detection.class_name == class_name)
    if since:
        query = query.where(Detection.detected_at >= since)

    query = query.order_by(Detection.detected_at.desc()).limit(limit)
    result = await db.execute(query)
    detections = result.scalars().all()

    return {
        "detections": [
            {
                "id": d.id,
                "camera_id": d.camera_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "detected_at": d.detected_at.isoformat() if d.detected_at else None,
                "snapshot_url": f"/api/snapshots/{d.snapshot_path}"
                if d.snapshot_path
                else None,
                "bbox": {
                    "x": d.bbox_x,
                    "y": d.bbox_y,
                    "width": d.bbox_width,
                    "height": d.bbox_height,
                },
            }
            for d in detections
        ],
        "count": len(detections),
    }


@router.get("/live-detections/status")
async def get_live_detection_status(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Get live detection worker status."""
    settings = get_settings()

    # Count recent live detections (last hour)
    from datetime import timedelta

    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    count_query = select(func.count()).select_from(
        select(Detection)
        .where(Detection.recording_id.is_(None))
        .where(Detection.detected_at >= one_hour_ago)
        .subquery()
    )
    recent_count = await db.scalar(count_query) or 0

    # Get cameras with recent live detections
    cameras_query = (
        select(Detection.camera_id)
        .where(Detection.recording_id.is_(None))
        .where(Detection.detected_at >= one_hour_ago)
        .distinct()
    )
    result = await db.execute(cameras_query)
    active_cameras = [r[0] for r in result.all()]

    return {
        "enabled": settings.live_detection_enabled,
        "config": {
            "fps": settings.live_detection_fps,
            "cooldown": settings.live_detection_cooldown,
            "confidence": settings.live_detection_confidence,
            "classes": settings.live_detection_classes.split(","),
        },
        "detections_last_hour": recent_count,
        "active_cameras": active_cameras,
    }


# === Snapshots ===


@router.get("/snapshots/{path:path}")
async def get_snapshot(
    path: str,
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    """Serve snapshot images.

    Snapshots are JPG images with detection bounding boxes drawn,
    stored in .snapshots/{camera_id}/{date}/{time}.jpg
    """
    settings = get_settings()
    snapshot_path = Path(settings.storage_root) / ".snapshots" / path

    if not snapshot_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Snapshot not found",
        )

    # Security: ensure path doesn't escape snapshot directory
    try:
        snapshot_path.resolve().relative_to(
            (Path(settings.storage_root) / ".snapshots").resolve()
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid snapshot path",
        )

    return FileResponse(
        snapshot_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=86400"},  # Cache for 24 hours
    )
