"""ML API endpoints for managing inference jobs and detections."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.detection import Detection
from app.models.ml_job import JobStatus, MLJob
from app.models.ml_model import MLModel
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
    current_user: User = Depends(get_current_user),
) -> MLStatusResponse:
    """Get ML system status."""
    status_data = ml_coordinator.get_status()

    return MLStatusResponse(
        running=status_data["running"],
        workers=status_data["workers"],
        worker_status=[
            WorkerStatusResponse(**w) for w in status_data["worker_status"]
        ],
        queue=QueueStatusResponse(**status_data["queue"]),
        models_loaded=status_data["models_loaded"],
    )
