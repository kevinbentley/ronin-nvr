"""Storage management API endpoints."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_admin_user, get_current_user
from app.models.user import User
from app.services.retention import retention_service

router = APIRouter(prefix="/storage", tags=["storage"])


class CameraStorageStats(BaseModel):
    """Storage stats for a single camera."""

    name: str
    size_bytes: int
    size_gb: float
    file_count: int


class StorageStatsResponse(BaseModel):
    """Storage statistics response."""

    total_size_bytes: int
    total_size_gb: float
    total_size_mb: float
    total_files: int
    oldest_file: Optional[datetime] = None
    newest_file: Optional[datetime] = None
    cameras: list[CameraStorageStats]


class RetentionResult(BaseModel):
    """Result of retention enforcement."""

    files_scanned: int
    files_deleted: int
    bytes_freed: int
    gb_freed: float
    storage_before_gb: float
    storage_after_gb: float


@router.get("/stats", response_model=StorageStatsResponse)
async def get_storage_stats(
    current_user: User = Depends(get_current_user),
) -> StorageStatsResponse:
    """Get storage statistics."""
    stats = retention_service.get_stats()

    cameras = [
        CameraStorageStats(
            name=name,
            size_bytes=data["size_bytes"],
            size_gb=data["size_gb"],
            file_count=data["file_count"],
        )
        for name, data in stats.cameras.items()
    ]

    return StorageStatsResponse(
        total_size_bytes=stats.total_size_bytes,
        total_size_gb=stats.total_size_gb,
        total_size_mb=stats.total_size_mb,
        total_files=stats.total_files,
        oldest_file=stats.oldest_file,
        newest_file=stats.newest_file,
        cameras=cameras,
    )


@router.post("/cleanup", response_model=RetentionResult)
async def run_retention_cleanup(
    admin_user: User = Depends(get_admin_user),
) -> RetentionResult:
    """Manually trigger retention cleanup."""
    result = retention_service.enforce_retention()
    return RetentionResult(**result)
