"""Storage management API endpoints."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.config import get_settings
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


# === Retention Settings ===


class RetentionSettings(BaseModel):
    """Retention policy settings."""

    retention_days: Optional[int] = Field(
        None, description="Days to keep recordings (null = unlimited)"
    )
    retention_max_gb: Optional[float] = Field(
        None, description="Max storage in GB (null = unlimited)"
    )
    retention_check_interval_minutes: int = Field(
        60, description="Minutes between automatic retention checks"
    )


class RetentionSettingsUpdate(BaseModel):
    """Update request for retention settings."""

    retention_days: Optional[int] = Field(
        None, ge=1, le=3650, description="Days to keep recordings (1-3650, null = unlimited)"
    )
    retention_max_gb: Optional[float] = Field(
        None, ge=1, le=100000, description="Max storage in GB (1-100000, null = unlimited)"
    )


def _get_retention_settings_path() -> Path:
    """Get path to retention settings JSON file."""
    settings = get_settings()
    return settings.storage_root / ".retention_settings.json"


def _load_retention_settings() -> dict:
    """Load retention settings from file or return defaults from config."""
    settings_path = _get_retention_settings_path()
    settings = get_settings()

    defaults = {
        "retention_days": settings.retention_days,
        "retention_max_gb": settings.retention_max_gb,
        "retention_check_interval_minutes": settings.retention_check_interval_minutes,
    }

    if settings_path.exists():
        try:
            with open(settings_path) as f:
                saved = json.load(f)
                # Merge saved values over defaults
                defaults.update(saved)
        except (json.JSONDecodeError, IOError):
            pass

    return defaults


def _save_retention_settings(data: dict) -> None:
    """Save retention settings to file."""
    settings_path = _get_retention_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_path, "w") as f:
        json.dump(data, f, indent=2)


@router.get("/retention/settings", response_model=RetentionSettings)
async def get_retention_settings(
    current_user: User = Depends(get_current_user),
) -> RetentionSettings:
    """Get current retention policy settings."""
    data = _load_retention_settings()
    return RetentionSettings(**data)


@router.put("/retention/settings", response_model=RetentionSettings)
async def update_retention_settings(
    update: RetentionSettingsUpdate,
    admin_user: User = Depends(get_admin_user),
) -> RetentionSettings:
    """Update retention policy settings.

    Note: Changes take effect on the next retention check cycle.
    """
    current = _load_retention_settings()

    # Update only provided fields
    if update.retention_days is not None:
        current["retention_days"] = update.retention_days
    elif "retention_days" in update.model_fields_set:
        # Explicitly set to None (unlimited)
        current["retention_days"] = None

    if update.retention_max_gb is not None:
        current["retention_max_gb"] = update.retention_max_gb
    elif "retention_max_gb" in update.model_fields_set:
        # Explicitly set to None (unlimited)
        current["retention_max_gb"] = None

    _save_retention_settings(current)

    # Update the retention service to use new settings
    retention_service.reload_settings(
        retention_days=current.get("retention_days"),
        retention_max_gb=current.get("retention_max_gb"),
    )

    return RetentionSettings(**current)


# === Transcoding Status ===


class TranscodeFileStats(BaseModel):
    """Stats for a transcoded file."""

    original_size: int
    new_size: int
    savings_percent: float
    duration_seconds: float
    encoder: str
    transcoded_at: str


class TranscodeQueueStatus(BaseModel):
    """Transcode queue/backlog status."""

    pending_files: int
    pending_size_bytes: int
    pending_size_gb: float


class TranscodeWorkerInfo(BaseModel):
    """Information about a transcode worker."""

    worker_id: str
    is_active: bool
    current_file: Optional[str] = None
    last_seen: Optional[str] = None


class TranscodeStatsResponse(BaseModel):
    """Transcoding statistics response."""

    enabled: bool
    files_transcoded: int
    files_failed: int
    total_original_gb: float
    total_new_gb: float
    total_savings_gb: float
    average_savings_percent: float
    by_encoder: dict[str, int]
    queue: TranscodeQueueStatus
    workers: list[TranscodeWorkerInfo]


def _get_transcode_status_path() -> Path:
    """Get path to transcode status JSON file."""
    settings = get_settings()
    return settings.storage_root / ".transcode_status.json"


def _scan_pending_transcode_files() -> tuple[int, int]:
    """Scan storage for files pending transcoding.

    Returns:
        Tuple of (pending_count, pending_size_bytes)
    """
    settings = get_settings()
    storage_root = settings.storage_root

    # Load transcoded files set
    status_path = _get_transcode_status_path()
    transcoded_files: set[str] = set()
    failed_files: set[str] = set()

    if status_path.exists():
        try:
            with open(status_path) as f:
                data = json.load(f)
                transcoded_files = set(data.get("transcoded", {}).keys())
                failed_files = set(data.get("failed", {}).keys())
        except (json.JSONDecodeError, IOError):
            pass

    # Scan for mp4 files not yet transcoded
    pending_count = 0
    pending_size = 0
    min_age_minutes = settings.transcode_min_age_minutes
    now = datetime.now().timestamp()

    for mp4_file in storage_root.rglob("*.mp4"):
        # Skip hidden directories and files
        if any(part.startswith(".") for part in mp4_file.parts):
            continue
        if mp4_file.name.startswith("."):
            continue

        file_path_str = str(mp4_file)

        # Skip if already transcoded or failed
        if file_path_str in transcoded_files or file_path_str in failed_files:
            continue

        # Skip if too recent
        try:
            mtime = mp4_file.stat().st_mtime
            if (now - mtime) < (min_age_minutes * 60):
                continue
            pending_count += 1
            pending_size += mp4_file.stat().st_size
        except OSError:
            continue

    return pending_count, pending_size


def _get_active_workers() -> list[TranscodeWorkerInfo]:
    """Get list of active transcode workers by checking lock files."""
    settings = get_settings()
    storage_root = settings.storage_root
    workers: list[TranscodeWorkerInfo] = []

    # Look for lock files that indicate active workers
    for lock_file in storage_root.rglob(".lock_*.mp4"):
        try:
            stat = lock_file.stat()
            lock_age_seconds = datetime.now().timestamp() - stat.st_mtime

            # Read worker PID from lock file
            try:
                with open(lock_file) as f:
                    pid = f.read().strip()
            except IOError:
                pid = "unknown"

            # Check if lock is stale (older than 2 hours = 7200 seconds)
            is_active = lock_age_seconds < 7200

            # Get the filename being processed
            filename = lock_file.name.replace(".lock_", "")

            workers.append(
                TranscodeWorkerInfo(
                    worker_id=f"pid-{pid}",
                    is_active=is_active,
                    current_file=filename if is_active else None,
                    last_seen=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                )
            )
        except OSError:
            continue

    return workers


@router.get("/transcode/status", response_model=TranscodeStatsResponse)
async def get_transcode_status(
    current_user: User = Depends(get_current_user),
) -> TranscodeStatsResponse:
    """Get transcoding status and statistics.

    Returns information about the transcoding system including:
    - Total files transcoded and failed
    - Storage savings achieved
    - Current queue/backlog of files waiting to be transcoded
    - Active worker processes
    """
    settings = get_settings()
    status_path = _get_transcode_status_path()

    # Default stats if no status file exists
    stats = {
        "files_transcoded": 0,
        "files_failed": 0,
        "total_original_gb": 0.0,
        "total_new_gb": 0.0,
        "total_savings_gb": 0.0,
        "average_savings_percent": 0.0,
        "by_encoder": {},
    }

    # Load stats from transcode status file
    if status_path.exists():
        try:
            with open(status_path) as f:
                data = json.load(f)

            transcoded = data.get("transcoded", {})
            failed = data.get("failed", {})

            total_original = sum(
                v.get("original_size", 0) for v in transcoded.values()
            )
            total_new = sum(v.get("new_size", 0) for v in transcoded.values())

            # Count by encoder type
            encoder_counts: dict[str, int] = {}
            for v in transcoded.values():
                enc = v.get("encoder", "unknown")
                encoder_counts[enc] = encoder_counts.get(enc, 0) + 1

            stats = {
                "files_transcoded": len(transcoded),
                "files_failed": len(failed),
                "total_original_gb": round(total_original / (1024**3), 2),
                "total_new_gb": round(total_new / (1024**3), 2),
                "total_savings_gb": round(
                    (total_original - total_new) / (1024**3), 2
                ),
                "average_savings_percent": (
                    round((1 - total_new / total_original) * 100, 1)
                    if total_original > 0
                    else 0.0
                ),
                "by_encoder": encoder_counts,
            }
        except (json.JSONDecodeError, IOError):
            pass

    # Get pending queue info
    pending_count, pending_size = _scan_pending_transcode_files()
    queue = TranscodeQueueStatus(
        pending_files=pending_count,
        pending_size_bytes=pending_size,
        pending_size_gb=round(pending_size / (1024**3), 2),
    )

    # Get active workers
    workers = _get_active_workers()

    return TranscodeStatsResponse(
        enabled=settings.transcode_enabled,
        files_transcoded=stats["files_transcoded"],
        files_failed=stats["files_failed"],
        total_original_gb=stats["total_original_gb"],
        total_new_gb=stats["total_new_gb"],
        total_savings_gb=stats["total_savings_gb"],
        average_savings_percent=stats["average_savings_percent"],
        by_encoder=stats["by_encoder"],
        queue=queue,
        workers=workers,
    )
