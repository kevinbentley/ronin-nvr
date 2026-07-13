"""Storage management API endpoints."""

import asyncio
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import async_session_maker, get_db
from app.dependencies import get_admin_user, get_current_user
from app.models.recording import Recording, RecordingStatus, StorageTier
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


# ============================================================
# Tiered Storage API
# ============================================================


class TierStats(BaseModel):
    """Statistics for a storage tier."""

    enabled: bool
    total_size_gb: float
    file_count: int
    oldest_file: Optional[datetime] = None
    newest_file: Optional[datetime] = None
    max_size_gb: Optional[float] = None
    retention_days: Optional[int] = None
    percent_full: Optional[float] = None


class TierStatsResponse(BaseModel):
    """Response with stats for all storage tiers."""

    hot: TierStats
    warm: Optional[TierStats] = None
    cold: Optional[TierStats] = None
    # Filesystem-based stats (actual disk usage, not just DB-tracked)
    filesystem_total_gb: float = 0.0
    filesystem_total_files: int = 0


class TierConfigResponse(BaseModel):
    """Current tier configuration."""

    # Hot storage
    hot_max_gb: Optional[float] = None
    hot_retention_days: Optional[int] = None

    # Warm storage
    warm_storage_enabled: bool = False
    warm_storage_path: Optional[str] = None
    warm_max_gb: Optional[float] = None
    warm_retention_days: Optional[int] = None

    # Cold storage
    cold_storage_enabled: bool = False
    s3_endpoint_url: Optional[str] = None
    s3_bucket_name: Optional[str] = None
    s3_region: str = "us-east-1"
    s3_prefix: str = "ronin-nvr/"
    s3_configured: bool = False  # True if S3 credentials are set

    # Migration
    tier_migration_check_interval_minutes: int = 15


class TierConfigUpdate(BaseModel):
    """Update request for tier configuration."""

    # Hot storage thresholds
    hot_max_gb: Optional[float] = Field(
        None, ge=1, description="Migrate to warm when hot storage exceeds this GB"
    )
    hot_retention_days: Optional[int] = Field(
        None, ge=1, description="Migrate to warm when files older than this many days"
    )

    # Warm storage thresholds
    warm_max_gb: Optional[float] = Field(
        None, ge=1, description="Migrate to cold when warm storage exceeds this GB"
    )
    warm_retention_days: Optional[int] = Field(
        None, ge=1, description="Migrate to cold when files older than this many days"
    )


class MigrationTriggerRequest(BaseModel):
    """Request to manually trigger migration."""

    from_tier: str = Field(..., description="Source tier (hot or warm)")
    max_files: int = Field(100, ge=1, le=10000, description="Max files per batch")
    clean_orphans: bool = Field(
        False, description="Delete DB records for files that no longer exist"
    )


class MigrationResult(BaseModel):
    """Result of a migration operation."""

    files_migrated: int
    files_failed: int
    files_skipped: int = 0  # Files that don't exist (orphaned DB records)
    orphans_cleaned: int = 0  # Orphaned DB records removed
    bytes_migrated: int
    bytes_migrated_gb: float
    status: str = "completed"  # "completed", "in_progress", "error"
    message: str | None = None


# In-memory migration status tracking
_migration_status: dict = {
    "running": False,
    "files_migrated": 0,
    "files_failed": 0,
    "files_skipped": 0,
    "orphans_cleaned": 0,
    "bytes_migrated": 0,
    "message": None,
}

# Keep reference to background task to prevent garbage collection
_migration_task: asyncio.Task | None = None


def _get_tier_config_path() -> Path:
    """Get path to tier config JSON file."""
    settings = get_settings()
    return settings.storage_root / ".tier_config.json"


def _load_tier_config() -> dict:
    """Load tier config from file or return defaults from env config."""
    config_path = _get_tier_config_path()
    settings = get_settings()

    defaults = {
        "hot_max_gb": settings.hot_max_gb,
        "hot_retention_days": settings.hot_retention_days,
        "warm_max_gb": settings.warm_max_gb,
        "warm_retention_days": settings.warm_retention_days,
    }

    if config_path.exists():
        try:
            with open(config_path) as f:
                saved = json.load(f)
                defaults.update(saved)
        except (json.JSONDecodeError, IOError):
            pass

    return defaults


def _save_tier_config(data: dict) -> None:
    """Save tier config to file."""
    config_path = _get_tier_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)


@router.get("/tiers/config", response_model=TierConfigResponse)
async def get_tier_config(
    current_user: User = Depends(get_current_user),
) -> TierConfigResponse:
    """Get current tier storage configuration."""
    settings = get_settings()
    saved_config = _load_tier_config()

    # Check if S3 is configured
    s3_configured = bool(
        settings.s3_bucket_name
        and settings.s3_access_key
        and settings.s3_secret_key
    )

    return TierConfigResponse(
        hot_max_gb=saved_config.get("hot_max_gb"),
        hot_retention_days=saved_config.get("hot_retention_days"),
        warm_storage_enabled=settings.warm_storage_enabled,
        warm_storage_path=str(settings.warm_storage_path) if settings.warm_storage_path else None,
        warm_max_gb=saved_config.get("warm_max_gb"),
        warm_retention_days=saved_config.get("warm_retention_days"),
        cold_storage_enabled=settings.cold_storage_enabled,
        s3_endpoint_url=settings.s3_endpoint_url,
        s3_bucket_name=settings.s3_bucket_name,
        s3_region=settings.s3_region,
        s3_prefix=settings.s3_prefix,
        s3_configured=s3_configured,
        tier_migration_check_interval_minutes=settings.tier_migration_check_interval_minutes,
    )


@router.put("/tiers/config", response_model=TierConfigResponse)
async def update_tier_config(
    update: TierConfigUpdate,
    admin_user: User = Depends(get_admin_user),
) -> TierConfigResponse:
    """Update tier storage configuration.

    Note: Only threshold values can be updated via API. Storage paths and
    S3 credentials must be set via environment variables.
    """
    current = _load_tier_config()

    # Update provided fields
    if update.hot_max_gb is not None:
        current["hot_max_gb"] = update.hot_max_gb
    elif "hot_max_gb" in update.model_fields_set:
        current["hot_max_gb"] = None

    if update.hot_retention_days is not None:
        current["hot_retention_days"] = update.hot_retention_days
    elif "hot_retention_days" in update.model_fields_set:
        current["hot_retention_days"] = None

    if update.warm_max_gb is not None:
        current["warm_max_gb"] = update.warm_max_gb
    elif "warm_max_gb" in update.model_fields_set:
        current["warm_max_gb"] = None

    if update.warm_retention_days is not None:
        current["warm_retention_days"] = update.warm_retention_days
    elif "warm_retention_days" in update.model_fields_set:
        current["warm_retention_days"] = None

    _save_tier_config(current)

    # Return full config
    return await get_tier_config(admin_user)


@router.get("/tiers/stats", response_model=TierStatsResponse)
async def get_tier_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TierStatsResponse:
    """Get statistics for all storage tiers."""
    settings = get_settings()
    saved_config = _load_tier_config()

    async def get_stats_for_tier(tier: str) -> tuple[int, int, Optional[datetime], Optional[datetime]]:
        """Get stats for a specific tier."""
        size_query = (
            select(func.coalesce(func.sum(Recording.file_size), 0))
            .where(Recording.storage_tier == tier)
            .where(Recording.status == RecordingStatus.COMPLETED.value)
        )
        size_result = await db.execute(size_query)
        total_size = size_result.scalar() or 0

        count_query = (
            select(func.count())
            .where(Recording.storage_tier == tier)
            .where(Recording.status == RecordingStatus.COMPLETED.value)
        )
        count_result = await db.execute(count_query)
        file_count = count_result.scalar() or 0

        oldest_query = (
            select(func.min(Recording.start_time))
            .where(Recording.storage_tier == tier)
            .where(Recording.status == RecordingStatus.COMPLETED.value)
        )
        oldest_result = await db.execute(oldest_query)
        oldest = oldest_result.scalar()

        newest_query = (
            select(func.max(Recording.start_time))
            .where(Recording.storage_tier == tier)
            .where(Recording.status == RecordingStatus.COMPLETED.value)
        )
        newest_result = await db.execute(newest_query)
        newest = newest_result.scalar()

        return total_size, file_count, oldest, newest

    # Hot tier stats
    hot_size, hot_count, hot_oldest, hot_newest = await get_stats_for_tier(
        StorageTier.HOT.value
    )
    hot_max_gb = saved_config.get("hot_max_gb")
    hot_size_gb = float(hot_size) / (1024 ** 3)

    hot_stats = TierStats(
        enabled=True,
        total_size_gb=round(hot_size_gb, 2),
        file_count=hot_count,
        oldest_file=hot_oldest,
        newest_file=hot_newest,
        max_size_gb=hot_max_gb,
        retention_days=saved_config.get("hot_retention_days"),
        percent_full=round(hot_size_gb / float(hot_max_gb) * 100, 1) if hot_max_gb else None,
    )

    # Warm tier stats
    warm_stats = None
    if settings.warm_storage_enabled:
        warm_size, warm_count, warm_oldest, warm_newest = await get_stats_for_tier(
            StorageTier.WARM.value
        )
        warm_max_gb = saved_config.get("warm_max_gb")
        warm_size_gb = float(warm_size) / (1024 ** 3)

        warm_stats = TierStats(
            enabled=True,
            total_size_gb=round(warm_size_gb, 2),
            file_count=warm_count,
            oldest_file=warm_oldest,
            newest_file=warm_newest,
            max_size_gb=warm_max_gb,
            retention_days=saved_config.get("warm_retention_days"),
            percent_full=round(warm_size_gb / float(warm_max_gb) * 100, 1) if warm_max_gb else None,
        )

    # Cold tier stats
    cold_stats = None
    if settings.cold_storage_enabled:
        cold_size, cold_count, cold_oldest, cold_newest = await get_stats_for_tier(
            StorageTier.COLD.value
        )
        cold_size_gb = float(cold_size) / (1024 ** 3)

        cold_stats = TierStats(
            enabled=True,
            total_size_gb=round(cold_size_gb, 2),
            file_count=cold_count,
            oldest_file=cold_oldest,
            newest_file=cold_newest,
            max_size_gb=None,  # Cold has no limit
            retention_days=None,
            percent_full=None,
        )

    # Get actual filesystem usage (not just DB-tracked)
    fs_stats = retention_service.get_stats()

    return TierStatsResponse(
        hot=hot_stats,
        warm=warm_stats,
        cold=cold_stats,
        filesystem_total_gb=fs_stats.total_size_gb,
        filesystem_total_files=fs_stats.total_files,
    )


async def _run_migration_until_threshold(
    from_tier: str,
    target_tier: str,
    max_files_per_batch: int,
    clean_orphans: bool,
) -> None:
    """Background task to migrate files until tier is under threshold."""
    from app.services.tier_migration import tier_migration_service

    global _migration_status

    # Load config from saved file (not just env vars)
    tier_config = _load_tier_config()

    # Get threshold for source tier
    if from_tier == StorageTier.HOT.value:
        max_gb = tier_config.get("hot_max_gb")
        retention_days = tier_config.get("hot_retention_days")
    else:
        max_gb = tier_config.get("warm_max_gb")
        retention_days = tier_config.get("warm_retention_days")

    print(f"=== BACKGROUND MIGRATION STARTING: {from_tier} -> {target_tier} ===", flush=True)
    print(f"=== Thresholds: max_gb={max_gb}, retention_days={retention_days} ===", flush=True)
    logger.info(f"Migration thresholds: max_gb={max_gb}, retention_days={retention_days}")
    logger.info(f"Starting background migration: {from_tier} -> {target_tier}")

    _migration_status["running"] = True
    _migration_status["message"] = f"Migrating from {from_tier} to {target_tier}..."

    try:
        logger.info("Creating database session for migration...")
        async with async_session_maker() as db:
            logger.info("Database session created, starting migration loop")
            while True:
                # Check if we should continue migrating
                logger.info("Checking if migration should continue...")
                should_migrate = await tier_migration_service.should_migrate_from_tier(
                    db, from_tier, max_gb, retention_days
                )
                logger.info(f"should_migrate_from_tier returned: {should_migrate}")

                if not should_migrate:
                    _migration_status["message"] = "Migration complete - tier is under threshold"
                    logger.info("Migration complete - tier is under threshold")
                    break

                # Get next batch of candidates
                logger.info(f"Getting up to {max_files_per_batch} migration candidates...")
                candidates = await tier_migration_service.get_migration_candidates(
                    db, from_tier, max_files_per_batch
                )
                logger.info(f"Got {len(candidates)} candidates")

                if not candidates:
                    _migration_status["message"] = "Migration complete - no more files to migrate"
                    logger.info("No more candidates, migration complete")
                    break

                # Process this batch
                batch_had_real_files = False
                for recording in candidates:
                    source_path = Path(recording.file_path)
                    if recording.storage_tier == StorageTier.WARM.value and recording.storage_path:
                        source_path = Path(recording.storage_path)

                    if not source_path.exists():
                        _migration_status["files_skipped"] += 1
                        # Always handle orphaned records to prevent
                        # infinite loop re-fetching the same missing files.
                        # Mark as error so they're excluded from future
                        # candidate queries (which filter on COMPLETED).
                        try:
                            if clean_orphans:
                                await db.delete(recording)
                            else:
                                recording.status = RecordingStatus.ERROR.value
                                recording.error_message = (
                                    "File missing from disk during migration"
                                )
                            await db.commit()
                            _migration_status["orphans_cleaned"] += 1
                            logger.info(
                                f"Orphaned recording {recording.id}: "
                                f"file missing at {source_path}"
                            )
                        except Exception:
                            await db.rollback()
                        continue

                    batch_had_real_files = True
                    try:
                        if target_tier == StorageTier.WARM.value:
                            success = await tier_migration_service.migrate_to_warm(recording, db)
                        else:
                            success = await tier_migration_service.migrate_to_cold(recording, db)

                        if success:
                            _migration_status["files_migrated"] += 1
                            _migration_status["bytes_migrated"] += recording.file_size or 0
                        else:
                            _migration_status["files_failed"] += 1
                    except Exception as e:
                        _migration_status["files_failed"] += 1
                        logger.exception(f"Migration error: {e}")

                # Yield to event loop between batches
                await asyncio.sleep(0)

    except Exception as e:
        _migration_status["message"] = f"Migration error: {e}"
        logger.exception(f"Background migration error: {e}")
    finally:
        _migration_status["running"] = False


@router.get("/tiers/migrate/status", response_model=MigrationResult)
async def get_migration_status(
    admin_user: User = Depends(get_admin_user),
) -> MigrationResult:
    """Get the status of a running or completed migration."""
    return MigrationResult(
        files_migrated=_migration_status["files_migrated"],
        files_failed=_migration_status["files_failed"],
        files_skipped=_migration_status["files_skipped"],
        orphans_cleaned=_migration_status["orphans_cleaned"],
        bytes_migrated=_migration_status["bytes_migrated"],
        bytes_migrated_gb=round(_migration_status["bytes_migrated"] / (1024 ** 3), 2),
        status="in_progress" if _migration_status["running"] else "completed",
        message=_migration_status["message"],
    )


@router.post("/tiers/migrate", response_model=MigrationResult)
async def trigger_migration(
    request: MigrationTriggerRequest,
    admin_user: User = Depends(get_admin_user),
) -> MigrationResult:
    """Manually trigger migration from one tier to the next.

    Runs in background until tier is under threshold.
    Hot -> Warm (if warm enabled) or Hot -> Cold (if only cold enabled)
    Warm -> Cold
    """
    print(f"=== MIGRATE ENDPOINT CALLED: from_tier={request.from_tier} ===", flush=True)
    logger.info(f"Migrate endpoint called: from_tier={request.from_tier}")

    global _migration_status
    settings = get_settings()

    # Check if migration is already running
    if _migration_status["running"]:
        return MigrationResult(
            files_migrated=_migration_status["files_migrated"],
            files_failed=_migration_status["files_failed"],
            files_skipped=_migration_status["files_skipped"],
            orphans_cleaned=_migration_status["orphans_cleaned"],
            bytes_migrated=_migration_status["bytes_migrated"],
            bytes_migrated_gb=round(_migration_status["bytes_migrated"] / (1024 ** 3), 2),
            status="in_progress",
            message="Migration already in progress",
        )

    # Validate source tier
    if request.from_tier not in [StorageTier.HOT.value, StorageTier.WARM.value]:
        raise HTTPException(
            status_code=400,
            detail="from_tier must be 'hot' or 'warm'",
        )

    # Determine target tier
    if request.from_tier == StorageTier.HOT.value:
        if settings.warm_storage_enabled:
            target_tier = StorageTier.WARM.value
        elif settings.cold_storage_enabled:
            target_tier = StorageTier.COLD.value
        else:
            raise HTTPException(
                status_code=400,
                detail="No target tier enabled (warm or cold storage not configured)",
            )
    else:  # from warm
        if not settings.cold_storage_enabled:
            raise HTTPException(
                status_code=400,
                detail="Cold storage not enabled",
            )
        target_tier = StorageTier.COLD.value

    # Reset status and start background migration
    _migration_status = {
        "running": True,
        "files_migrated": 0,
        "files_failed": 0,
        "files_skipped": 0,
        "orphans_cleaned": 0,
        "bytes_migrated": 0,
        "message": "Starting migration...",
    }

    # Use asyncio.create_task for proper async execution
    global _migration_task
    _migration_task = asyncio.create_task(
        _run_migration_until_threshold(
            request.from_tier,
            target_tier,
            request.max_files,
            request.clean_orphans,
        )
    )

    return MigrationResult(
        files_migrated=0,
        files_failed=0,
        files_skipped=0,
        orphans_cleaned=0,
        bytes_migrated=0,
        bytes_migrated_gb=0.0,
        status="in_progress",
        message=f"Migration started in background: {request.from_tier} -> {target_tier}",
    )


# ============================================================
# Offline Export API
# ============================================================


class OfflineExportRequest(BaseModel):
    """Request to create an offline export."""

    camera_ids: list[int] = Field(..., description="Camera IDs to export")
    start_time: datetime = Field(..., description="Start of time range")
    end_time: datetime = Field(..., description="End of time range")
    output_path: str = Field(..., description="Output path (mounted filesystem)")
    include_detections: bool = Field(True, description="Include detection events")
    include_snapshots: bool = Field(True, description="Include detection snapshots")
    delete_after_copy: bool = Field(
        False, description="Delete source files after successful copy"
    )


class OfflineExportResponse(BaseModel):
    """Response for offline export creation."""

    export_id: str
    success: bool
    output_path: str
    files_exported: int
    bytes_exported: int
    bytes_exported_gb: float
    events_exported: int
    snapshots_exported: int
    error_message: Optional[str] = None
    manifest_path: Optional[str] = None


class ExportProgressResponse(BaseModel):
    """Response for export progress."""

    export_id: str
    status: str
    total_files: int
    files_copied: int
    total_bytes: int
    bytes_copied: int
    percent_complete: float
    current_file: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@router.post("/export/offline", response_model=OfflineExportResponse)
async def create_offline_export(
    request: OfflineExportRequest,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(get_admin_user),
) -> OfflineExportResponse:
    """Create an offline export of recordings to removable media.

    Exports recordings and detection events to a specified path, creating
    a structured directory with video files, detection logs, and snapshots.
    """
    from app.services.offline_export import offline_export_service

    # Validate output path
    output_path = Path(request.output_path)
    if not output_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Output path does not exist: {request.output_path}",
        )

    if not output_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Output path is not a directory: {request.output_path}",
        )

    # Validate time range
    if request.end_time <= request.start_time:
        raise HTTPException(
            status_code=400,
            detail="end_time must be after start_time",
        )

    # Perform export
    result = await offline_export_service.create_export(
        camera_ids=request.camera_ids,
        start_time=request.start_time,
        end_time=request.end_time,
        output_path=output_path,
        include_detections=request.include_detections,
        include_snapshots=request.include_snapshots,
        delete_after_copy=request.delete_after_copy,
        db=db,
    )

    return OfflineExportResponse(
        export_id=result.export_id,
        success=result.success,
        output_path=result.output_path,
        files_exported=result.files_exported,
        bytes_exported=result.bytes_exported,
        bytes_exported_gb=round(result.bytes_exported / (1024 ** 3), 2),
        events_exported=result.events_exported,
        snapshots_exported=result.snapshots_exported,
        error_message=result.error_message,
        manifest_path=result.manifest_path,
    )


@router.get("/export/offline/{export_id}/status", response_model=ExportProgressResponse)
async def get_export_status(
    export_id: str,
    current_user: User = Depends(get_current_user),
) -> ExportProgressResponse:
    """Get the status of an offline export."""
    from app.services.offline_export import offline_export_service

    progress = offline_export_service.get_export_progress(export_id)
    if not progress:
        raise HTTPException(
            status_code=404,
            detail=f"Export {export_id} not found",
        )

    percent_complete = 0.0
    if progress.total_bytes > 0:
        percent_complete = round(progress.bytes_copied / progress.total_bytes * 100, 1)

    return ExportProgressResponse(
        export_id=progress.export_id,
        status=progress.status,
        total_files=progress.total_files,
        files_copied=progress.files_copied,
        total_bytes=progress.total_bytes,
        bytes_copied=progress.bytes_copied,
        percent_complete=percent_complete,
        current_file=progress.current_file,
        error_message=progress.error_message,
        started_at=progress.started_at,
        completed_at=progress.completed_at,
    )


@router.post("/export/offline/{export_id}/cancel")
async def cancel_export(
    export_id: str,
    admin_user: User = Depends(get_admin_user),
) -> dict:
    """Cancel a running offline export."""
    from app.services.offline_export import offline_export_service

    success = offline_export_service.cancel_export(export_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Export {export_id} is not running or not found",
        )

    return {"success": True, "message": f"Export {export_id} cancelled"}
