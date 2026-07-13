"""Tier migration service for moving recordings between storage tiers."""

import asyncio
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import async_session_maker
from app.models.recording import Recording, RecordingStatus, StorageTier
from app.services.s3_storage import S3StorageClient, get_s3_client

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class MigrationStats:
    """Statistics for a migration run."""

    files_migrated: int = 0
    files_failed: int = 0
    bytes_migrated: int = 0
    bytes_freed: int = 0


class TierMigrationService:
    """Service for migrating recordings between storage tiers.

    Tiers:
    - hot: Primary active storage where recordings are written
    - warm: Secondary local storage when hot is full (optional)
    - cold: S3-compatible remote storage when warm is full (optional)
    """

    def __init__(
        self,
        hot_storage_path: Optional[Path] = None,
        warm_storage_path: Optional[Path] = None,
        s3_client: Optional[S3StorageClient] = None,
    ):
        self.hot_storage_path = hot_storage_path or Path(settings.storage_root)
        self.warm_storage_path = warm_storage_path or settings.warm_storage_path
        self.s3_client = s3_client

    def _get_s3_client(self) -> S3StorageClient:
        """Get S3 client, creating if needed."""
        if self.s3_client is None:
            self.s3_client = get_s3_client()
        return self.s3_client

    def _disk_free_bytes(self, path: Optional[Path]) -> Optional[int]:
        """Return free bytes on the filesystem holding ``path``.

        Returns None if the path is unset or cannot be stat'd, so callers can
        distinguish "unknown" from "zero free".
        """
        if not path:
            return None
        try:
            return shutil.disk_usage(path).free
        except OSError:
            return None

    def _get_relative_path(self, file_path: str, storage_root: Path) -> str:
        """Get the relative path from a storage root."""
        try:
            return str(Path(file_path).relative_to(storage_root))
        except ValueError:
            # Already relative or from different root
            return file_path

    def _generate_s3_key(self, recording: Recording) -> str:
        """Generate S3 key for a recording.

        Uses format: camera_name/date/filename.mp4
        """
        rel_path = self._get_relative_path(recording.file_path, self.hot_storage_path)
        return rel_path

    async def migrate_to_warm(
        self,
        recording: Recording,
        db: AsyncSession,
    ) -> bool:
        """Migrate a recording from hot to warm storage.

        Args:
            recording: Recording to migrate
            db: Database session

        Returns:
            True if migration successful, False otherwise
        """
        if not settings.warm_storage_enabled or not self.warm_storage_path:
            logger.warning("Warm storage is not enabled")
            return False

        if recording.storage_tier != StorageTier.HOT.value:
            logger.warning(
                f"Recording {recording.id} is not in hot storage "
                f"(tier={recording.storage_tier})"
            )
            return False

        source_path = Path(recording.file_path)
        if not source_path.exists():
            logger.error(f"Source file does not exist: {source_path}")
            recording.status = "error"
            recording.error_message = "File missing from disk during migration"
            await db.commit()
            return False

        # Calculate destination path (maintain same structure)
        rel_path = self._get_relative_path(recording.file_path, self.hot_storage_path)
        dest_path = self.warm_storage_path / rel_path

        # Clear enough old data from warm to fit this file before copying, so
        # the copy can't fail for lack of space (and hot keeps draining).
        needed_bytes = source_path.stat().st_size
        await self.ensure_warm_room(db, needed_bytes)

        try:
            # Ensure destination directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy file to warm storage
            logger.info(f"Migrating {source_path} to warm storage: {dest_path}")
            shutil.copy2(source_path, dest_path)

            # Verify copy by checking size
            if dest_path.stat().st_size != source_path.stat().st_size:
                logger.error(f"Size mismatch after copy: {source_path}")
                dest_path.unlink()
                return False

            # Update database record
            recording.storage_tier = StorageTier.WARM.value
            recording.storage_path = str(dest_path)
            recording.migrated_at = datetime.now(timezone.utc)
            await db.commit()

            # Delete from hot storage
            source_path.unlink()
            self._cleanup_empty_dirs(source_path.parent, self.hot_storage_path)

            logger.info(
                f"Successfully migrated recording {recording.id} to warm storage"
            )
            return True

        except Exception as e:
            logger.exception(f"Failed to migrate recording {recording.id} to warm: {e}")
            await db.rollback()
            # Cleanup partial copy if it exists
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except OSError:
                    pass
            return False

    async def migrate_to_cold(
        self,
        recording: Recording,
        db: AsyncSession,
    ) -> bool:
        """Migrate a recording from warm (or hot) to cold storage (S3).

        Args:
            recording: Recording to migrate
            db: Database session

        Returns:
            True if migration successful, False otherwise
        """
        if not settings.cold_storage_enabled:
            logger.warning("Cold storage is not enabled")
            return False

        s3 = self._get_s3_client()
        if not s3.is_configured():
            logger.error("S3 client is not properly configured")
            return False

        # Determine source path based on current tier
        if recording.storage_tier == StorageTier.HOT.value:
            source_path = Path(recording.file_path)
            storage_root = self.hot_storage_path
        elif recording.storage_tier == StorageTier.WARM.value:
            source_path = Path(recording.storage_path or recording.file_path)
            storage_root = self.warm_storage_path or self.hot_storage_path
        else:
            logger.warning(
                f"Recording {recording.id} is already in cold storage"
            )
            return False

        if not source_path.exists():
            logger.error(f"Source file does not exist: {source_path}")
            recording.status = "error"
            recording.error_message = "File missing from disk during migration"
            await db.commit()
            return False

        # Generate S3 key
        s3_key = self._generate_s3_key(recording)

        try:
            # Upload to S3
            logger.info(f"Migrating {source_path} to cold storage: {s3_key}")
            full_key = s3.upload_file(source_path, s3_key)

            # Verify upload by checking size
            s3_size = s3.get_object_size(s3_key)
            local_size = source_path.stat().st_size

            if s3_size != local_size:
                logger.error(
                    f"Size mismatch after S3 upload: local={local_size}, s3={s3_size}"
                )
                s3.delete_file(s3_key)
                return False

            # Update database record
            recording.storage_tier = StorageTier.COLD.value
            recording.storage_path = full_key
            recording.migrated_at = datetime.now(timezone.utc)
            await db.commit()

            # Delete from source storage
            source_path.unlink()
            self._cleanup_empty_dirs(source_path.parent, storage_root)

            logger.info(
                f"Successfully migrated recording {recording.id} to cold storage"
            )
            return True

        except Exception as e:
            logger.exception(f"Failed to migrate recording {recording.id} to cold: {e}")
            await db.rollback()
            return False

    def _cleanup_empty_dirs(self, dir_path: Path, storage_root: Path) -> None:
        """Remove empty directories up to storage root."""
        try:
            while dir_path != storage_root and dir_path.exists():
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    logger.debug(f"Removed empty directory: {dir_path}")
                    dir_path = dir_path.parent
                else:
                    break
        except OSError:
            pass

    async def get_migration_candidates(
        self,
        db: AsyncSession,
        from_tier: str,
        max_count: int = 100,
    ) -> list[Recording]:
        """Get recordings that are candidates for migration.

        Returns recordings ordered by start_time (oldest first).

        Args:
            db: Database session
            from_tier: Source tier to migrate from
            max_count: Maximum number of candidates to return

        Returns:
            List of Recording objects
        """
        query = (
            select(Recording)
            .where(Recording.storage_tier == from_tier)
            .where(Recording.status == RecordingStatus.COMPLETED.value)
            .order_by(Recording.start_time.asc())
            .limit(max_count)
        )

        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_tier_size(self, db: AsyncSession, tier: str) -> int:
        """Get total size of recordings in a tier.

        Args:
            db: Database session
            tier: Storage tier to check

        Returns:
            Total size in bytes
        """
        from sqlalchemy import func

        query = (
            select(func.coalesce(func.sum(Recording.file_size), 0))
            .where(Recording.storage_tier == tier)
            .where(Recording.status == RecordingStatus.COMPLETED.value)
        )

        result = await db.execute(query)
        return result.scalar() or 0

    async def get_oldest_recording_time(
        self,
        db: AsyncSession,
        tier: str,
    ) -> Optional[datetime]:
        """Get the start time of the oldest recording in a tier.

        Args:
            db: Database session
            tier: Storage tier to check

        Returns:
            Start time of oldest recording, or None if tier is empty
        """
        from sqlalchemy import func

        query = (
            select(func.min(Recording.start_time))
            .where(Recording.storage_tier == tier)
            .where(Recording.status == RecordingStatus.COMPLETED.value)
        )

        result = await db.execute(query)
        return result.scalar()

    async def should_migrate_from_tier(
        self,
        db: AsyncSession,
        tier: str,
        max_gb: Optional[float],
        retention_days: Optional[int],
        min_free_gb: Optional[float] = None,
        disk_path: Optional[Path] = None,
    ) -> bool:
        """Check if migration from a tier should be triggered.

        Migration fires when ANY configured limit is exceeded: the tier's
        DB-tracked size passes ``max_gb``, the filesystem free space at
        ``disk_path`` drops below ``min_free_gb``, or the oldest file is older
        than ``retention_days``.

        Args:
            db: Database session
            tier: Source tier to check
            max_gb: Maximum size in GB for this tier
            retention_days: Maximum age in days for files in this tier
            min_free_gb: Minimum filesystem free space to keep at ``disk_path``
            disk_path: Filesystem path backing this tier (for the free check)

        Returns:
            True if migration should be triggered
        """
        # Check size threshold
        if max_gb is not None:
            tier_size = await self.get_tier_size(db, tier)
            tier_size_gb = tier_size / (1024 ** 3)
            if tier_size_gb > max_gb:
                logger.info(
                    f"Tier {tier} exceeds size limit: {tier_size_gb:.2f}GB > {max_gb}GB"
                )
                return True

        # Check filesystem free-space threshold
        if min_free_gb is not None:
            free = self._disk_free_bytes(disk_path)
            if free is not None:
                free_gb = free / (1024 ** 3)
                if free_gb < min_free_gb:
                    logger.info(
                        f"Tier {tier} disk low on space: {free_gb:.2f}GB free "
                        f"< {min_free_gb}GB"
                    )
                    return True

        # Check retention threshold
        if retention_days is not None:
            oldest_time = await self.get_oldest_recording_time(db, tier)
            if oldest_time:
                cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
                if oldest_time < cutoff:
                    logger.info(
                        f"Tier {tier} has files older than {retention_days} days"
                    )
                    return True

        return False

    async def _evict_one_from_warm(
        self,
        recording: Recording,
        db: AsyncSession,
    ) -> int:
        """Free a single warm recording, returning bytes reclaimed from warm.

        Prefers pushing to cold storage (no data loss) when cold is enabled;
        otherwise permanently deletes the file and its DB row. Returns 0 if
        nothing could be reclaimed.
        """
        size = recording.file_size or 0

        if settings.cold_storage_enabled:
            if await self.migrate_to_cold(recording, db):
                return size
            return 0

        warm_path = Path(recording.storage_path or recording.file_path)
        try:
            reclaimed = size
            if warm_path.exists():
                reclaimed = warm_path.stat().st_size
                warm_path.unlink()
                if self.warm_storage_path:
                    self._cleanup_empty_dirs(warm_path.parent, self.warm_storage_path)
            else:
                logger.warning(
                    f"Warm file missing during eviction: {warm_path} "
                    f"(recording {recording.id})"
                )
            await db.delete(recording)
            await db.commit()
            logger.info(
                f"Evicted warm recording {recording.id} "
                f"({reclaimed / (1024**2):.1f} MB)"
            )
            return reclaimed
        except Exception:
            await db.rollback()
            logger.exception(
                f"Failed to evict warm recording {recording.id}"
            )
            return 0

    async def _evict_warm_while(
        self,
        db: AsyncSession,
        should_continue,
        reason: str,
    ) -> int:
        """Evict oldest warm recordings while ``should_continue()`` is true.

        ``should_continue`` is an async predicate re-checked before each
        recording. Terminates when the predicate is satisfied, warm has no more
        evictable recordings, or a full batch reclaims nothing (guards against
        an infinite loop on undeletable files). Returns total bytes freed.
        """
        freed = 0
        while await should_continue():
            candidates = await self.get_migration_candidates(
                db, StorageTier.WARM.value, max_count=50
            )
            if not candidates:
                logger.warning(
                    f"Warm needs freeing ({reason}) but no evictable "
                    f"recordings remain"
                )
                break

            progressed = False
            for recording in candidates:
                reclaimed = await self._evict_one_from_warm(recording, db)
                if reclaimed > 0:
                    freed += reclaimed
                    progressed = True
                    if not await should_continue():
                        break

            if not progressed:
                logger.warning(
                    f"Warm eviction ({reason}) made no progress; stopping"
                )
                break

        return freed

    async def ensure_warm_room(
        self,
        db: AsyncSession,
        needed_bytes: int,
    ) -> int:
        """Clear enough old warm data to fit ``needed_bytes`` before a copy.

        Frees warm until it has at least ``needed_bytes`` plus the configured
        headroom of free filesystem space. No-op if warm's free space can't be
        determined. Returns bytes freed.
        """
        if not self.warm_storage_path:
            return 0

        headroom = int(settings.tier_migration_headroom_gb * (1024 ** 3))
        target_free = needed_bytes + headroom

        async def below_target() -> bool:
            free = self._disk_free_bytes(self.warm_storage_path)
            return free is not None and free < target_free

        return await self._evict_warm_while(db, below_target, "make room for hot")

    async def maintain_warm(self, db: AsyncSession) -> int:
        """Enforce warm's standing size cap and free-space floor.

        Runs independently of hot migration so warm never grows unbounded when
        cold storage is disabled. Returns bytes freed.
        """
        if not settings.warm_storage_enabled or not self.warm_storage_path:
            return 0

        freed = 0

        # Enforce the DB-size cap (oldest first).
        if settings.warm_max_gb is not None:
            async def over_cap() -> bool:
                size_gb = (
                    await self.get_tier_size(db, StorageTier.WARM.value)
                    / (1024 ** 3)
                )
                return size_gb > settings.warm_max_gb

            freed += await self._evict_warm_while(db, over_cap, "over size cap")

        # Enforce the filesystem free-space floor.
        if settings.warm_min_free_gb is not None:
            floor = settings.warm_min_free_gb * (1024 ** 3)

            async def below_floor() -> bool:
                free = self._disk_free_bytes(self.warm_storage_path)
                return free is not None and free < floor

            freed += await self._evict_warm_while(db, below_floor, "below free floor")

        return freed


class TierMigrationMonitor:
    """Background service for periodic tier migration."""

    def __init__(
        self,
        check_interval_minutes: Optional[int] = None,
        migration_service: Optional[TierMigrationService] = None,
    ):
        self.check_interval = (
            check_interval_minutes or settings.tier_migration_check_interval_minutes
        ) * 60  # Convert to seconds
        self.migration_service = migration_service or TierMigrationService()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the background migration task."""
        if self._running:
            return

        # Only start if tiered storage is configured
        if not settings.warm_storage_enabled and not settings.cold_storage_enabled:
            logger.info("Tiered storage not enabled, skipping migration monitor")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"Tier migration monitor started (interval: {self.check_interval // 60} min)"
        )

    async def stop(self) -> None:
        """Stop the background migration task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Tier migration monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_and_migrate()
            except Exception:
                logger.exception("Error in tier migration check")

            try:
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break

    async def _check_and_migrate(self) -> None:
        """Check if migration is needed and perform it."""
        async with async_session_maker() as db:
            stats = MigrationStats()

            svc = self.migration_service

            # Keep warm within its cap / free-space floor first, so there's
            # room for anything we're about to migrate in from hot.
            if settings.warm_storage_enabled:
                await svc.maintain_warm(db)

            # Check hot -> warm migration
            if settings.warm_storage_enabled:
                should_migrate = await svc.should_migrate_from_tier(
                    db,
                    StorageTier.HOT.value,
                    settings.hot_max_gb,
                    settings.hot_retention_days,
                    settings.hot_min_free_gb,
                    svc.hot_storage_path,
                )

                if should_migrate:
                    candidates = await svc.get_migration_candidates(
                        db, StorageTier.HOT.value
                    )

                    for recording in candidates:
                        if await svc.migrate_to_warm(recording, db):
                            stats.files_migrated += 1
                            stats.bytes_migrated += recording.file_size or 0
                        else:
                            stats.files_failed += 1

                        # Re-check if we still need to migrate
                        should_continue = await svc.should_migrate_from_tier(
                            db,
                            StorageTier.HOT.value,
                            settings.hot_max_gb,
                            settings.hot_retention_days,
                            settings.hot_min_free_gb,
                            svc.hot_storage_path,
                        )
                        if not should_continue:
                            break

            # Check warm -> cold migration
            if settings.cold_storage_enabled:
                should_migrate = await self.migration_service.should_migrate_from_tier(
                    db,
                    StorageTier.WARM.value,
                    settings.warm_max_gb,
                    settings.warm_retention_days,
                )

                if should_migrate:
                    candidates = await self.migration_service.get_migration_candidates(
                        db, StorageTier.WARM.value
                    )

                    for recording in candidates:
                        if await self.migration_service.migrate_to_cold(recording, db):
                            stats.files_migrated += 1
                            stats.bytes_migrated += recording.file_size or 0
                        else:
                            stats.files_failed += 1

                        # Re-check if we still need to migrate
                        should_continue = (
                            await self.migration_service.should_migrate_from_tier(
                                db,
                                StorageTier.WARM.value,
                                settings.warm_max_gb,
                                settings.warm_retention_days,
                            )
                        )
                        if not should_continue:
                            break

            # Also check hot -> cold if warm is not enabled
            if settings.cold_storage_enabled and not settings.warm_storage_enabled:
                should_migrate = await svc.should_migrate_from_tier(
                    db,
                    StorageTier.HOT.value,
                    settings.hot_max_gb,
                    settings.hot_retention_days,
                    settings.hot_min_free_gb,
                    svc.hot_storage_path,
                )

                if should_migrate:
                    candidates = await svc.get_migration_candidates(
                        db, StorageTier.HOT.value
                    )

                    for recording in candidates:
                        if await svc.migrate_to_cold(recording, db):
                            stats.files_migrated += 1
                            stats.bytes_migrated += recording.file_size or 0
                        else:
                            stats.files_failed += 1

                        # Re-check
                        should_continue = await svc.should_migrate_from_tier(
                            db,
                            StorageTier.HOT.value,
                            settings.hot_max_gb,
                            settings.hot_retention_days,
                            settings.hot_min_free_gb,
                            svc.hot_storage_path,
                        )
                        if not should_continue:
                            break

            if stats.files_migrated > 0:
                logger.info(
                    f"Tier migration: migrated {stats.files_migrated} files "
                    f"({stats.bytes_migrated / (1024**3):.2f} GB), "
                    f"failed: {stats.files_failed}"
                )
            else:
                logger.debug("Tier migration check: no migration needed")


# Global instances
tier_migration_service = TierMigrationService()
tier_migration_monitor = TierMigrationMonitor(migration_service=tier_migration_service)
