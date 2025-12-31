"""Storage retention management service."""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class StorageStats:
    """Storage statistics."""

    total_size_bytes: int
    total_files: int
    oldest_file: Optional[datetime]
    newest_file: Optional[datetime]
    cameras: dict[str, dict]

    @property
    def total_size_gb(self) -> float:
        """Get total size in GB."""
        return self.total_size_bytes / (1024 ** 3)

    @property
    def total_size_mb(self) -> float:
        """Get total size in MB."""
        return self.total_size_bytes / (1024 ** 2)


@dataclass
class FileInfo:
    """Information about a recording file."""

    path: Path
    size: int
    mtime: datetime
    camera_name: str


class RetentionService:
    """Service for managing storage retention."""

    def __init__(self, storage_root: Optional[Path] = None):
        self.storage_root = storage_root or Path(settings.storage_root)

    def scan_storage(self) -> tuple[list[FileInfo], StorageStats]:
        """Scan storage directory and return file list and stats."""
        files: list[FileInfo] = []
        camera_stats: dict[str, dict] = {}

        if not self.storage_root.exists():
            return files, StorageStats(
                total_size_bytes=0,
                total_files=0,
                oldest_file=None,
                newest_file=None,
                cameras={},
            )

        # Scan all camera directories
        for camera_dir in self.storage_root.iterdir():
            if not camera_dir.is_dir():
                continue

            camera_name = camera_dir.name
            camera_size = 0
            camera_files = 0

            # Scan date directories
            for date_dir in camera_dir.iterdir():
                if not date_dir.is_dir():
                    continue

                # Scan video files
                for video_file in date_dir.glob("*.mp4"):
                    try:
                        stat = video_file.stat()
                        # Use UTC for consistent timezone handling
                        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                        size = stat.st_size

                        files.append(FileInfo(
                            path=video_file,
                            size=size,
                            mtime=mtime,
                            camera_name=camera_name,
                        ))

                        camera_size += size
                        camera_files += 1
                    except OSError:
                        continue

            if camera_files > 0:
                camera_stats[camera_name] = {
                    "size_bytes": camera_size,
                    "size_gb": camera_size / (1024 ** 3),
                    "file_count": camera_files,
                }

        # Sort files by modification time (oldest first)
        files.sort(key=lambda f: f.mtime)

        total_size = sum(f.size for f in files)
        oldest = files[0].mtime if files else None
        newest = files[-1].mtime if files else None

        stats = StorageStats(
            total_size_bytes=total_size,
            total_files=len(files),
            oldest_file=oldest,
            newest_file=newest,
            cameras=camera_stats,
        )

        return files, stats

    def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        _, stats = self.scan_storage()
        return stats

    def get_files_to_delete(
        self,
        files: list[FileInfo],
        retention_days: Optional[int] = None,
        max_size_gb: Optional[float] = None,
    ) -> list[FileInfo]:
        """Determine which files should be deleted based on retention policy."""
        to_delete: list[FileInfo] = []

        if retention_days is None:
            retention_days = settings.retention_days
        if max_size_gb is None:
            max_size_gb = settings.retention_max_gb

        # First pass: delete files older than retention_days
        if retention_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
            for f in files:
                if f.mtime < cutoff:
                    to_delete.append(f)

        # Second pass: delete oldest files if over size limit
        if max_size_gb is not None:
            max_size_bytes = max_size_gb * (1024 ** 3)
            current_size = sum(f.size for f in files if f not in to_delete)

            # Sort remaining files by age
            remaining = [f for f in files if f not in to_delete]

            for f in remaining:
                if current_size <= max_size_bytes:
                    break
                to_delete.append(f)
                current_size -= f.size

        return to_delete

    def delete_files(self, files: list[FileInfo]) -> tuple[int, int]:
        """Delete files and return (deleted_count, freed_bytes)."""
        deleted_count = 0
        freed_bytes = 0

        for f in files:
            try:
                f.path.unlink()
                deleted_count += 1
                freed_bytes += f.size
                logger.info(f"Deleted: {f.path}")

                # Remove empty date directories
                self._cleanup_empty_dirs(f.path.parent)
            except OSError as e:
                logger.error(f"Failed to delete {f.path}: {e}")

        return deleted_count, freed_bytes

    def _cleanup_empty_dirs(self, dir_path: Path) -> None:
        """Remove empty directories up to storage root."""
        try:
            while dir_path != self.storage_root and dir_path.exists():
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    logger.debug(f"Removed empty directory: {dir_path}")
                    dir_path = dir_path.parent
                else:
                    break
        except OSError:
            pass

    def enforce_retention(self) -> dict:
        """Enforce retention policy and return summary."""
        files, stats = self.scan_storage()

        if not files:
            return {
                "files_scanned": 0,
                "files_deleted": 0,
                "bytes_freed": 0,
                "storage_before_gb": 0,
                "storage_after_gb": 0,
            }

        to_delete = self.get_files_to_delete(files)

        if to_delete:
            logger.info(f"Retention policy: deleting {len(to_delete)} files")
            deleted_count, freed_bytes = self.delete_files(to_delete)
        else:
            deleted_count, freed_bytes = 0, 0

        return {
            "files_scanned": len(files),
            "files_deleted": deleted_count,
            "bytes_freed": freed_bytes,
            "gb_freed": freed_bytes / (1024 ** 3),
            "storage_before_gb": stats.total_size_gb,
            "storage_after_gb": stats.total_size_gb - (freed_bytes / (1024 ** 3)),
        }


class RetentionMonitor:
    """Background service for periodic retention enforcement."""

    def __init__(self, check_interval_minutes: Optional[int] = None):
        self.check_interval = (
            check_interval_minutes or settings.retention_check_interval_minutes
        ) * 60  # Convert to seconds
        self.service = RetentionService()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the background retention task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"Retention monitor started (interval: {self.check_interval // 60} min)"
        )

    async def stop(self) -> None:
        """Stop the background retention task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Retention monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Run retention in thread pool to avoid blocking
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.service.enforce_retention
                )

                if result["files_deleted"] > 0:
                    logger.info(
                        f"Retention cleanup: deleted {result['files_deleted']} files, "
                        f"freed {result['gb_freed']:.2f} GB"
                    )
                else:
                    logger.debug("Retention check: no files to delete")

            except Exception:
                logger.exception("Error in retention check")

            try:
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break


# Global instances
retention_service = RetentionService()
retention_monitor = RetentionMonitor()
