"""Offline export service for exporting recordings to removable media."""

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.database import async_session_maker
from app.models.camera import Camera
from app.models.object_event import ObjectEvent
from app.models.recording import Recording, RecordingStatus, StorageTier

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ExportProgress:
    """Tracks export progress."""

    export_id: str
    status: str = "pending"  # pending, running, completed, failed, cancelled
    total_files: int = 0
    files_copied: int = 0
    total_bytes: int = 0
    bytes_copied: int = 0
    current_file: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class ExportResult:
    """Result of an export operation."""

    export_id: str
    success: bool
    output_path: str
    files_exported: int
    bytes_exported: int
    events_exported: int
    snapshots_exported: int
    error_message: Optional[str] = None
    manifest_path: Optional[str] = None


# In-memory storage for export progress (could be Redis in production)
_export_progress: dict[str, ExportProgress] = {}


class OfflineExportService:
    """Service for exporting recordings with detection metadata to removable media.

    Creates an export directory with the following structure:
    /mnt/usb/ronin-export-YYYYMMDD-HHMMSS/
    +-- manifest.json
    +-- cameras/
    |   +-- {camera_name}/{date}/{time}.mp4
    +-- detections/
        +-- events.json          # Machine-readable
        +-- events.txt           # Human-readable log
        +-- snapshots/           # Detection snapshots
    """

    def __init__(
        self,
        hot_storage_path: Optional[Path] = None,
        warm_storage_path: Optional[Path] = None,
    ):
        self.hot_storage_path = hot_storage_path or Path(settings.storage_root)
        self.warm_storage_path = warm_storage_path or settings.warm_storage_path

    def _get_recording_source_path(self, recording: Recording) -> Optional[Path]:
        """Get the source path for a recording based on its tier."""
        if recording.storage_tier == StorageTier.HOT.value:
            return Path(recording.file_path)
        elif recording.storage_tier == StorageTier.WARM.value:
            if recording.storage_path:
                return Path(recording.storage_path)
            return Path(recording.file_path)
        elif recording.storage_tier == StorageTier.COLD.value:
            # Cold storage requires download first - not supported in this version
            logger.warning(
                f"Recording {recording.id} is in cold storage, skipping"
            )
            return None
        return Path(recording.file_path)

    async def create_export(
        self,
        camera_ids: list[int],
        start_time: datetime,
        end_time: datetime,
        output_path: Path,
        include_detections: bool = True,
        include_snapshots: bool = True,
        delete_after_copy: bool = False,
        db: Optional[AsyncSession] = None,
    ) -> ExportResult:
        """Create an offline export of recordings and detections.

        Args:
            camera_ids: List of camera IDs to export
            start_time: Start of time range
            end_time: End of time range
            output_path: Base output path (will create subdirectory)
            include_detections: Include detection events
            include_snapshots: Include detection snapshots
            delete_after_copy: Delete source files after successful copy
            db: Database session (creates one if not provided)

        Returns:
            ExportResult with details of the export
        """
        export_id = str(uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        export_dir = output_path / f"ronin-export-{timestamp}"

        # Initialize progress
        progress = ExportProgress(
            export_id=export_id,
            status="running",
            started_at=datetime.now(timezone.utc),
        )
        _export_progress[export_id] = progress

        try:
            # Create export directory structure
            export_dir.mkdir(parents=True, exist_ok=True)
            cameras_dir = export_dir / "cameras"
            cameras_dir.mkdir(exist_ok=True)

            if include_detections:
                detections_dir = export_dir / "detections"
                detections_dir.mkdir(exist_ok=True)
                if include_snapshots:
                    snapshots_dir = detections_dir / "snapshots"
                    snapshots_dir.mkdir(exist_ok=True)

            # Use provided session or create new one
            close_session = False
            if db is None:
                db = async_session_maker()
                close_session = True

            try:
                # Get cameras
                camera_query = select(Camera).where(Camera.id.in_(camera_ids))
                camera_result = await db.execute(camera_query)
                cameras = {c.id: c for c in camera_result.scalars().all()}

                if not cameras:
                    raise ValueError("No cameras found for the given IDs")

                # Get recordings in the time range
                recordings_query = (
                    select(Recording)
                    .where(Recording.camera_id.in_(camera_ids))
                    .where(Recording.status == RecordingStatus.COMPLETED.value)
                    .where(Recording.start_time >= start_time)
                    .where(Recording.start_time <= end_time)
                    .order_by(Recording.start_time)
                )
                recordings_result = await db.execute(recordings_query)
                recordings = list(recordings_result.scalars().all())

                progress.total_files = len(recordings)
                progress.total_bytes = sum(r.file_size or 0 for r in recordings)

                # Copy recordings
                files_exported = 0
                bytes_exported = 0
                deleted_files: list[Path] = []

                for recording in recordings:
                    source_path = self._get_recording_source_path(recording)
                    if not source_path or not source_path.exists():
                        logger.warning(f"Recording file not found: {source_path}")
                        continue

                    camera = cameras.get(recording.camera_id)
                    if not camera:
                        continue

                    # Create destination path
                    date_str = recording.start_time.strftime("%Y-%m-%d")
                    time_str = recording.start_time.strftime("%H-%M-%S")
                    dest_dir = cameras_dir / camera.name / date_str
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = dest_dir / f"{time_str}.mp4"

                    # Update progress
                    progress.current_file = str(source_path.name)

                    # Copy file
                    try:
                        shutil.copy2(source_path, dest_path)
                        files_exported += 1
                        bytes_exported += source_path.stat().st_size
                        progress.files_copied += 1
                        progress.bytes_copied = bytes_exported

                        if delete_after_copy:
                            deleted_files.append(source_path)

                    except Exception as e:
                        logger.error(f"Failed to copy {source_path}: {e}")

                # Export detection events if requested
                events_exported = 0
                snapshots_exported = 0

                if include_detections:
                    events_query = (
                        select(ObjectEvent)
                        .options(selectinload(ObjectEvent.camera))
                        .where(ObjectEvent.camera_id.in_(camera_ids))
                        .where(ObjectEvent.event_time >= start_time)
                        .where(ObjectEvent.event_time <= end_time)
                        .order_by(ObjectEvent.event_time)
                    )
                    events_result = await db.execute(events_query)
                    events = list(events_result.scalars().all())

                    # Write events.json (machine-readable)
                    events_json = []
                    for event in events:
                        events_json.append({
                            "id": event.id,
                            "event_type": event.event_type,
                            "class_name": event.class_name,
                            "track_id": event.track_id,
                            "confidence": event.confidence,
                            "duration_seconds": event.duration_seconds,
                            "camera_id": event.camera_id,
                            "camera_name": event.camera.name if event.camera else None,
                            "event_time": event.event_time.isoformat(),
                            "old_state": event.old_state,
                            "new_state": event.new_state,
                        })
                        events_exported += 1

                    events_json_path = detections_dir / "events.json"
                    with open(events_json_path, "w") as f:
                        json.dump(events_json, f, indent=2)

                    # Write events.txt (human-readable)
                    events_txt_path = detections_dir / "events.txt"
                    await self._write_events_txt(
                        events_txt_path,
                        events,
                        cameras,
                        start_time,
                        end_time,
                    )

                    # Copy snapshots if requested
                    if include_snapshots:
                        for event in events:
                            if event.snapshot_path:
                                snapshot_source = self.hot_storage_path / event.snapshot_path
                                if snapshot_source.exists():
                                    # Create unique snapshot filename
                                    timestamp_str = event.event_time.strftime("%Y-%m-%d_%H-%M-%S")
                                    snapshot_name = (
                                        f"{event.track_id}_{event.class_name}_"
                                        f"{timestamp_str}.jpg"
                                    )
                                    snapshot_dest = snapshots_dir / snapshot_name
                                    try:
                                        shutil.copy2(snapshot_source, snapshot_dest)
                                        snapshots_exported += 1
                                    except Exception as e:
                                        logger.error(
                                            f"Failed to copy snapshot: {e}"
                                        )

                # Write manifest
                manifest = {
                    "export_version": "1.0",
                    "export_time": datetime.now(timezone.utc).isoformat(),
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                    },
                    "cameras": [
                        {"id": c.id, "name": c.name}
                        for c in cameras.values()
                    ],
                    "statistics": {
                        "files_exported": files_exported,
                        "bytes_exported": bytes_exported,
                        "events_exported": events_exported,
                        "snapshots_exported": snapshots_exported,
                    },
                    "options": {
                        "include_detections": include_detections,
                        "include_snapshots": include_snapshots,
                        "delete_after_copy": delete_after_copy,
                    },
                }

                manifest_path = export_dir / "manifest.json"
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)

                # Delete source files if requested (after successful export)
                if delete_after_copy:
                    for file_path in deleted_files:
                        try:
                            file_path.unlink()
                            # Cleanup empty directories
                            self._cleanup_empty_dirs(file_path.parent)
                        except Exception as e:
                            logger.error(f"Failed to delete {file_path}: {e}")

                progress.status = "completed"
                progress.completed_at = datetime.now(timezone.utc)
                progress.current_file = None

                return ExportResult(
                    export_id=export_id,
                    success=True,
                    output_path=str(export_dir),
                    files_exported=files_exported,
                    bytes_exported=bytes_exported,
                    events_exported=events_exported,
                    snapshots_exported=snapshots_exported,
                    manifest_path=str(manifest_path),
                )

            finally:
                if close_session:
                    await db.close()

        except Exception as e:
            logger.exception(f"Export failed: {e}")
            progress.status = "failed"
            progress.error_message = str(e)
            progress.completed_at = datetime.now(timezone.utc)

            return ExportResult(
                export_id=export_id,
                success=False,
                output_path=str(export_dir),
                files_exported=0,
                bytes_exported=0,
                events_exported=0,
                snapshots_exported=0,
                error_message=str(e),
            )

    async def _write_events_txt(
        self,
        output_path: Path,
        events: list[ObjectEvent],
        cameras: dict[int, Camera],
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Write human-readable events.txt file."""
        camera_names = ", ".join(c.name for c in cameras.values())

        lines = [
            "=" * 60,
            "Ronin NVR Detection Export",
            "=" * 60,
            f"Export Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"Time Range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} - "
            f"{end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Cameras: {camera_names}",
            "",
            "-" * 60,
            "Detection Events",
            "-" * 60,
            "",
        ]

        for event in events:
            camera = cameras.get(event.camera_id)
            camera_name = camera.name if camera else f"Camera {event.camera_id}"
            timestamp = event.event_time.strftime("%Y-%m-%d %H:%M:%S")
            confidence_pct = round(event.confidence * 100)

            # Main event line
            line = (
                f"{timestamp} | {camera_name} | {event.event_type} | "
                f"{event.class_name} | {confidence_pct}% confidence"
            )
            lines.append(line)

            # Details
            details = f"  Track ID: {event.track_id}"
            if event.duration_seconds:
                details += f" | Duration: {event.duration_seconds:.1f}s"
            lines.append(details)

            # Snapshot reference
            if event.snapshot_path:
                snapshot_name = (
                    f"{event.track_id}_{event.class_name}_"
                    f"{event.event_time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                )
                lines.append(f"  Snapshot: snapshots/{snapshot_name}")

            lines.append("")

        # Write to file
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

    def _cleanup_empty_dirs(self, dir_path: Path) -> None:
        """Remove empty directories up to storage root."""
        try:
            while dir_path != self.hot_storage_path and dir_path.exists():
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    dir_path = dir_path.parent
                else:
                    break
        except OSError:
            pass

    @staticmethod
    def get_export_progress(export_id: str) -> Optional[ExportProgress]:
        """Get the progress of an export."""
        return _export_progress.get(export_id)

    @staticmethod
    def cancel_export(export_id: str) -> bool:
        """Cancel a running export."""
        progress = _export_progress.get(export_id)
        if progress and progress.status == "running":
            progress.status = "cancelled"
            return True
        return False


# Global service instance
offline_export_service = OfflineExportService()
