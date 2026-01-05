"""Playback service for recorded video files."""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RecordingFile:
    """Information about a recording file."""

    path: Path
    camera_name: str
    date: date
    start_time: datetime
    size: int
    duration_seconds: Optional[int] = None
    is_in_progress: bool = False  # True if this recording is currently being written

    @property
    def id(self) -> str:
        """Generate unique ID from path.

        Uses '::' as separator since it won't appear in camera names or filenames.
        """
        rel_path = self.path.relative_to(Path(settings.storage_root))
        return str(rel_path).replace("/", "::").replace("\\", "::")

    @property
    def filename(self) -> str:
        """Get filename."""
        return self.path.name


@dataclass
class DayRecordings:
    """Recordings for a single day."""

    camera_name: str
    date: date
    files: list[RecordingFile]
    total_duration_seconds: int
    total_size_bytes: int

    @property
    def start_time(self) -> Optional[datetime]:
        """Get earliest recording time."""
        if not self.files:
            return None
        return min(f.start_time for f in self.files)

    @property
    def end_time(self) -> Optional[datetime]:
        """Get latest recording end time."""
        if not self.files:
            return None
        last = max(self.files, key=lambda f: f.start_time)
        if last.duration_seconds:
            return last.start_time + timedelta(seconds=last.duration_seconds)
        return last.start_time


class PlaybackService:
    """Service for managing video playback."""

    def __init__(self, storage_root: Optional[Path] = None):
        self.storage_root = storage_root or Path(settings.storage_root)

    def _parse_filename_time(self, filename: str) -> Optional[datetime]:
        """Parse time from filename like 12-30-00.mp4."""
        match = re.match(r"(\d{2})-(\d{2})-(\d{2})\.mp4", filename)
        if match:
            return datetime.strptime(f"{match.group(1)}:{match.group(2)}:{match.group(3)}", "%H:%M:%S")
        return None

    def _parse_date_dir(self, dirname: str) -> Optional[date]:
        """Parse date from directory name like 2024-01-15."""
        try:
            return datetime.strptime(dirname, "%Y-%m-%d").date()
        except ValueError:
            return None

    def scan_recordings(
        self,
        camera_name: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> list[RecordingFile]:
        """Scan storage for recording files."""
        recordings: list[RecordingFile] = []

        if not self.storage_root.exists():
            return recordings

        # Iterate camera directories
        for camera_dir in self.storage_root.iterdir():
            if not camera_dir.is_dir() or camera_dir.name.startswith("."):
                continue

            cam_name = camera_dir.name
            if camera_name and cam_name != camera_name:
                continue

            # Iterate date directories
            for date_dir in camera_dir.iterdir():
                if not date_dir.is_dir():
                    continue

                rec_date = self._parse_date_dir(date_dir.name)
                if not rec_date:
                    continue

                # Apply date filters
                if start_date and rec_date < start_date:
                    continue
                if end_date and rec_date > end_date:
                    continue

                # Iterate video files
                for video_file in sorted(date_dir.glob("*.mp4")):
                    file_time = self._parse_filename_time(video_file.name)
                    if not file_time:
                        continue

                    # Combine date and time with UTC timezone.
                    # Note: filename encodes local time when recorded, but we
                    # store as UTC for consistent backend handling. The frontend
                    # is responsible for displaying without timezone conversion.
                    start_dt = datetime.combine(
                        rec_date, file_time.time(), tzinfo=timezone.utc
                    )

                    try:
                        stat = video_file.stat()
                        recordings.append(RecordingFile(
                            path=video_file,
                            camera_name=cam_name,
                            date=rec_date,
                            start_time=start_dt,
                            size=stat.st_size,
                            duration_seconds=settings.segment_duration_minutes * 60,
                            is_in_progress=False,  # Will be set below
                        ))
                    except OSError:
                        continue

        # Sort by start time
        recordings.sort(key=lambda r: (r.camera_name, r.start_time))

        # Mark only the most recent recording per camera as in-progress
        # if it was modified recently (within 2x segment duration)
        now = datetime.now(timezone.utc)
        threshold_seconds = settings.segment_duration_minutes * 60 * 2
        latest_per_camera: dict[str, RecordingFile] = {}
        for rec in recordings:
            latest_per_camera[rec.camera_name] = rec

        for rec in latest_per_camera.values():
            try:
                mtime = datetime.fromtimestamp(rec.path.stat().st_mtime, tz=timezone.utc)
                age_seconds = (now - mtime).total_seconds()
                if age_seconds < threshold_seconds:
                    rec.is_in_progress = True
            except OSError:
                pass

        return recordings

    def get_recording_by_id(self, recording_id: str) -> Optional[RecordingFile]:
        """Get a recording file by its ID."""
        # Convert ID back to path (uses '::' as separator)
        rel_path = recording_id.replace("::", "/")
        full_path = self.storage_root / rel_path

        if not full_path.exists() or not full_path.is_file():
            return None

        # Parse path components
        parts = Path(rel_path).parts
        if len(parts) != 3:
            return None

        camera_name = parts[0]
        rec_date = self._parse_date_dir(parts[1])
        file_time = self._parse_filename_time(parts[2])

        if not rec_date or not file_time:
            return None

        # Combine date and time with UTC timezone (matches scan_recordings)
        start_dt = datetime.combine(rec_date, file_time.time(), tzinfo=timezone.utc)

        try:
            stat = full_path.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            now = datetime.now(timezone.utc)
            age_seconds = (now - mtime).total_seconds()
            threshold_seconds = settings.segment_duration_minutes * 60 * 2

            # Only mark as in-progress if recently modified AND is the latest file
            is_in_progress = False
            if age_seconds < threshold_seconds:
                # Check if this is the most recent recording for this camera
                camera_dir = self.storage_root / camera_name
                latest_file = None
                latest_mtime = None
                for date_dir in camera_dir.iterdir():
                    if not date_dir.is_dir():
                        continue
                    for mp4 in date_dir.glob("*.mp4"):
                        try:
                            mp4_mtime = mp4.stat().st_mtime
                            if latest_mtime is None or mp4_mtime > latest_mtime:
                                latest_mtime = mp4_mtime
                                latest_file = mp4
                        except OSError:
                            continue
                is_in_progress = latest_file == full_path

            return RecordingFile(
                path=full_path,
                camera_name=camera_name,
                date=rec_date,
                start_time=start_dt,
                size=stat.st_size,
                duration_seconds=settings.segment_duration_minutes * 60,
                is_in_progress=is_in_progress,
            )
        except OSError:
            return None

    def get_available_dates(self, camera_name: str) -> list[date]:
        """Get list of dates with recordings for a camera."""
        dates: list[date] = []

        camera_dir = self.storage_root / camera_name
        if not camera_dir.exists():
            return dates

        for date_dir in camera_dir.iterdir():
            if not date_dir.is_dir():
                continue

            rec_date = self._parse_date_dir(date_dir.name)
            if rec_date and any(date_dir.glob("*.mp4")):
                dates.append(rec_date)

        dates.sort(reverse=True)
        return dates

    def get_day_recordings(
        self,
        camera_name: str,
        rec_date: date,
    ) -> Optional[DayRecordings]:
        """Get all recordings for a camera on a specific date."""
        files = self.scan_recordings(
            camera_name=camera_name,
            start_date=rec_date,
            end_date=rec_date,
        )

        if not files:
            return None

        total_duration = sum(f.duration_seconds or 0 for f in files)
        total_size = sum(f.size for f in files)

        return DayRecordings(
            camera_name=camera_name,
            date=rec_date,
            files=files,
            total_duration_seconds=total_duration,
            total_size_bytes=total_size,
        )

    def get_cameras_with_recordings(self) -> list[str]:
        """Get list of camera names that have recordings."""
        cameras: list[str] = []

        if not self.storage_root.exists():
            return cameras

        for camera_dir in self.storage_root.iterdir():
            if not camera_dir.is_dir() or camera_dir.name.startswith("."):
                continue

            # Check if any date directories have files
            has_recordings = any(
                date_dir.is_dir() and any(date_dir.glob("*.mp4"))
                for date_dir in camera_dir.iterdir()
            )
            if has_recordings:
                cameras.append(camera_dir.name)

        cameras.sort()
        return cameras

    def export_clip(
        self,
        camera_name: str,
        start_time: datetime,
        end_time: datetime,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """Export a time range as a single video file."""
        # Find all recordings that overlap with the time range
        start_date = start_time.date()
        end_date = end_time.date()

        recordings = self.scan_recordings(
            camera_name=camera_name,
            start_date=start_date,
            end_date=end_date,
        )

        # Filter to recordings that overlap with time range
        segment_duration = settings.segment_duration_minutes * 60
        relevant = []
        for rec in recordings:
            rec_end = rec.start_time + timedelta(seconds=segment_duration)
            if rec.start_time <= end_time and rec_end >= start_time:
                relevant.append(rec)

        if not relevant:
            return None

        # Create output path if not specified
        if not output_path:
            export_dir = self.storage_root / ".exports"
            export_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = export_dir / f"{camera_name}_{timestamp}.mp4"

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            logger.error("FFmpeg not found")
            return None

        # For single file, just copy/trim
        if len(relevant) == 1:
            rec = relevant[0]
            # Calculate trim times
            trim_start = max(0, (start_time - rec.start_time).total_seconds())
            trim_end = min(segment_duration, (end_time - rec.start_time).total_seconds())
            duration = trim_end - trim_start

            cmd = [
                ffmpeg_path, "-y",
                "-ss", str(trim_start),
                "-i", str(rec.path),
                "-t", str(duration),
                "-c", "copy",
                str(output_path),
            ]
        else:
            # Create concat file for multiple segments
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                for rec in relevant:
                    f.write(f"file '{rec.path}'\n")
                concat_file = f.name

            try:
                # First concat, then trim
                cmd = [
                    ffmpeg_path, "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_file,
                    "-c", "copy",
                    str(output_path),
                ]

                # TODO: Add trimming for multi-file export
            finally:
                pass  # Will clean up concat_file after

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"FFmpeg export failed: {result.stderr.decode()}")
                return None
            return output_path
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg export timed out")
            return None
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None


# Global service instance
playback_service = PlaybackService()
