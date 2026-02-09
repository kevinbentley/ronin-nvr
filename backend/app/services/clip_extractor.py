"""Clip extraction service for detection events.

Extracts short video clips around detection events (arrivals/departures)
for easy review and download.
"""

import asyncio
import logging
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ClipExtractorService:
    """Service for extracting video clips around detection events."""

    def __init__(
        self,
        storage_root: Optional[Path] = None,
        pre_duration: float = 5.0,
        post_duration: float = 5.0,
    ):
        """Initialize the clip extractor.

        Args:
            storage_root: Root directory for video storage
            pre_duration: Seconds of video before the event
            post_duration: Seconds of video after the event
        """
        self.storage_root = storage_root or Path(settings.storage_root)
        self.pre_duration = pre_duration
        self.post_duration = post_duration
        self.clips_dir = self.storage_root / ".clips"

    def _safe_camera_name(self, name: str) -> str:
        """Convert camera name to filesystem-safe version."""
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)

    def _find_segment_for_time(
        self,
        camera_name: str,
        target_time: datetime,
    ) -> Optional[Path]:
        """Find the recording segment containing the target time.

        Args:
            camera_name: Name of the camera
            target_time: Time to find segment for

        Returns:
            Path to the segment file, or None if not found
        """
        safe_name = self._safe_camera_name(camera_name)
        camera_dir = self.storage_root / safe_name

        if not camera_dir.exists():
            logger.warning(f"Camera directory not found: {camera_dir}")
            return None

        # Get date directory (recordings use UTC time for directories)
        target_utc = target_time.astimezone(timezone.utc)
        date_str = target_utc.strftime("%Y-%m-%d")
        date_dir = camera_dir / date_str

        if not date_dir.exists():
            logger.warning(f"Date directory not found: {date_dir}")
            return None

        # Find the segment file - filename is HH-MM-SS.mp4 in UTC
        # Each segment is ~15 minutes, so we need to find which one contains our time
        segment_duration = settings.segment_duration_minutes * 60

        # List all mp4 files and find the one that contains our target time
        mp4_files = sorted(date_dir.glob("*.mp4"))

        for mp4_file in mp4_files:
            # Parse time from filename (HH-MM-SS.mp4)
            try:
                time_str = mp4_file.stem  # e.g., "14-30-00"
                parts = time_str.split("-")
                if len(parts) != 3:
                    continue

                hour, minute, second = int(parts[0]), int(parts[1]), int(parts[2])
                segment_start = target_utc.replace(
                    hour=hour, minute=minute, second=second, microsecond=0
                )

                # Handle day boundary - if segment time is later than target time
                # and we're near midnight, segment might be from previous query
                if segment_start > target_utc:
                    # Check previous day's files
                    continue

                segment_end = segment_start + timedelta(seconds=segment_duration)

                # Check if target time falls within this segment
                if segment_start <= target_utc <= segment_end:
                    return mp4_file

            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse segment filename {mp4_file}: {e}")
                continue

        logger.warning(
            f"No segment found for camera={camera_name}, time={target_time}"
        )
        return None

    def _get_clip_output_path(
        self,
        camera_id: int,
        event_id: int,
        event_time: datetime,
    ) -> Path:
        """Generate output path for a video clip.

        Args:
            camera_id: ID of the camera
            event_id: ID of the detection event
            event_time: Time of the event

        Returns:
            Path where the clip should be saved
        """
        # Use UTC for directory structure
        event_utc = event_time.astimezone(timezone.utc)
        date_str = event_utc.strftime("%Y-%m-%d")
        time_str = event_utc.strftime("%H-%M-%S")

        clip_dir = self.clips_dir / str(camera_id) / date_str
        clip_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{event_id}-{time_str}.mp4"
        return clip_dir / filename

    async def extract_clip_for_event(
        self,
        event_id: int,
        camera_id: int,
        camera_name: str,
        event_time: datetime,
    ) -> Optional[str]:
        """Extract a video clip for a detection event.

        Extracts a clip starting pre_duration seconds before the event
        and ending post_duration seconds after.

        Args:
            event_id: Database ID of the object event
            camera_id: ID of the camera
            camera_name: Name of the camera
            event_time: Time of the detection event

        Returns:
            Relative path to the clip (from storage_root), or None if extraction failed
        """
        try:
            # Calculate time range for clip
            clip_start = event_time - timedelta(seconds=self.pre_duration)
            clip_end = event_time + timedelta(seconds=self.post_duration)
            total_duration = self.pre_duration + self.post_duration

            # Find the segment(s) containing our time range
            # For simplicity, we'll use the segment at the event time
            # (clips are short enough to usually fit in one segment)
            segment_path = self._find_segment_for_time(camera_name, event_time)

            if not segment_path:
                logger.error(
                    f"Could not find segment for event {event_id} at {event_time}"
                )
                return None

            # Calculate seek position within the segment
            # Parse segment start time from filename
            time_str = segment_path.stem
            parts = time_str.split("-")
            hour, minute, second = int(parts[0]), int(parts[1]), int(parts[2])

            event_utc = event_time.astimezone(timezone.utc)
            segment_start = event_utc.replace(
                hour=hour, minute=minute, second=second, microsecond=0
            )

            # Calculate offset into segment for clip start
            clip_start_utc = clip_start.astimezone(timezone.utc)
            seek_offset = (clip_start_utc - segment_start).total_seconds()

            # Ensure seek offset is non-negative
            if seek_offset < 0:
                # Clip starts before this segment - adjust
                logger.warning(
                    f"Clip start {clip_start} is before segment start {segment_start}, "
                    f"adjusting clip"
                )
                seek_offset = 0
                total_duration = min(
                    total_duration,
                    (clip_end.astimezone(timezone.utc) - segment_start).total_seconds()
                )

            # Generate output path
            output_path = self._get_clip_output_path(camera_id, event_id, event_time)

            # Use FFmpeg to extract the clip
            success = await self._extract_with_ffmpeg(
                input_path=segment_path,
                output_path=output_path,
                seek_offset=seek_offset,
                duration=total_duration,
            )

            if success:
                # Return relative path from storage root
                rel_path = output_path.relative_to(self.storage_root)
                return str(rel_path)
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to extract clip for event {event_id}: {e}")
            return None

    async def _extract_with_ffmpeg(
        self,
        input_path: Path,
        output_path: Path,
        seek_offset: float,
        duration: float,
    ) -> bool:
        """Extract a clip using FFmpeg.

        Args:
            input_path: Path to the source video
            output_path: Path for the output clip
            seek_offset: Seconds to seek into the source video
            duration: Duration of the clip in seconds

        Returns:
            True if extraction succeeded, False otherwise
        """
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            logger.error("FFmpeg not found in PATH")
            return False

        # Build FFmpeg command
        # Use -ss before -i for fast seeking
        cmd = [
            ffmpeg_path,
            "-y",  # Overwrite output
            "-hide_banner",
            "-loglevel", "warning",
            "-ss", f"{seek_offset:.3f}",  # Seek to start position
            "-i", str(input_path),
            "-t", f"{duration:.3f}",  # Duration
            "-c", "copy",  # Copy without re-encoding
            "-movflags", "+faststart",  # Optimize for web playback
            str(output_path),
        ]

        try:
            # Run FFmpeg asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30.0,  # 30 second timeout for clip extraction
            )

            if process.returncode != 0:
                logger.error(
                    f"FFmpeg clip extraction failed (code {process.returncode}): "
                    f"{stderr.decode()}"
                )
                return False

            logger.info(f"Successfully extracted clip to {output_path}")
            return True

        except asyncio.TimeoutError:
            logger.error("FFmpeg clip extraction timed out")
            if process:
                process.kill()
            return False
        except Exception as e:
            logger.error(f"FFmpeg clip extraction error: {e}")
            return False


# Global service instance
clip_extractor_service = ClipExtractorService()
