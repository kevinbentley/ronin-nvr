"""Thumbnail sprite generation service for video scrubbing preview."""

import asyncio
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import get_settings

# Thumbnail configuration
THUMB_WIDTH = 160
THUMB_HEIGHT = 90
THUMB_INTERVAL = 10  # seconds between thumbnails
SPRITE_COLUMNS = 10  # thumbnails per row in sprite sheet


@dataclass
class ThumbnailSprite:
    """Represents a generated thumbnail sprite sheet."""

    sprite_path: Path
    vtt_path: Path
    duration_seconds: int
    thumbnail_count: int
    interval_seconds: int


def get_thumbnail_dir() -> Path:
    """Get the directory for storing thumbnail sprites."""
    settings = get_settings()
    thumb_dir = Path(settings.storage_root) / ".thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    return thumb_dir


def get_sprite_id(recording_path: Path) -> str:
    """Generate a unique ID for a sprite based on the recording path."""
    # Use hash of path + mtime for cache invalidation
    stat = recording_path.stat()
    content = f"{recording_path}:{stat.st_mtime}:{stat.st_size}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


async def get_video_duration(video_path: Path) -> Optional[float]:
    """Get the duration of a video file using ffprobe."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0 and stdout:
            return float(stdout.decode().strip())
    except (ValueError, FileNotFoundError):
        pass
    return None


async def generate_sprite(recording_path: Path) -> Optional[ThumbnailSprite]:
    """Generate a thumbnail sprite sheet for a recording.

    Creates a sprite sheet image and a VTT file that maps timestamps
    to coordinates in the sprite.

    Args:
        recording_path: Path to the video recording file

    Returns:
        ThumbnailSprite with paths to generated files, or None if failed
    """
    if not recording_path.exists():
        return None

    sprite_id = get_sprite_id(recording_path)
    thumb_dir = get_thumbnail_dir()
    sprite_path = thumb_dir / f"{sprite_id}.jpg"
    vtt_path = thumb_dir / f"{sprite_id}.vtt"

    # Check if already generated (cache hit)
    if sprite_path.exists() and vtt_path.exists():
        # Read VTT to get metadata
        vtt_content = vtt_path.read_text()
        lines = [l for l in vtt_content.split("\n") if "-->" in l]
        thumb_count = len(lines)
        duration = thumb_count * THUMB_INTERVAL
        return ThumbnailSprite(
            sprite_path=sprite_path,
            vtt_path=vtt_path,
            duration_seconds=duration,
            thumbnail_count=thumb_count,
            interval_seconds=THUMB_INTERVAL,
        )

    # Get video duration
    duration = await get_video_duration(recording_path)
    if not duration:
        return None

    duration_int = int(duration)
    thumb_count = max(1, duration_int // THUMB_INTERVAL)

    # Calculate sprite dimensions
    rows = (thumb_count + SPRITE_COLUMNS - 1) // SPRITE_COLUMNS

    try:
        # Generate sprite sheet using FFmpeg
        # fps=1/{interval} extracts one frame every N seconds
        # tile={cols}x{rows} arranges them in a grid
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-i", str(recording_path),
            "-vf", f"fps=1/{THUMB_INTERVAL},scale={THUMB_WIDTH}:{THUMB_HEIGHT},tile={SPRITE_COLUMNS}x{rows}",
            "-frames:v", "1",
            "-q:v", "5",  # JPEG quality (2-31, lower is better)
            str(sprite_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            # Log error but don't raise
            print(f"FFmpeg sprite generation failed: {stderr.decode()[:500]}")
            return None

        # Generate VTT file
        vtt_lines = ["WEBVTT", ""]

        for i in range(thumb_count):
            start_sec = i * THUMB_INTERVAL
            end_sec = min((i + 1) * THUMB_INTERVAL, duration_int)

            # Format times as HH:MM:SS.mmm
            start_time = format_vtt_time(start_sec)
            end_time = format_vtt_time(end_sec)

            # Calculate sprite coordinates
            col = i % SPRITE_COLUMNS
            row = i // SPRITE_COLUMNS
            x = col * THUMB_WIDTH
            y = row * THUMB_HEIGHT

            vtt_lines.append(f"{start_time} --> {end_time}")
            vtt_lines.append(f"{sprite_id}.jpg#xywh={x},{y},{THUMB_WIDTH},{THUMB_HEIGHT}")
            vtt_lines.append("")

        vtt_path.write_text("\n".join(vtt_lines))

        return ThumbnailSprite(
            sprite_path=sprite_path,
            vtt_path=vtt_path,
            duration_seconds=duration_int,
            thumbnail_count=thumb_count,
            interval_seconds=THUMB_INTERVAL,
        )

    except Exception as e:
        print(f"Sprite generation error: {e}")
        # Clean up partial files
        sprite_path.unlink(missing_ok=True)
        vtt_path.unlink(missing_ok=True)
        return None


def format_vtt_time(seconds: int) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.000"


def get_cached_sprite(recording_path: Path) -> Optional[ThumbnailSprite]:
    """Check if a sprite already exists for a recording.

    Returns the sprite info if cached, None otherwise.
    """
    if not recording_path.exists():
        return None

    sprite_id = get_sprite_id(recording_path)
    thumb_dir = get_thumbnail_dir()
    sprite_path = thumb_dir / f"{sprite_id}.jpg"
    vtt_path = thumb_dir / f"{sprite_id}.vtt"

    if sprite_path.exists() and vtt_path.exists():
        vtt_content = vtt_path.read_text()
        lines = [l for l in vtt_content.split("\n") if "-->" in l]
        thumb_count = len(lines)
        duration = thumb_count * THUMB_INTERVAL
        return ThumbnailSprite(
            sprite_path=sprite_path,
            vtt_path=vtt_path,
            duration_seconds=duration,
            thumbnail_count=thumb_count,
            interval_seconds=THUMB_INTERVAL,
        )

    return None


async def cleanup_old_sprites(max_age_days: int = 30) -> int:
    """Remove sprite files older than max_age_days.

    Returns number of files removed.
    """
    import time

    thumb_dir = get_thumbnail_dir()
    now = time.time()
    max_age_seconds = max_age_days * 24 * 3600
    removed = 0

    for path in thumb_dir.iterdir():
        if path.suffix in (".jpg", ".vtt"):
            age = now - path.stat().st_mtime
            if age > max_age_seconds:
                path.unlink()
                removed += 1

    return removed
