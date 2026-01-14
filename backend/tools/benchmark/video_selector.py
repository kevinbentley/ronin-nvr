"""Video selection with timezone-aware daytime filtering."""

import logging
import random
import re
from datetime import datetime
from pathlib import Path

import cv2

from .config import BenchmarkConfig
from .models import VideoInfo

logger = logging.getLogger(__name__)

# Pattern: CameraName/YYYY-MM-DD/HH-MM-SS.mp4
# Camera names can be like: Hangar_East, House_NorthEast, etc.
VIDEO_PATTERN = re.compile(
    r"([A-Za-z0-9_]+)/(\d{4}-\d{2}-\d{2})/(\d{2}-\d{2}-\d{2})\.mp4$"
)


def parse_video_path(path: Path) -> tuple[str, str, str, datetime] | None:
    """Parse video path to extract camera ID, date, time, and UTC timestamp.

    Expected format: .../CameraName/YYYY-MM-DD/HH-MM-SS.mp4
    e.g., /opt3/ronin/storage/Hangar_East/2025-12-31/14-30-00.mp4

    Args:
        path: Path to video file

    Returns:
        Tuple of (camera_id, date_str, time_str, timestamp_utc) or None if invalid
    """
    path_str = str(path)
    match = VIDEO_PATTERN.search(path_str)
    if not match:
        return None

    camera_id = match.group(1)
    date_str = match.group(2)
    time_str = match.group(3)

    # Parse timestamp (files are in UTC)
    try:
        timestamp_utc = datetime.strptime(
            f"{date_str} {time_str.replace('-', ':')}", "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        return None

    return camera_id, date_str, time_str, timestamp_utc


def is_daytime_utc(timestamp: datetime, utc_ranges: list[tuple[int, int]]) -> bool:
    """Check if timestamp falls within daytime UTC hours.

    Args:
        timestamp: UTC timestamp to check
        utc_ranges: List of (start_hour, end_hour) tuples defining daytime

    Returns:
        True if timestamp is during daytime
    """
    hour = timestamp.hour
    for start, end in utc_ranges:
        if start <= end:
            # Simple range like (9, 17)
            if start <= hour <= end:
                return True
        else:
            # Wrapping range like (22, 6) - not used in our case
            if hour >= start or hour <= end:
                return True
    return False


def get_video_info(path: Path) -> VideoInfo | None:
    """Create VideoInfo from a video file path.

    Args:
        path: Path to video file

    Returns:
        VideoInfo object or None if parsing fails
    """
    parsed = parse_video_path(path)
    if not parsed:
        return None

    camera_id, date_str, time_str, timestamp_utc = parsed

    try:
        file_size_mb = path.stat().st_size / (1024 * 1024)
    except OSError:
        return None

    return VideoInfo(
        path=path,
        camera_id=camera_id,
        date_str=date_str,
        time_str=time_str,
        timestamp_utc=timestamp_utc,
        file_size_mb=file_size_mb,
    )


def get_video_metadata(video: VideoInfo) -> VideoInfo:
    """Populate video metadata by reading the file.

    Args:
        video: VideoInfo with basic info

    Returns:
        VideoInfo with metadata populated
    """
    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video.path}")
        return video

    try:
        video.fps = cap.get(cv2.CAP_PROP_FPS)
        video.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if video.fps and video.fps > 0 and video.frame_count:
            video.duration_seconds = video.frame_count / video.fps
    finally:
        cap.release()

    return video


def select_videos(
    config: BenchmarkConfig,
    exclude_paths: set[str] | None = None,
) -> list[VideoInfo]:
    """Select daytime videos for benchmarking.

    Args:
        config: Benchmark configuration
        exclude_paths: Set of video paths to exclude (already processed)

    Returns:
        List of selected VideoInfo objects
    """
    if exclude_paths is None:
        exclude_paths = set()

    storage_root = config.storage_root
    if not storage_root.exists():
        logger.error(f"Storage root not found: {storage_root}")
        return []

    # Find all MP4 files
    logger.info(f"Scanning for videos in {storage_root}...")
    all_videos: list[VideoInfo] = []

    for mp4_path in storage_root.rglob("*.mp4"):
        # Skip already processed
        if str(mp4_path) in exclude_paths:
            continue

        video = get_video_info(mp4_path)
        if video is None:
            continue

        # Filter by file size
        if video.file_size_mb < config.min_file_size_mb:
            continue
        if video.file_size_mb > config.max_file_size_mb:
            continue

        # Filter by daytime
        if not is_daytime_utc(video.timestamp_utc, config.daytime_utc_ranges):
            continue

        all_videos.append(video)

    logger.info(f"Found {len(all_videos)} daytime videos matching criteria")

    # Randomly select if we have more than needed
    if len(all_videos) > config.target_total_videos:
        random.shuffle(all_videos)
        all_videos = all_videos[: config.target_total_videos]
        logger.info(f"Randomly selected {len(all_videos)} videos for benchmark")

    # Sort by timestamp for consistent ordering
    all_videos.sort(key=lambda v: v.timestamp_utc)

    return all_videos


def select_videos_by_camera(
    config: BenchmarkConfig,
    videos_per_camera: int = 10,
    exclude_paths: set[str] | None = None,
) -> list[VideoInfo]:
    """Select videos ensuring even distribution across cameras.

    Args:
        config: Benchmark configuration
        videos_per_camera: Target number of videos per camera
        exclude_paths: Set of video paths to exclude

    Returns:
        List of selected VideoInfo objects
    """
    if exclude_paths is None:
        exclude_paths = set()

    storage_root = config.storage_root
    if not storage_root.exists():
        logger.error(f"Storage root not found: {storage_root}")
        return []

    # Group videos by camera
    logger.info(f"Scanning for videos by camera in {storage_root}...")
    by_camera: dict[str, list[VideoInfo]] = {}

    for mp4_path in storage_root.rglob("*.mp4"):
        if str(mp4_path) in exclude_paths:
            continue

        video = get_video_info(mp4_path)
        if video is None:
            continue

        if video.file_size_mb < config.min_file_size_mb:
            continue
        if video.file_size_mb > config.max_file_size_mb:
            continue

        if not is_daytime_utc(video.timestamp_utc, config.daytime_utc_ranges):
            continue

        if video.camera_id not in by_camera:
            by_camera[video.camera_id] = []
        by_camera[video.camera_id].append(video)

    logger.info(f"Found videos from {len(by_camera)} cameras")

    # Select from each camera
    selected: list[VideoInfo] = []
    for camera_id, videos in sorted(by_camera.items()):
        if len(videos) > videos_per_camera:
            random.shuffle(videos)
            videos = videos[:videos_per_camera]
        selected.extend(videos)
        logger.info(f"  {camera_id}: selected {len(videos)} videos")

    # Cap at target total
    if len(selected) > config.target_total_videos:
        random.shuffle(selected)
        selected = selected[: config.target_total_videos]

    selected.sort(key=lambda v: (v.camera_id, v.timestamp_utc))

    logger.info(f"Selected {len(selected)} total videos for benchmark")
    return selected
