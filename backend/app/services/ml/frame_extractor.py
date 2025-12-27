"""Frame extraction service using FFmpeg."""

import asyncio
import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Video metadata extracted from file."""

    duration_seconds: float
    width: int
    height: int
    fps: float
    codec: str
    total_frames: int

    @property
    def frame_size_bytes(self) -> int:
        """Size of a single raw BGR frame in bytes."""
        return self.width * self.height * 3


class FrameExtractor:
    """Extract frames from video files using FFmpeg.

    Extracts frames at a configurable rate and yields them as numpy arrays
    suitable for ML model input.
    """

    def __init__(
        self,
        fps: float = 1.0,
        max_dimension: int = 640,
        pixel_format: str = "bgr24",
    ):
        """Initialize frame extractor.

        Args:
            fps: Frames per second to extract (default 1.0)
            max_dimension: Scale video so largest dimension doesn't exceed this
            pixel_format: Output pixel format (bgr24 for OpenCV/ML compatibility)
        """
        self.fps = fps
        self.max_dimension = max_dimension
        self.pixel_format = pixel_format

    async def get_video_info(self, video_path: Path) -> Optional[VideoInfo]:
        """Get video metadata using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            VideoInfo object or None if extraction fails
        """
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None

        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-select_streams", "v:0",
            str(video_path),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"ffprobe failed: {stderr.decode()}")
                return None

            data = json.loads(stdout.decode())

            if not data.get("streams"):
                logger.error("No video streams found")
                return None

            stream = data["streams"][0]
            format_info = data.get("format", {})

            # Parse frame rate (may be "30/1" format)
            fps_str = stream.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 30.0
            else:
                fps = float(fps_str)

            duration = float(format_info.get("duration", 0))
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))
            codec = stream.get("codec_name", "unknown")

            # Calculate total frames
            total_frames = int(duration * fps) if duration > 0 else 0

            return VideoInfo(
                duration_seconds=duration,
                width=width,
                height=height,
                fps=fps,
                codec=codec,
                total_frames=total_frames,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ffprobe output: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return None

    def _calculate_scale(self, width: int, height: int) -> tuple[int, int]:
        """Calculate scaled dimensions maintaining aspect ratio.

        Args:
            width: Original width
            height: Original height

        Returns:
            Tuple of (new_width, new_height), both divisible by 2 for FFmpeg
        """
        if width <= self.max_dimension and height <= self.max_dimension:
            # Ensure dimensions are even
            return (width // 2) * 2, (height // 2) * 2

        if width > height:
            new_width = self.max_dimension
            new_height = int(height * (self.max_dimension / width))
        else:
            new_height = self.max_dimension
            new_width = int(width * (self.max_dimension / height))

        # Ensure dimensions are even (FFmpeg requirement)
        return (new_width // 2) * 2, (new_height // 2) * 2

    async def extract_frames(
        self,
        video_path: Path,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> AsyncIterator[tuple[np.ndarray, int, float]]:
        """Extract frames from video file.

        Yields frames as numpy arrays along with frame number and timestamp.

        Args:
            video_path: Path to video file
            start_time: Optional start time in seconds
            duration: Optional duration to process in seconds

        Yields:
            Tuple of (frame, frame_number, timestamp_ms)
            - frame: numpy array of shape (height, width, 3) in BGR format
            - frame_number: 0-indexed frame number
            - timestamp_ms: timestamp in milliseconds from video start
        """
        video_info = await self.get_video_info(video_path)
        if not video_info:
            logger.error(f"Could not get video info for {video_path}")
            return

        # Calculate output dimensions
        out_width, out_height = self._calculate_scale(
            video_info.width, video_info.height
        )
        frame_size = out_width * out_height * 3

        # Build FFmpeg command
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

        # Input seeking (if specified)
        if start_time is not None and start_time > 0:
            cmd.extend(["-ss", str(start_time)])

        cmd.extend(["-i", str(video_path)])

        # Duration limit (if specified)
        if duration is not None and duration > 0:
            cmd.extend(["-t", str(duration)])

        # Video filter: fps and scale
        vf_parts = [f"fps={self.fps}"]
        vf_parts.append(f"scale={out_width}:{out_height}")
        cmd.extend(["-vf", ",".join(vf_parts)])

        # Output format
        cmd.extend([
            "-f", "rawvideo",
            "-pix_fmt", self.pixel_format,
            "pipe:1",
        ])

        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            frame_number = 0
            base_timestamp = start_time * 1000 if start_time else 0

            while True:
                # Read exactly one frame
                frame_data = await process.stdout.read(frame_size)

                if not frame_data:
                    break

                if len(frame_data) < frame_size:
                    # Incomplete frame at end of video
                    logger.debug(
                        f"Incomplete frame: got {len(frame_data)}, expected {frame_size}"
                    )
                    break

                # Convert to numpy array
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((out_height, out_width, 3))

                # Calculate timestamp
                timestamp_ms = base_timestamp + (frame_number / self.fps) * 1000

                yield frame, frame_number, timestamp_ms
                frame_number += 1

            # Wait for process to finish
            await process.wait()

            if process.returncode != 0:
                stderr = await process.stderr.read()
                logger.warning(f"FFmpeg exited with code {process.returncode}: {stderr.decode()}")

            logger.info(f"Extracted {frame_number} frames from {video_path}")

        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise

    async def extract_frames_to_list(
        self,
        video_path: Path,
        max_frames: Optional[int] = None,
    ) -> list[tuple[np.ndarray, int, float]]:
        """Extract frames and return as a list (for testing/small videos).

        Args:
            video_path: Path to video file
            max_frames: Optional maximum number of frames to extract

        Returns:
            List of (frame, frame_number, timestamp_ms) tuples
        """
        frames = []
        async for frame, frame_num, timestamp_ms in self.extract_frames(video_path):
            frames.append((frame, frame_num, timestamp_ms))
            if max_frames and len(frames) >= max_frames:
                break
        return frames

    async def count_frames(self, video_path: Path) -> int:
        """Estimate the number of frames that will be extracted.

        Args:
            video_path: Path to video file

        Returns:
            Estimated frame count
        """
        video_info = await self.get_video_info(video_path)
        if not video_info:
            return 0

        return int(video_info.duration_seconds * self.fps)
