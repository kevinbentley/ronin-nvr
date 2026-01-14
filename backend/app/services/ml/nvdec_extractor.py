"""NVDEC hardware video decoder for GPU-accelerated frame extraction.

This module provides GPU-accelerated video decoding using NVIDIA's NVDEC
(Video Decoder) hardware through OpenCV's cudacodec module. Key features:

- Hardware H.264/H.265/VP9/AV1 decoding
- Frames decode directly to GpuMat (VRAM)
- Zero-copy pipeline integration
- Significant CPU offload vs FFmpeg software decode

Performance comparison (1080p H.264):
- CPU FFmpeg: ~200 FPS, 100% CPU core
- NVDEC: ~500+ FPS, <5% CPU, dedicated decoder hardware

Typical surveillance camera codecs supported:
- H.264 (AVC) - most common
- H.265 (HEVC) - newer cameras
- MJPEG - webcams/older cameras (CPU fallback)

Note: NVDEC has limited decode instances per GPU:
- Consumer GPUs (RTX 3090): 3 concurrent sessions
- Professional GPUs (A4000): unlimited
For 16+ cameras, may need software decode fallback.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Video file/stream metadata."""

    width: int
    height: int
    fps: float
    frame_count: int
    codec: str
    duration_seconds: float

    @property
    def resolution(self) -> str:
        """Human-readable resolution."""
        if self.height >= 2160:
            return "4K"
        elif self.height >= 1440:
            return "1440p"
        elif self.height >= 1080:
            return "1080p"
        elif self.height >= 720:
            return "720p"
        else:
            return f"{self.height}p"


class NVDECExtractor:
    """NVDEC hardware-accelerated video decoder.

    Decodes video files or RTSP streams using GPU hardware, delivering
    frames directly to GPU memory (GpuMat) for zero-copy processing.

    Example:
        >>> extractor = NVDECExtractor("/path/to/video.mp4")
        >>> for gpu_frame in extractor.frames():
        ...     # gpu_frame is cv2.cuda.GpuMat - already in GPU memory
        ...     result = motion_detector.apply(gpu_frame)

    For RTSP streams:
        >>> extractor = NVDECExtractor("rtsp://camera/stream", is_stream=True)
        >>> for gpu_frame in extractor.frames():
        ...     process(gpu_frame)
    """

    def __init__(
        self,
        source: Union[str, Path],
        is_stream: bool = False,
        device_id: int = 0,
        fallback_to_cpu: bool = True,
    ):
        """Initialize NVDEC extractor.

        Args:
            source: Video file path or RTSP URL
            is_stream: True if source is a live stream (affects buffering)
            device_id: CUDA device ID for decoding
            fallback_to_cpu: Fall back to CPU decode if NVDEC unavailable
        """
        self.source = str(source)
        self.is_stream = is_stream
        self.device_id = device_id
        self.fallback_to_cpu = fallback_to_cpu

        self._reader: Optional[cv2.cudacodec.VideoReader] = None
        self._cpu_reader: Optional[cv2.VideoCapture] = None
        self._using_nvdec = False
        self._info: Optional[VideoInfo] = None
        self._frame_count = 0

    def open(self) -> bool:
        """Open the video source.

        Returns:
            True if opened successfully
        """
        # Try NVDEC first
        if self._try_nvdec():
            self._using_nvdec = True
            logger.info(f"NVDEC decode: {self.source}")
            return True

        # Fall back to CPU
        if self.fallback_to_cpu:
            if self._try_cpu():
                self._using_nvdec = False
                logger.info(f"CPU decode (fallback): {self.source}")
                return True

        logger.error(f"Failed to open video: {self.source}")
        return False

    def _try_nvdec(self) -> bool:
        """Try to open with NVDEC hardware decoder."""
        try:
            # Check if cudacodec is available
            if not hasattr(cv2, "cudacodec"):
                logger.debug("cv2.cudacodec not available")
                return False

            # Create video reader
            # For RTSP streams, we might need special handling
            if self.is_stream:
                # RTSP streams need specific params
                params = cv2.cudacodec.VideoReaderInitParams()
                self._reader = cv2.cudacodec.createVideoReader(
                    self.source, params=params
                )
            else:
                self._reader = cv2.cudacodec.createVideoReader(self.source)

            # Try to read a frame to verify it works
            ret, _ = self._reader.nextFrame()
            if not ret:
                logger.debug("NVDEC reader created but failed to read frame")
                return False

            # Get video info
            self._extract_nvdec_info()
            return True

        except Exception as e:
            logger.debug(f"NVDEC init failed: {e}")
            return False

    def _try_cpu(self) -> bool:
        """Try to open with CPU decoder (FFmpeg)."""
        try:
            self._cpu_reader = cv2.VideoCapture(self.source)
            if not self._cpu_reader.isOpened():
                return False

            # Read a test frame
            ret, _ = self._cpu_reader.read()
            if not ret:
                return False

            # Reset to beginning
            self._cpu_reader.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self._extract_cpu_info()
            return True

        except Exception as e:
            logger.debug(f"CPU decode init failed: {e}")
            return False

    def _extract_nvdec_info(self) -> None:
        """Extract video info from NVDEC reader."""
        try:
            format_info = self._reader.format()
            self._info = VideoInfo(
                width=format_info.width,
                height=format_info.height,
                fps=format_info.fps if hasattr(format_info, "fps") else 30.0,
                frame_count=0,  # Not available from cudacodec
                codec=str(format_info.codec) if hasattr(format_info, "codec") else "unknown",
                duration_seconds=0,
            )
        except Exception as e:
            logger.warning(f"Could not extract NVDEC video info: {e}")
            self._info = VideoInfo(0, 0, 30.0, 0, "unknown", 0)

    def _extract_cpu_info(self) -> None:
        """Extract video info from CPU reader."""
        self._info = VideoInfo(
            width=int(self._cpu_reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cpu_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self._cpu_reader.get(cv2.CAP_PROP_FPS) or 30.0,
            frame_count=int(self._cpu_reader.get(cv2.CAP_PROP_FRAME_COUNT)),
            codec=self._get_fourcc_string(),
            duration_seconds=0,
        )
        if self._info.fps > 0 and self._info.frame_count > 0:
            self._info.duration_seconds = self._info.frame_count / self._info.fps

    def _get_fourcc_string(self) -> str:
        """Get codec FourCC string from CPU reader."""
        if not self._cpu_reader:
            return "unknown"
        fourcc = int(self._cpu_reader.get(cv2.CAP_PROP_FOURCC))
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    @property
    def info(self) -> Optional[VideoInfo]:
        """Get video metadata."""
        return self._info

    @property
    def is_open(self) -> bool:
        """Check if video is open."""
        if self._using_nvdec:
            return self._reader is not None
        return self._cpu_reader is not None and self._cpu_reader.isOpened()

    @property
    def using_hardware_decode(self) -> bool:
        """Check if using NVDEC hardware decode."""
        return self._using_nvdec

    def read(self) -> tuple[bool, Optional[cv2.cuda.GpuMat]]:
        """Read next frame as GpuMat.

        Returns:
            Tuple of (success, gpu_frame). gpu_frame is None if read failed.
            Frame is already in GPU memory if using NVDEC, uploaded if CPU.
        """
        if self._using_nvdec:
            return self._read_nvdec()
        else:
            return self._read_cpu()

    def _read_nvdec(self) -> tuple[bool, Optional[cv2.cuda.GpuMat]]:
        """Read frame using NVDEC."""
        if not self._reader:
            return False, None

        try:
            ret, gpu_frame = self._reader.nextFrame()
            if ret:
                self._frame_count += 1
            return ret, gpu_frame if ret else None
        except Exception as e:
            logger.warning(f"NVDEC read error: {e}")
            return False, None

    def _read_cpu(self) -> tuple[bool, Optional[cv2.cuda.GpuMat]]:
        """Read frame using CPU decode, upload to GPU."""
        if not self._cpu_reader:
            return False, None

        ret, frame = self._cpu_reader.read()
        if not ret:
            return False, None

        # Upload to GPU
        gpu_frame = cv2.cuda.GpuMat()
        gpu_frame.upload(frame)

        self._frame_count += 1
        return True, gpu_frame

    def read_cpu(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read next frame as CPU numpy array.

        Useful when GPU processing is not needed.

        Returns:
            Tuple of (success, frame).
        """
        if self._using_nvdec:
            ret, gpu_frame = self._read_nvdec()
            if ret and gpu_frame is not None:
                return True, gpu_frame.download()
            return False, None
        else:
            if not self._cpu_reader:
                return False, None
            ret, frame = self._cpu_reader.read()
            if ret:
                self._frame_count += 1
            return ret, frame

    def frames(
        self,
        max_frames: Optional[int] = None,
        skip_frames: int = 0,
    ) -> Iterator[cv2.cuda.GpuMat]:
        """Iterate over frames as GpuMat.

        Args:
            max_frames: Maximum frames to yield (None = all)
            skip_frames: Skip N frames between yields (for subsampling)

        Yields:
            GpuMat frames
        """
        if not self.is_open:
            if not self.open():
                return

        frames_yielded = 0
        skip_counter = 0

        while True:
            if max_frames is not None and frames_yielded >= max_frames:
                break

            ret, gpu_frame = self.read()
            if not ret:
                break

            # Handle frame skipping
            if skip_counter > 0:
                skip_counter -= 1
                continue

            yield gpu_frame
            frames_yielded += 1
            skip_counter = skip_frames

    def frames_cpu(
        self,
        max_frames: Optional[int] = None,
        skip_frames: int = 0,
    ) -> Iterator[np.ndarray]:
        """Iterate over frames as numpy arrays.

        Args:
            max_frames: Maximum frames to yield
            skip_frames: Skip N frames between yields

        Yields:
            Numpy array frames (BGR)
        """
        if not self.is_open:
            if not self.open():
                return

        frames_yielded = 0
        skip_counter = 0

        while True:
            if max_frames is not None and frames_yielded >= max_frames:
                break

            ret, frame = self.read_cpu()
            if not ret:
                break

            if skip_counter > 0:
                skip_counter -= 1
                continue

            yield frame
            frames_yielded += 1
            skip_counter = skip_frames

    def seek(self, frame_number: int) -> bool:
        """Seek to a specific frame (CPU decode only).

        Args:
            frame_number: Frame number to seek to

        Returns:
            True if seek succeeded
        """
        if self._using_nvdec:
            logger.warning("Seek not supported with NVDEC decode")
            return False

        if self._cpu_reader:
            return self._cpu_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return False

    def close(self) -> None:
        """Close video source and release resources."""
        if self._reader:
            # NVDEC reader doesn't have explicit close
            self._reader = None

        if self._cpu_reader:
            self._cpu_reader.release()
            self._cpu_reader = None

        self._frame_count = 0

    def __enter__(self) -> "NVDECExtractor":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class NVDECExtractorPool:
    """Pool of NVDEC extractors for multiple video sources.

    Manages multiple video decoders, respecting NVDEC session limits
    and providing fallback to CPU for overflow.
    """

    def __init__(
        self,
        max_nvdec_sessions: int = 3,
        device_id: int = 0,
    ):
        """Initialize extractor pool.

        Args:
            max_nvdec_sessions: Maximum concurrent NVDEC sessions
            device_id: CUDA device ID
        """
        self.max_nvdec_sessions = max_nvdec_sessions
        self.device_id = device_id
        self._extractors: dict[str, NVDECExtractor] = {}
        self._nvdec_count = 0

    def get_extractor(
        self,
        source: Union[str, Path],
        is_stream: bool = False,
    ) -> NVDECExtractor:
        """Get or create an extractor for a source.

        Args:
            source: Video file path or RTSP URL
            is_stream: True if source is live stream

        Returns:
            NVDECExtractor for the source
        """
        key = str(source)

        if key not in self._extractors:
            # Force CPU if NVDEC sessions exhausted
            force_cpu = self._nvdec_count >= self.max_nvdec_sessions

            extractor = NVDECExtractor(
                source=source,
                is_stream=is_stream,
                device_id=self.device_id,
                fallback_to_cpu=True,
            )

            if force_cpu:
                extractor._using_nvdec = False
                extractor._try_cpu()
            else:
                extractor.open()
                if extractor.using_hardware_decode:
                    self._nvdec_count += 1

            self._extractors[key] = extractor

        return self._extractors[key]

    def close_all(self) -> None:
        """Close all extractors."""
        for extractor in self._extractors.values():
            extractor.close()
        self._extractors.clear()
        self._nvdec_count = 0

    @property
    def stats(self) -> dict:
        """Get pool statistics."""
        nvdec_active = sum(1 for e in self._extractors.values() if e.using_hardware_decode)
        cpu_active = len(self._extractors) - nvdec_active
        return {
            "total_extractors": len(self._extractors),
            "nvdec_active": nvdec_active,
            "cpu_active": cpu_active,
            "nvdec_limit": self.max_nvdec_sessions,
        }
