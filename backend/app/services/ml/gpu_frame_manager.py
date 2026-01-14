"""GPU frame memory management for efficient VRAM utilization.

This module provides pooled GpuMat management to avoid repeated memory
allocations and minimize PCIe transfer overhead. Key features:

- GpuMat pooling: Reuse GPU memory allocations
- Stream management: Async upload/download operations
- Zero-copy sharing: Share frames between pipeline stages
- Memory tracking: Monitor VRAM usage

Memory layout for a typical 1080p frame:
- BGR frame: 1920 x 1080 x 3 = ~6.2MB
- Grayscale: 1920 x 1080 = ~2.1MB
- Total per camera (with buffers): ~20MB

For 16 cameras, expect ~320MB base + model memory.
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GPUMemoryStats:
    """GPU memory usage statistics."""

    total_bytes: int = 0
    free_bytes: int = 0
    used_bytes: int = 0
    pool_allocated: int = 0
    pool_in_use: int = 0

    @property
    def used_percent(self) -> float:
        """Percentage of GPU memory in use."""
        if self.total_bytes == 0:
            return 0.0
        return (self.used_bytes / self.total_bytes) * 100


@dataclass
class PooledGpuMat:
    """A GpuMat wrapper with pooling metadata."""

    mat: cv2.cuda.GpuMat
    size: tuple[int, int]  # (height, width)
    channels: int
    dtype: int
    pool_key: str
    in_use: bool = False
    frame_id: int = 0


class GpuMatPool:
    """Pool of reusable GpuMat objects to minimize GPU memory allocations.

    GPU memory allocation is expensive (~1ms per allocation). This pool
    maintains pre-allocated GpuMat objects that can be reused, reducing
    allocation overhead to near-zero for steady-state operation.

    Thread Safety:
        The pool uses a lock to ensure thread-safe access. However, the
        GpuMat objects themselves are not thread-safe - each should only
        be used by one thread at a time.

    Example:
        >>> pool = GpuMatPool()
        >>> with pool.get(height=1080, width=1920, channels=3) as gpu_mat:
        ...     gpu_mat.upload(frame)
        ...     # Use gpu_mat for processing
        >>> # gpu_mat is automatically returned to pool
    """

    def __init__(self, max_size: int = 100):
        """Initialize GPU memory pool.

        Args:
            max_size: Maximum number of GpuMat objects to keep in pool
        """
        self.max_size = max_size
        self._pool: dict[str, list[PooledGpuMat]] = defaultdict(list)
        self._lock = threading.Lock()
        self._total_allocated = 0
        self._frame_counter = 0

    @staticmethod
    def _make_key(height: int, width: int, channels: int, dtype: int) -> str:
        """Create a pool key from frame dimensions."""
        return f"{height}x{width}x{channels}:{dtype}"

    def acquire(
        self,
        height: int,
        width: int,
        channels: int = 3,
        dtype: int = cv2.CV_8UC3,
    ) -> PooledGpuMat:
        """Acquire a GpuMat from the pool, creating one if necessary.

        Args:
            height: Frame height
            width: Frame width
            channels: Number of channels (default 3 for BGR)
            dtype: OpenCV data type (default CV_8UC3)

        Returns:
            PooledGpuMat wrapper around a GpuMat
        """
        key = self._make_key(height, width, channels, dtype)

        with self._lock:
            self._frame_counter += 1

            # Try to get from pool
            if self._pool[key]:
                pooled = self._pool[key].pop()
                pooled.in_use = True
                pooled.frame_id = self._frame_counter
                return pooled

            # Create new GpuMat
            mat = cv2.cuda.GpuMat(height, width, dtype)
            self._total_allocated += 1

            pooled = PooledGpuMat(
                mat=mat,
                size=(height, width),
                channels=channels,
                dtype=dtype,
                pool_key=key,
                in_use=True,
                frame_id=self._frame_counter,
            )

            if self._total_allocated % 10 == 0:
                logger.debug(f"GpuMatPool: {self._total_allocated} total allocations")

            return pooled

    def release(self, pooled: PooledGpuMat) -> None:
        """Return a GpuMat to the pool.

        Args:
            pooled: The PooledGpuMat to return
        """
        with self._lock:
            pooled.in_use = False

            # Only keep up to max_size items per key
            if len(self._pool[pooled.pool_key]) < self.max_size:
                self._pool[pooled.pool_key].append(pooled)

    def get(
        self,
        height: int,
        width: int,
        channels: int = 3,
        dtype: int = cv2.CV_8UC3,
    ) -> "PooledGpuMatContext":
        """Get a pooled GpuMat as a context manager.

        Usage:
            with pool.get(1080, 1920) as gpu_mat:
                gpu_mat.upload(frame)
                # process...
            # Automatically returned to pool

        Args:
            height: Frame height
            width: Frame width
            channels: Number of channels
            dtype: OpenCV data type

        Returns:
            Context manager yielding a PooledGpuMat
        """
        return PooledGpuMatContext(self, height, width, channels, dtype)

    def clear(self) -> None:
        """Clear all pooled GpuMat objects.

        Call this to free GPU memory when the pool is no longer needed.
        """
        with self._lock:
            self._pool.clear()
            self._total_allocated = 0
            logger.info("GpuMatPool: Cleared all pooled memory")

    def stats(self) -> dict:
        """Get pool statistics."""
        with self._lock:
            total_pooled = sum(len(mats) for mats in self._pool.values())
            return {
                "total_allocated": self._total_allocated,
                "pooled_available": total_pooled,
                "pool_keys": list(self._pool.keys()),
            }


class PooledGpuMatContext:
    """Context manager for automatic GpuMat pool release."""

    def __init__(
        self,
        pool: GpuMatPool,
        height: int,
        width: int,
        channels: int,
        dtype: int,
    ):
        self.pool = pool
        self.height = height
        self.width = width
        self.channels = channels
        self.dtype = dtype
        self._pooled: Optional[PooledGpuMat] = None

    def __enter__(self) -> cv2.cuda.GpuMat:
        self._pooled = self.pool.acquire(
            self.height, self.width, self.channels, self.dtype
        )
        return self._pooled.mat

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._pooled:
            self.pool.release(self._pooled)


class GPUFrameManager:
    """Manages GPU frames and memory for a processing pipeline.

    Provides centralized management of:
    - GpuMat pooling
    - CUDA stream management
    - Frame upload/download
    - Memory monitoring

    Example:
        >>> manager = GPUFrameManager()
        >>> gpu_frame = manager.upload(camera_id=1, frame=bgr_frame)
        >>> # Process gpu_frame...
        >>> manager.release(camera_id=1)
    """

    def __init__(self, pool_size: int = 50):
        """Initialize frame manager.

        Args:
            pool_size: Maximum GpuMat objects per size in pool
        """
        self._pool = GpuMatPool(max_size=pool_size)

        # Per-camera frame storage
        self._camera_frames: dict[int, PooledGpuMat] = {}

        # Per-camera CUDA streams for async operations
        self._streams: dict[int, cv2.cuda_Stream] = {}

        self._lock = threading.Lock()

    def get_stream(self, camera_id: int) -> cv2.cuda_Stream:
        """Get or create a CUDA stream for a camera.

        Using per-camera streams allows parallel processing of
        multiple cameras.

        Args:
            camera_id: Camera identifier

        Returns:
            CUDA stream for this camera
        """
        with self._lock:
            if camera_id not in self._streams:
                self._streams[camera_id] = cv2.cuda_Stream()
            return self._streams[camera_id]

    def upload(
        self,
        camera_id: int,
        frame: np.ndarray,
        stream: Optional[cv2.cuda_Stream] = None,
    ) -> cv2.cuda.GpuMat:
        """Upload a frame to GPU memory.

        Args:
            camera_id: Camera identifier
            frame: BGR frame as numpy array
            stream: Optional CUDA stream (uses camera's stream if not provided)

        Returns:
            GpuMat containing the uploaded frame
        """
        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) == 3 else 1

        # Determine dtype
        if frame.dtype == np.uint8:
            if channels == 3:
                dtype = cv2.CV_8UC3
            elif channels == 1:
                dtype = cv2.CV_8UC1
            else:
                dtype = cv2.CV_8UC(channels)
        else:
            dtype = cv2.CV_32FC3 if channels == 3 else cv2.CV_32FC1

        # Release previous frame if any
        self.release(camera_id)

        # Acquire from pool
        with self._lock:
            pooled = self._pool.acquire(height, width, channels, dtype)
            self._camera_frames[camera_id] = pooled

        # Get stream
        cuda_stream = stream or self.get_stream(camera_id)

        # Upload asynchronously
        pooled.mat.upload(frame, cuda_stream)

        return pooled.mat

    def get_frame(self, camera_id: int) -> Optional[cv2.cuda.GpuMat]:
        """Get the current GPU frame for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            GpuMat if frame exists, None otherwise
        """
        with self._lock:
            pooled = self._camera_frames.get(camera_id)
            return pooled.mat if pooled else None

    def download(
        self,
        camera_id: int,
        stream: Optional[cv2.cuda_Stream] = None,
    ) -> Optional[np.ndarray]:
        """Download a camera's frame from GPU to CPU.

        Args:
            camera_id: Camera identifier
            stream: Optional CUDA stream

        Returns:
            Numpy array if frame exists, None otherwise
        """
        with self._lock:
            pooled = self._camera_frames.get(camera_id)
            if not pooled:
                return None

        cuda_stream = stream or self.get_stream(camera_id)
        return pooled.mat.download(cuda_stream)

    def release(self, camera_id: int) -> None:
        """Release GPU memory for a camera.

        Args:
            camera_id: Camera identifier
        """
        with self._lock:
            pooled = self._camera_frames.pop(camera_id, None)
            if pooled:
                self._pool.release(pooled)

    def release_all(self) -> None:
        """Release all GPU frames."""
        with self._lock:
            for pooled in self._camera_frames.values():
                self._pool.release(pooled)
            self._camera_frames.clear()

    def clear_pool(self) -> None:
        """Clear the entire GPU memory pool."""
        self.release_all()
        self._pool.clear()

    def get_stats(self) -> dict:
        """Get memory usage statistics.

        Returns:
            Dictionary with pool and frame statistics
        """
        with self._lock:
            return {
                "active_cameras": len(self._camera_frames),
                "active_streams": len(self._streams),
                "pool": self._pool.stats(),
            }


# Global frame manager instance
gpu_frame_manager = GPUFrameManager()


def get_gpu_memory_info() -> GPUMemoryStats:
    """Get GPU memory usage information.

    Returns:
        GPUMemoryStats with current memory state
    """
    try:
        # Get device info using OpenCV CUDA
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        if device_count == 0:
            return GPUMemoryStats()

        # Note: OpenCV doesn't provide direct memory query
        # For detailed stats, we'd need pycuda or nvidia-ml-py
        # This is a simplified version
        return GPUMemoryStats(
            total_bytes=0,  # Would need nvidia-ml-py
            free_bytes=0,
            used_bytes=0,
            pool_allocated=gpu_frame_manager.get_stats()["pool"]["total_allocated"],
            pool_in_use=gpu_frame_manager.get_stats()["active_cameras"],
        )
    except Exception as e:
        logger.warning(f"Could not get GPU memory info: {e}")
        return GPUMemoryStats()
