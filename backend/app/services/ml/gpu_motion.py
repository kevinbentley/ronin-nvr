"""GPU-accelerated motion detection using CUDA MOG2 background subtraction.

This module provides GPU-accelerated background subtraction using OpenCV's
CUDA implementation of the MOG2 (Mixture of Gaussians) algorithm. Unlike
simple frame differencing, MOG2 maintains a temporal model of the background
which provides:

- Better handling of gradual lighting changes
- Shadow detection and suppression
- Reduced false positives from rain/snow/leaves
- More robust motion segmentation

The GPU implementation is significantly faster than CPU (~10x speedup) and
keeps all processing in VRAM to minimize PCIe transfers.

Key parameters tuned for surveillance:
- history=500: ~17 seconds at 30fps for background learning
- varThreshold=16: Balance between sensitivity and noise rejection
- detectShadows=True: Distinguish shadows from actual motion
- Morphological filtering: Removes rain speckles and noise
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GPUMotionResult:
    """Result from GPU motion detection."""

    motion_detected: bool
    motion_percent: float  # Percentage of frame with motion (0-100)
    motion_mask: Optional[np.ndarray]  # Downloaded mask for visualization
    contour_count: int
    largest_contour_area: float  # Normalized 0-1


class GPUBackgroundSubtractor:
    """GPU-accelerated MOG2 background subtractor for motion detection.

    Uses CUDA-accelerated MOG2 with morphological filtering to detect
    motion while rejecting noise from rain, shadows, and lighting changes.

    Thread Safety:
        Each camera should have its own instance. The background model
        is camera-specific and not thread-safe.

    Memory:
        Maintains GPU memory for:
        - Background model (~50MB for 1080p)
        - Morphological kernels
        - Intermediate processing buffers

    Example:
        >>> subtractor = GPUBackgroundSubtractor(camera_id=1)
        >>> for frame in video_frames:
        ...     gpu_frame = cv2.cuda_GpuMat()
        ...     gpu_frame.upload(frame)
        ...     result = subtractor.apply(gpu_frame)
        ...     if result.motion_detected:
        ...         run_detection(frame)
    """

    def __init__(
        self,
        camera_id: int,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        learning_rate: float = -1.0,
        min_motion_percent: float = 0.1,
        min_contour_area: int = 500,
        erosion_kernel_size: int = 3,
        dilation_kernel_size: int = 5,
        shadow_threshold: int = 127,
    ):
        """Initialize GPU background subtractor.

        Args:
            camera_id: Camera identifier for logging
            history: Number of frames for background model (default 500 = ~17s at 30fps)
            var_threshold: Variance threshold for foreground detection (lower=more sensitive)
            detect_shadows: Whether to detect and mark shadows (value 127 in mask)
            learning_rate: Background learning rate (-1 = auto, 0 = never update, 1 = instant)
            min_motion_percent: Minimum % of frame that must change to trigger motion
            min_contour_area: Minimum contour area in pixels to count as valid motion
            erosion_kernel_size: Size of erosion kernel (removes small noise/rain)
            dilation_kernel_size: Size of dilation kernel (connects nearby regions)
            shadow_threshold: Value below which shadow pixels are excluded (127=shadow)
        """
        self.camera_id = camera_id
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.learning_rate = learning_rate
        self.min_motion_percent = min_motion_percent
        self.min_contour_area = min_contour_area
        self.shadow_threshold = shadow_threshold

        # Create CUDA MOG2 background subtractor
        self._mog2 = cv2.cuda.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

        # Create CUDA stream for async operations
        self._stream = cv2.cuda_Stream()

        # Create morphological kernels (erosion removes noise, dilation connects regions)
        self._erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size)
        )
        self._dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size)
        )

        # Create GPU morphology filters
        self._erode_filter = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_ERODE, cv2.CV_8UC1, self._erosion_kernel
        )
        self._dilate_filter = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_DILATE, cv2.CV_8UC1, self._dilation_kernel
        )

        # GPU mat for intermediate results
        self._gpu_gray: Optional[cv2.cuda.GpuMat] = None
        self._gpu_mask: Optional[cv2.cuda.GpuMat] = None
        self._gpu_mask_filtered: Optional[cv2.cuda.GpuMat] = None

        self._frame_count = 0
        self._initialized = False

        logger.info(
            f"GPUBackgroundSubtractor initialized for camera {camera_id}: "
            f"history={history}, varThreshold={var_threshold}, "
            f"erosion={erosion_kernel_size}x{erosion_kernel_size}, "
            f"dilation={dilation_kernel_size}x{dilation_kernel_size}"
        )

    def apply(
        self,
        gpu_frame: cv2.cuda.GpuMat,
        learning_rate: Optional[float] = None,
        download_mask: bool = False,
    ) -> GPUMotionResult:
        """Apply background subtraction to detect motion.

        Args:
            gpu_frame: Input frame as GpuMat (BGR or grayscale)
            learning_rate: Override default learning rate for this frame
            download_mask: Whether to download mask to CPU (for visualization)

        Returns:
            GPUMotionResult with motion detection results
        """
        lr = learning_rate if learning_rate is not None else self.learning_rate
        self._frame_count += 1

        # Convert to grayscale if needed (MOG2 works on single channel)
        if gpu_frame.channels() == 3:
            self._gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            input_mat = self._gpu_gray
        else:
            input_mat = gpu_frame

        # Apply MOG2 background subtraction
        if self._gpu_mask is None:
            self._gpu_mask = cv2.cuda.GpuMat()
        self._gpu_mask = self._mog2.apply(input_mat, lr, self._stream)

        # Wait for MOG2 to complete before morphology
        self._stream.waitForCompletion()

        # Apply morphological filtering to remove noise
        # Erosion removes small noise (rain drops, sensor noise)
        eroded = self._erode_filter.apply(self._gpu_mask)

        # Dilation connects nearby motion regions
        self._gpu_mask_filtered = self._dilate_filter.apply(eroded)

        # Download mask to CPU for analysis
        mask = self._gpu_mask_filtered.download()

        # Remove shadow pixels (value 127) - only keep definite foreground (255)
        if self.detect_shadows:
            mask = np.where(mask > self.shadow_threshold, 255, 0).astype(np.uint8)

        # Calculate motion percentage
        total_pixels = mask.shape[0] * mask.shape[1]
        motion_pixels = cv2.countNonZero(mask)
        motion_percent = (motion_pixels / total_pixels) * 100

        # Find contours for more detailed analysis
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by minimum area
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]

        # Find largest contour area (normalized)
        largest_area = 0.0
        if valid_contours:
            largest_area = max(cv2.contourArea(c) for c in valid_contours) / total_pixels

        # Motion is detected if:
        # 1. Motion percentage exceeds threshold AND
        # 2. At least one valid contour exists
        motion_detected = (
            motion_percent >= self.min_motion_percent and len(valid_contours) > 0
        )

        # Mark as initialized after first few frames
        if self._frame_count >= 30 and not self._initialized:
            self._initialized = True
            logger.debug(f"Camera {self.camera_id}: Background model initialized")

        return GPUMotionResult(
            motion_detected=motion_detected,
            motion_percent=round(motion_percent, 2),
            motion_mask=mask if download_mask else None,
            contour_count=len(valid_contours),
            largest_contour_area=round(largest_area, 4),
        )

    def get_motion_regions(
        self,
        gpu_frame: cv2.cuda.GpuMat,
    ) -> list[tuple[float, float, float, float]]:
        """Get bounding boxes of motion regions in normalized coordinates.

        Useful for focusing detection on regions of interest.

        Args:
            gpu_frame: Input frame as GpuMat

        Returns:
            List of (x, y, width, height) tuples normalized to 0-1
        """
        result = self.apply(gpu_frame, download_mask=False)

        if not result.motion_detected or result.motion_mask is None:
            return []

        mask = result.motion_mask
        height, width = mask.shape[:2]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append((
                    x / width,
                    y / height,
                    w / width,
                    h / height,
                ))

        return regions

    def reset(self) -> None:
        """Reset the background model.

        Call this when camera view changes significantly (e.g., PTZ movement).
        """
        self._mog2 = cv2.cuda.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )
        self._frame_count = 0
        self._initialized = False
        logger.info(f"Camera {self.camera_id}: Background model reset")

    @property
    def is_initialized(self) -> bool:
        """Whether the background model has been initialized with enough frames."""
        return self._initialized

    @property
    def frame_count(self) -> int:
        """Number of frames processed."""
        return self._frame_count


class GPUMotionGate:
    """Motion gate using GPU background subtraction.

    Drop-in replacement for the CPU MotionGate class, using GPU-accelerated
    MOG2 for better accuracy and performance.

    This class manages a pool of GPUBackgroundSubtractor instances,
    one per camera, and handles GpuMat upload/download.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        min_motion_percent: float = 0.1,
        min_contour_area: int = 500,
    ):
        """Initialize GPU motion gate.

        Args:
            history: Background model history length
            var_threshold: MOG2 variance threshold
            min_motion_percent: Minimum motion percentage to trigger
            min_contour_area: Minimum contour area for valid motion
        """
        self.history = history
        self.var_threshold = var_threshold
        self.min_motion_percent = min_motion_percent
        self.min_contour_area = min_contour_area

        # Per-camera subtractor instances
        self._subtractors: dict[int, GPUBackgroundSubtractor] = {}

        # Shared GpuMat for frame upload
        self._gpu_frame = cv2.cuda.GpuMat()

    def get_subtractor(self, camera_id: int) -> GPUBackgroundSubtractor:
        """Get or create subtractor for a camera."""
        if camera_id not in self._subtractors:
            self._subtractors[camera_id] = GPUBackgroundSubtractor(
                camera_id=camera_id,
                history=self.history,
                var_threshold=self.var_threshold,
                min_motion_percent=self.min_motion_percent,
                min_contour_area=self.min_contour_area,
            )
        return self._subtractors[camera_id]

    def check(
        self,
        camera_id: int,
        frame: np.ndarray,
        download_mask: bool = False,
    ) -> GPUMotionResult:
        """Check for motion in a frame.

        Args:
            camera_id: Camera identifier
            frame: BGR frame as numpy array
            download_mask: Whether to include mask in result

        Returns:
            GPUMotionResult with motion detection info
        """
        subtractor = self.get_subtractor(camera_id)

        # Upload frame to GPU
        self._gpu_frame.upload(frame)

        # Apply background subtraction
        return subtractor.apply(self._gpu_frame, download_mask=download_mask)

    def check_gpu(
        self,
        camera_id: int,
        gpu_frame: cv2.cuda.GpuMat,
        download_mask: bool = False,
    ) -> GPUMotionResult:
        """Check for motion using a frame already on GPU.

        More efficient when frame is already in GPU memory.

        Args:
            camera_id: Camera identifier
            gpu_frame: Frame as GpuMat
            download_mask: Whether to include mask in result

        Returns:
            GPUMotionResult with motion detection info
        """
        subtractor = self.get_subtractor(camera_id)
        return subtractor.apply(gpu_frame, download_mask=download_mask)

    def reset_camera(self, camera_id: int) -> None:
        """Reset background model for a specific camera."""
        if camera_id in self._subtractors:
            self._subtractors[camera_id].reset()

    def reset_all(self) -> None:
        """Reset background models for all cameras."""
        for subtractor in self._subtractors.values():
            subtractor.reset()

    def get_last_mask(self, camera_id: int) -> Optional[np.ndarray]:
        """Get the last motion mask for a camera.

        Useful for visualization after calling check().

        Args:
            camera_id: Camera identifier

        Returns:
            Motion mask as numpy array, or None if not available
        """
        if camera_id not in self._subtractors:
            return None

        subtractor = self._subtractors[camera_id]

        # Re-download the filtered mask if available
        if subtractor._gpu_mask_filtered is not None:
            mask = subtractor._gpu_mask_filtered.download()
            # Remove shadow pixels
            if subtractor.detect_shadows:
                mask = np.where(mask > subtractor.shadow_threshold, 255, 0).astype(np.uint8)
            return mask

        return None
