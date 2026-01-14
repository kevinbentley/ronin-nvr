"""Motion gate for live detection using frame differencing or GPU MOG2.

This module provides motion detection mechanisms to determine if YOLO
inference should be run. Two backends are available:

1. CPU Frame Differencing (default):
   - Fast (~8ms per frame at 720p)
   - Works well with sparse frames
   - Simple but prone to false positives

2. GPU MOG2 Background Subtraction:
   - Faster (~2-3ms per frame at 720p with GPU)
   - Maintains temporal background model
   - Better rejection of rain, shadows, lighting changes
   - Requires CUDA-enabled OpenCV

The gate is designed to skip YOLO inference (~121ms) on static scenes,
providing significant performance benefits.

Also includes corruption detection to filter out false motion from
video decode errors (vertical banding from UDP packet loss).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_frame_corruption(
    frame: np.ndarray,
    repeated_col_threshold: float = 0.5,
    mean_diff_threshold: float = 0.5,
) -> bool:
    """Detect if a frame has vertical banding corruption from decode errors.

    UDP packet loss causes the video decoder to repeat the last successfully
    decoded row of pixels for all remaining rows in affected macroblocks.
    This creates columns where pixel values are nearly identical down their
    length - something that virtually never happens in natural scenes.

    Detection method:
    1. Focus on bottom 40% of frame (where corruption typically manifests)
    2. Compute row-to-row pixel differences for each column
    3. Corrupted columns have mean diff ≈ 0 (same value repeated down column)
    4. Flag as corrupt if >50% of columns show this pattern

    Args:
        frame: BGR image to check
        repeated_col_threshold: Fraction of columns with near-zero row diffs
                               needed to flag as corrupt (default 0.5 = 50%)
        mean_diff_threshold: Maximum mean row diff to consider a column as
                            "repeated" (default 0.5 pixels)

    Returns:
        True if corruption detected, False otherwise
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    height, width = gray.shape[:2]

    # Focus on bottom 40% where corruption typically appears
    bottom_start = int(height * 0.6)
    bottom = gray[bottom_start:, :]

    # Compute row-to-row differences for each column
    # In corrupted regions, the same pixel value repeats down the column
    # so these differences are ~0
    row_diffs = np.abs(np.diff(bottom.astype(float), axis=0))

    # Mean difference per column
    col_mean_diff = np.mean(row_diffs, axis=0)

    # Count columns with very small row differences (repeated pixels)
    repeated_cols = np.sum(col_mean_diff < mean_diff_threshold)
    repeated_pct = repeated_cols / width

    # Overall mean - corrupted images have very low overall mean
    overall_mean = np.mean(col_mean_diff)

    # Detection criteria:
    # - More than 50% of columns have repeated pixels, OR
    # - Overall mean row-diff is very low (strong corruption signal)
    return repeated_pct > repeated_col_threshold or overall_mean < mean_diff_threshold


@dataclass
class MotionGateResult:
    """Result from motion gate check."""

    should_run_inference: bool  # True if YOLO should run
    motion_detected: bool  # Whether motion was detected
    motion_percent: float  # Percentage of frame with motion (0-100)
    contour_count: int  # Number of motion regions found
    reason: str  # Human-readable reason for decision
    frame_corrupted: bool = False  # True if frame corruption detected


class MotionGate:
    """Gate that decides whether to run YOLO inference based on motion.

    Uses simple frame differencing which is:
    - Fast (~8ms per frame at 720p)
    - Works with sparse frames (no background model needed)
    - Robust for the purpose of gating inference

    Algorithm:
    1. Convert both frames to grayscale
    2. Apply Gaussian blur (21x21) to reduce noise
    3. Compute absolute pixel difference
    4. Threshold: pixels with diff > threshold marked as "changed"
    5. Calculate motion_percent = changed_pixels / total_pixels * 100
    6. Find contours (connected regions of change)
    7. Filter contours by minimum area
    8. Motion detected if motion_percent >= min_percent AND valid contours exist
    """

    def __init__(
        self,
        threshold: float = 25.0,
        min_area: int = 500,
        min_percent: float = 0.1,
        blur_size: int = 21,
    ):
        """Initialize motion gate.

        Args:
            threshold: Pixel difference threshold (0-255). Higher = less sensitive.
            min_area: Minimum contour area in pixels to count as motion.
            min_percent: Minimum percentage of frame that must change (0-100).
            blur_size: Gaussian blur kernel size (must be odd).
        """
        self.threshold = threshold
        self.min_area = min_area
        self.min_percent = min_percent
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1

        # Morphological kernel for dilation
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def check(
        self,
        current_frame: np.ndarray,
        previous_frame: Optional[np.ndarray],
    ) -> MotionGateResult:
        """Check if motion is detected between frames.

        Args:
            current_frame: Current BGR frame (numpy array)
            previous_frame: Previous BGR frame (None for first frame)

        Returns:
            MotionGateResult with decision and details
        """
        # Check for frame corruption (vertical banding from decode errors)
        is_corrupted = detect_frame_corruption(current_frame)
        if is_corrupted:
            logger.debug("Frame corruption detected, skipping motion detection")
            return MotionGateResult(
                should_run_inference=False,
                motion_detected=False,
                motion_percent=0.0,
                contour_count=0,
                reason="frame_corrupted",
                frame_corrupted=True,
            )

        # First frame: always run YOLO (no previous to compare)
        if previous_frame is None:
            return MotionGateResult(
                should_run_inference=True,
                motion_detected=False,
                motion_percent=0.0,
                contour_count=0,
                reason="first_frame",
            )

        # Size mismatch: resolution changed, run YOLO
        if current_frame.shape != previous_frame.shape:
            return MotionGateResult(
                should_run_inference=True,
                motion_detected=True,
                motion_percent=100.0,
                contour_count=1,
                reason="frame_size_changed",
            )

        # Convert to grayscale
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray_current = cv2.GaussianBlur(
            gray_current, (self.blur_size, self.blur_size), 0
        )
        gray_previous = cv2.GaussianBlur(
            gray_previous, (self.blur_size, self.blur_size), 0
        )

        # Compute absolute difference
        diff = cv2.absdiff(gray_current, gray_previous)

        # Threshold
        _, thresh = cv2.threshold(
            diff, int(self.threshold), 255, cv2.THRESH_BINARY
        )

        # Calculate motion percentage before morphological operations
        height, width = current_frame.shape[:2]
        total_pixels = height * width
        motion_pixels = cv2.countNonZero(thresh)
        motion_percent = (motion_pixels / total_pixels) * 100

        # Dilate to connect nearby regions (helps with fragmented motion)
        thresh = cv2.dilate(thresh, self._kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by minimum area
        valid_contours = [
            c for c in contours if cv2.contourArea(c) >= self.min_area
        ]

        # Motion detected if percentage threshold met AND valid contours exist
        motion_detected = (
            motion_percent >= self.min_percent and len(valid_contours) > 0
        )

        return MotionGateResult(
            should_run_inference=motion_detected,
            motion_detected=motion_detected,
            motion_percent=round(motion_percent, 2),
            contour_count=len(valid_contours),
            reason="motion_detected" if motion_detected else "no_motion",
        )

    def get_bounding_boxes(
        self,
        current_frame: np.ndarray,
        previous_frame: np.ndarray,
    ) -> list[tuple[float, float, float, float]]:
        """Get normalized bounding boxes for motion regions.

        Used when saving motion detections to database.

        Args:
            current_frame: Current BGR frame
            previous_frame: Previous BGR frame

        Returns:
            List of (x, y, width, height) tuples in normalized coordinates (0-1)
        """
        if previous_frame is None or current_frame.shape != previous_frame.shape:
            return [(0.0, 0.0, 1.0, 1.0)]  # Full frame

        height, width = current_frame.shape[:2]

        # Same processing as check()
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

        gray_current = cv2.GaussianBlur(
            gray_current, (self.blur_size, self.blur_size), 0
        )
        gray_previous = cv2.GaussianBlur(
            gray_previous, (self.blur_size, self.blur_size), 0
        )

        diff = cv2.absdiff(gray_current, gray_previous)
        _, thresh = cv2.threshold(diff, int(self.threshold), 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, self._kernel, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Normalize to 0-1 range
                bboxes.append((
                    x / width,
                    y / height,
                    w / width,
                    h / height,
                ))

        # Return full frame if no valid contours
        return bboxes if bboxes else [(0.0, 0.0, 1.0, 1.0)]


def create_motion_gate(
    backend: str = "cpu",
    **kwargs,
) -> "MotionGate":
    """Factory function to create a motion gate with the specified backend.

    Args:
        backend: Backend to use - "cpu" for frame differencing, "gpu" for MOG2
        **kwargs: Additional arguments passed to the gate constructor

    Returns:
        MotionGate instance (CPU or GPU variant)

    Raises:
        ValueError: If backend is not recognized
        RuntimeError: If GPU backend requested but CUDA not available
    """
    if backend == "cpu":
        return MotionGate(**kwargs)

    elif backend == "gpu":
        try:
            import cv2

            if cv2.cuda.getCudaEnabledDeviceCount() == 0:
                raise RuntimeError("GPU backend requested but no CUDA devices available")

            from app.services.ml.gpu_motion import GPUMotionGate

            # Map CPU params to GPU params where applicable
            gpu_kwargs = {}
            if "min_percent" in kwargs:
                gpu_kwargs["min_motion_percent"] = kwargs["min_percent"]
            if "min_area" in kwargs:
                gpu_kwargs["min_contour_area"] = kwargs["min_area"]
            if "threshold" in kwargs:
                # CPU threshold maps roughly to MOG2 var_threshold
                gpu_kwargs["var_threshold"] = kwargs["threshold"]

            return GPUMotionGate(**gpu_kwargs)

        except ImportError as e:
            raise RuntimeError(f"GPU motion gate not available: {e}") from e

    else:
        raise ValueError(f"Unknown motion gate backend: {backend}")
