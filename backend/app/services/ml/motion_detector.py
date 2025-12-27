"""Motion detection service using background subtraction.

Uses OpenCV's MOG2 (Mixture of Gaussians) algorithm which:
- Adapts to gradual lighting changes
- Handles shadows
- Maintains a background model that updates over time
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class MotionResult:
    """Result from motion detection on a single frame."""

    has_motion: bool
    motion_percent: float  # 0-100, percentage of frame with motion
    contour_count: int  # Number of motion regions
    bounding_boxes: list[tuple[float, float, float, float]]  # Normalized (x, y, w, h)
    largest_area_percent: float  # Size of largest motion region


class MotionDetector:
    """Detect motion in video frames using background subtraction.

    Uses MOG2 algorithm which is robust to:
    - Gradual lighting changes (adapts background model)
    - Shadows (can be configured to detect/ignore)
    - Camera noise

    The detector maintains state (background model) so it should be
    used for processing sequential frames from the same video.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        learning_rate: float = -1,  # -1 = auto
        min_contour_area: int = 500,  # Minimum pixel area to consider
        motion_threshold: float = 0.5,  # Percent of frame for motion detection
        blur_size: int = 21,  # Gaussian blur kernel size
        dilate_iterations: int = 2,  # Morphological dilation iterations
    ):
        """Initialize motion detector.

        Args:
            history: Number of frames for background model history
            var_threshold: Threshold for foreground/background decision
            detect_shadows: Whether to detect and mark shadows
            learning_rate: Background model learning rate (-1 = auto)
            min_contour_area: Minimum contour area in pixels to consider as motion
            motion_threshold: Percentage of frame that must have motion (0-100)
            blur_size: Size of Gaussian blur kernel (must be odd)
            dilate_iterations: Number of dilation iterations for noise reduction
        """
        self.min_contour_area = min_contour_area
        self.motion_threshold = motion_threshold
        self.learning_rate = learning_rate
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        self.dilate_iterations = dilate_iterations

        # Create background subtractor
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

        # Morphological kernels for noise reduction
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self._frame_count = 0
        self._warmup_frames = 30  # Frames needed to build initial background model

    @classmethod
    def from_settings(cls) -> "MotionDetector":
        """Create detector from application settings."""
        settings = get_settings()
        return cls(
            history=settings.motion_history,
            var_threshold=settings.motion_var_threshold,
            detect_shadows=settings.motion_detect_shadows,
            learning_rate=settings.motion_learning_rate,
            min_contour_area=settings.motion_min_contour_area,
            motion_threshold=settings.motion_threshold,
        )

    def reset(self) -> None:
        """Reset the background model.

        Call this when starting a new video or after a scene change.
        """
        settings = get_settings()
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=settings.motion_history,
            varThreshold=settings.motion_var_threshold,
            detectShadows=settings.motion_detect_shadows,
        )
        self._frame_count = 0

    def detect(self, frame: np.ndarray) -> MotionResult:
        """Detect motion in a frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            MotionResult with detection details
        """
        self._frame_count += 1
        height, width = frame.shape[:2]
        total_pixels = height * width

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (self.blur_size, self.blur_size), 0)

        # Apply background subtraction
        fg_mask = self._bg_subtractor.apply(
            blurred,
            learningRate=self.learning_rate,
        )

        # Remove shadows (marked as 127 in MOG2)
        # Keep only definite foreground (255)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to reduce noise
        fg_mask = cv2.erode(fg_mask, self._kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, self._kernel, iterations=self.dilate_iterations)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by minimum area
        valid_contours = [
            c for c in contours if cv2.contourArea(c) >= self.min_contour_area
        ]

        # Calculate motion statistics
        motion_pixels = cv2.countNonZero(fg_mask)
        motion_percent = (motion_pixels / total_pixels) * 100

        # Get bounding boxes (normalized coordinates)
        bounding_boxes: list[tuple[float, float, float, float]] = []
        largest_area = 0

        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            largest_area = max(largest_area, area)

            # Normalize to 0-1 range
            bounding_boxes.append((
                x / width,
                y / height,
                w / width,
                h / height,
            ))

        largest_area_percent = (largest_area / total_pixels) * 100

        # During warmup, don't report motion (background model still learning)
        if self._frame_count < self._warmup_frames:
            return MotionResult(
                has_motion=False,
                motion_percent=0.0,
                contour_count=0,
                bounding_boxes=[],
                largest_area_percent=0.0,
            )

        has_motion = motion_percent >= self.motion_threshold and len(valid_contours) > 0

        return MotionResult(
            has_motion=has_motion,
            motion_percent=round(motion_percent, 2),
            contour_count=len(valid_contours),
            bounding_boxes=bounding_boxes,
            largest_area_percent=round(largest_area_percent, 2),
        )

    def get_debug_frame(self, frame: np.ndarray) -> np.ndarray:
        """Get a debug visualization frame showing motion detection.

        Args:
            frame: Original BGR frame

        Returns:
            Frame with motion regions highlighted
        """
        result = self.detect(frame)

        # Draw bounding boxes on frame copy
        debug_frame = frame.copy()
        height, width = frame.shape[:2]

        for (x, y, w, h) in result.bounding_boxes:
            # Convert from normalized to pixel coordinates
            px = int(x * width)
            py = int(y * height)
            pw = int(w * width)
            ph = int(h * height)

            cv2.rectangle(debug_frame, (px, py), (px + pw, py + ph), (0, 255, 0), 2)

        # Add text overlay
        text = f"Motion: {result.motion_percent:.1f}% ({result.contour_count} regions)"
        cv2.putText(
            debug_frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        return debug_frame


# Convenience function for one-shot detection (creates new detector each time)
def detect_motion_simple(
    frame1: np.ndarray,
    frame2: np.ndarray,
    threshold: float = 25.0,
    min_area: int = 500,
) -> MotionResult:
    """Simple frame-difference motion detection between two frames.

    This is a simpler alternative that doesn't require maintaining state.
    Less robust to lighting changes but useful for quick checks.

    Args:
        frame1: First frame (BGR)
        frame2: Second frame (BGR)
        threshold: Pixel difference threshold (0-255)
        min_area: Minimum contour area in pixels

    Returns:
        MotionResult with detection details
    """
    height, width = frame1.shape[:2]
    total_pixels = height * width

    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)

    # Threshold
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Dilate to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    motion_pixels = cv2.countNonZero(thresh)
    motion_percent = (motion_pixels / total_pixels) * 100

    bounding_boxes = []
    largest_area = 0

    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        largest_area = max(largest_area, area)
        bounding_boxes.append((x / width, y / height, w / width, h / height))

    return MotionResult(
        has_motion=len(valid_contours) > 0,
        motion_percent=round(motion_percent, 2),
        contour_count=len(valid_contours),
        bounding_boxes=bounding_boxes,
        largest_area_percent=round((largest_area / total_pixels) * 100, 2),
    )
