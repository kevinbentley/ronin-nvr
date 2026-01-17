"""Frame mosaic utilities for VLLM activity characterization."""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def create_mosaic(
    frames: list[np.ndarray],
    grid_size: tuple[int, int] = (2, 2),
    target_size: Optional[tuple[int, int]] = None,
    labels: Optional[list[str]] = None,
    border_width: int = 2,
    border_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Create a mosaic image from multiple frames.

    Combines frames into a grid layout for sending to vision LLM.
    Frames are labeled with timestamps or indices for temporal context.

    Args:
        frames: List of BGR numpy arrays (OpenCV format)
        grid_size: Tuple of (rows, cols) for the grid layout
        target_size: Optional (width, height) for output image
        labels: Optional labels for each frame (e.g., timestamps)
        border_width: Width of border between frames
        border_color: BGR color for borders

    Returns:
        Combined mosaic image as BGR numpy array
    """
    if not frames:
        raise ValueError("No frames provided for mosaic")

    rows, cols = grid_size
    expected_count = rows * cols

    if len(frames) > expected_count:
        logger.warning(
            f"Too many frames ({len(frames)}) for {rows}x{cols} grid, "
            f"using first {expected_count}"
        )
        frames = frames[:expected_count]

    # Pad with black frames if not enough
    while len(frames) < expected_count:
        frames.append(np.zeros_like(frames[0]))

    # Get dimensions from first frame
    frame_height, frame_width = frames[0].shape[:2]

    # Calculate cell size
    cell_width = frame_width
    cell_height = frame_height

    # If target size specified, calculate scaling
    if target_size:
        target_width, target_height = target_size
        # Calculate per-cell dimensions
        cell_width = (target_width - border_width * (cols + 1)) // cols
        cell_height = (target_height - border_width * (rows + 1)) // rows

    # Create output canvas
    output_width = cell_width * cols + border_width * (cols + 1)
    output_height = cell_height * rows + border_width * (rows + 1)
    mosaic = np.full(
        (output_height, output_width, 3),
        border_color,
        dtype=np.uint8,
    )

    # Place frames in grid
    for i, frame in enumerate(frames):
        row = i // cols
        col = i % cols

        # Resize frame if needed
        if frame.shape[:2] != (cell_height, cell_width):
            frame = cv2.resize(frame, (cell_width, cell_height))

        # Calculate position
        x = border_width + col * (cell_width + border_width)
        y = border_width + row * (cell_height + border_width)

        # Place frame
        mosaic[y : y + cell_height, x : x + cell_width] = frame

        # Add label if provided
        if labels and i < len(labels):
            label = labels[i]
            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            # Draw semi-transparent background
            label_bg_start = (x + 5, y + 5)
            label_bg_end = (x + text_width + 15, y + text_height + 15)
            cv2.rectangle(
                mosaic,
                label_bg_start,
                label_bg_end,
                (0, 0, 0),
                -1,
            )

            # Draw text
            cv2.putText(
                mosaic,
                label,
                (x + 10, y + text_height + 10),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

    return mosaic


def add_frame_numbers(
    frames: list[np.ndarray],
    timestamps_ms: Optional[list[float]] = None,
) -> list[str]:
    """Generate labels for frames based on timestamps or indices.

    Args:
        frames: List of frames (used for count)
        timestamps_ms: Optional list of timestamps in milliseconds

    Returns:
        List of label strings
    """
    labels = []

    for i in range(len(frames)):
        if timestamps_ms and i < len(timestamps_ms):
            # Format as seconds with one decimal
            seconds = timestamps_ms[i] / 1000.0
            labels.append(f"T+{seconds:.1f}s")
        else:
            labels.append(f"Frame {i + 1}")

    return labels
