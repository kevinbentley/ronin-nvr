"""ByteTrack-style multi-object tracking.

This module implements a simplified ByteTrack algorithm for tracking objects
across video frames. Key features:

- Kalman filter for motion prediction
- IoU (Intersection over Union) matching
- Two-stage matching (high-confidence then low-confidence detections)
- Track persistence across occlusions

ByteTrack Paper: https://arxiv.org/abs/2110.06864

The tracker maintains a track buffer so objects that temporarily disappear
(e.g., behind obstacles) can be re-associated when they reappear.

Performance:
- Track update: ~0.5-1ms per frame
- Minimal memory overhead per track
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class TrackState(Enum):
    """State of a tracked object."""

    TENTATIVE = "tentative"  # New track, not yet confirmed
    CONFIRMED = "confirmed"  # Track has been confirmed with sufficient detections
    LOST = "lost"  # Track has been lost (no recent detections)


@dataclass
class Detection:
    """A single detection for tracking input."""

    x: float  # Normalized 0-1 bounding box
    y: float
    width: float
    height: float
    confidence: float
    class_id: int
    class_name: str = ""

    @property
    def center(self) -> tuple[float, float]:
        """Center point of bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Bounding box as (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def area(self) -> float:
        """Area of bounding box."""
        return self.width * self.height


@dataclass
class TrackedObject:
    """A tracked object with state and history."""

    track_id: int
    class_id: int
    class_name: str

    # Current state (normalized coordinates)
    x: float
    y: float
    width: float
    height: float
    confidence: float

    # Velocity (normalized per frame)
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    # Tracking state
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1  # Number of detections matched
    age: int = 0  # Total frames since creation
    time_since_update: int = 0  # Frames since last detection match

    # Initial position for displacement tracking
    initial_x: float = 0.0
    initial_y: float = 0.0
    max_displacement: float = 0.0  # Maximum displacement from initial position

    # Kalman filter state (internal)
    _kf_state: Optional[np.ndarray] = field(default=None, repr=False)
    _kf_covariance: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def center(self) -> tuple[float, float]:
        """Center point of bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Bounding box as (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    @property
    def is_confirmed(self) -> bool:
        """Whether the track is confirmed."""
        return self.state == TrackState.CONFIRMED

    @property
    def is_tentative(self) -> bool:
        """Whether the track is tentative."""
        return self.state == TrackState.TENTATIVE

    @property
    def is_lost(self) -> bool:
        """Whether the track is lost."""
        return self.state == TrackState.LOST


def iou(box1: tuple[float, float, float, float],
        box2: tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union between two boxes.

    Args:
        box1, box2: Boxes as (x1, y1, x2, y2)

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def iou_distance(tracks: list[TrackedObject],
                 detections: list[Detection]) -> np.ndarray:
    """Calculate IoU distance matrix between tracks and detections.

    Args:
        tracks: List of tracked objects
        detections: List of detections

    Returns:
        Distance matrix where distance = 1 - IoU
    """
    if not tracks or not detections:
        return np.empty((len(tracks), len(detections)))

    cost_matrix = np.zeros((len(tracks), len(detections)))

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            cost_matrix[i, j] = 1 - iou(track.xyxy, det.xyxy)

    return cost_matrix


class KalmanFilter:
    """Simple Kalman filter for bounding box tracking.

    State vector: [x, y, w, h, vx, vy, vw, vh]
    where (x, y) is center, (w, h) is size, v* are velocities
    """

    def __init__(self):
        # State transition matrix (constant velocity model)
        self.F = np.eye(8)
        self.F[0, 4] = 1  # x += vx
        self.F[1, 5] = 1  # y += vy
        self.F[2, 6] = 1  # w += vw
        self.F[3, 7] = 1  # h += vh

        # Measurement matrix (we observe x, y, w, h)
        self.H = np.eye(4, 8)

        # Process noise covariance
        self.Q = np.eye(8) * 0.01
        self.Q[4:, 4:] *= 0.1  # Lower noise for velocities

        # Measurement noise covariance
        self.R = np.eye(4) * 0.1

        # Initial state covariance
        self.P_init = np.eye(8) * 10
        self.P_init[4:, 4:] *= 100  # High uncertainty for initial velocities

    def initiate(self, detection: Detection) -> tuple[np.ndarray, np.ndarray]:
        """Initialize state from first detection.

        Args:
            detection: Initial detection

        Returns:
            Tuple of (state, covariance)
        """
        cx, cy = detection.center
        state = np.array([
            cx, cy, detection.width, detection.height,
            0, 0, 0, 0  # Initial velocities are zero
        ], dtype=np.float64)

        return state, self.P_init.copy()

    def predict(self, state: np.ndarray, covariance: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """Predict next state.

        Args:
            state: Current state vector
            covariance: Current covariance matrix

        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        predicted_state = self.F @ state
        predicted_covariance = self.F @ covariance @ self.F.T + self.Q
        return predicted_state, predicted_covariance

    def update(self, state: np.ndarray, covariance: np.ndarray,
               detection: Detection) -> tuple[np.ndarray, np.ndarray]:
        """Update state with new measurement.

        Args:
            state: Current state vector
            covariance: Current covariance matrix
            detection: New detection

        Returns:
            Tuple of (updated_state, updated_covariance)
        """
        cx, cy = detection.center
        measurement = np.array([cx, cy, detection.width, detection.height])

        # Innovation
        y = measurement - self.H @ state

        # Innovation covariance
        S = self.H @ covariance @ self.H.T + self.R

        # Kalman gain
        K = covariance @ self.H.T @ np.linalg.inv(S)

        # Update
        updated_state = state + K @ y
        updated_covariance = (np.eye(8) - K @ self.H) @ covariance

        return updated_state, updated_covariance


class ByteTracker:
    """ByteTrack-style multi-object tracker.

    Implements the ByteTrack algorithm with:
    - Two-stage matching (high then low confidence)
    - Kalman filter prediction
    - Track persistence buffer

    Example:
        >>> tracker = ByteTracker()
        >>> for frame_detections in video_detections:
        ...     tracked = tracker.update(frame_detections)
        ...     for obj in tracked:
        ...         print(f"Track {obj.track_id}: {obj.class_name} at {obj.center}")
    """

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_hits: int = 3,
        min_displacement: float = 0.0,
    ):
        """Initialize tracker.

        Args:
            track_high_thresh: Confidence threshold for high-confidence detections
            track_low_thresh: Confidence threshold for low-confidence detections
            match_thresh: IoU threshold for matching (1 - IoU < thresh)
            track_buffer: Number of frames to keep lost tracks
            min_hits: Minimum detections to confirm a track
            min_displacement: Minimum spatial displacement (normalized 0-1) required
                             to confirm a track. Helps filter stationary false positives
                             like flickering lights. Set to 0 to disable.
        """
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_hits = min_hits
        self.min_displacement = min_displacement

        self._tracks: list[TrackedObject] = []
        self._lost_tracks: list[TrackedObject] = []
        self._next_id = 1
        self._frame_count = 0
        self._kf = KalmanFilter()

    @property
    def active_tracks(self) -> list[TrackedObject]:
        """Get currently active (confirmed) tracks."""
        return [t for t in self._tracks if t.is_confirmed]

    @property
    def all_tracks(self) -> list[TrackedObject]:
        """Get all tracks including tentative."""
        return self._tracks.copy()

    def _create_track(self, detection: Detection) -> TrackedObject:
        """Create a new track from a detection."""
        state, covariance = self._kf.initiate(detection)
        cx, cy = detection.center

        track = TrackedObject(
            track_id=self._next_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            x=detection.x,
            y=detection.y,
            width=detection.width,
            height=detection.height,
            confidence=detection.confidence,
            state=TrackState.TENTATIVE,
            initial_x=cx,
            initial_y=cy,
            max_displacement=0.0,
            _kf_state=state,
            _kf_covariance=covariance,
        )
        logger.debug(
            f"New track {self._next_id}: {detection.class_name} conf={detection.confidence:.2f} "
            f"pos=({detection.x:.3f}, {detection.y:.3f})"
        )
        self._next_id += 1
        return track

    def _predict_tracks(self) -> None:
        """Run Kalman filter prediction on all tracks."""
        for track in self._tracks + self._lost_tracks:
            if track._kf_state is not None:
                state, cov = self._kf.predict(track._kf_state, track._kf_covariance)
                track._kf_state = state
                track._kf_covariance = cov

                # Update track position from prediction
                cx, cy, w, h = state[:4]
                track.x = cx - w / 2
                track.y = cy - h / 2
                track.width = w
                track.height = h
                track.velocity_x = state[4]
                track.velocity_y = state[5]

    def _update_track(self, track: TrackedObject, detection: Detection) -> None:
        """Update a track with a matched detection."""
        if track._kf_state is not None:
            state, cov = self._kf.update(
                track._kf_state, track._kf_covariance, detection
            )
            track._kf_state = state
            track._kf_covariance = cov

            # Update from Kalman state
            cx, cy, w, h = state[:4]
            track.x = cx - w / 2
            track.y = cy - h / 2
            track.width = w
            track.height = h
            track.velocity_x = state[4]
            track.velocity_y = state[5]
        else:
            # Fallback: direct update
            track.x = detection.x
            track.y = detection.y
            track.width = detection.width
            track.height = detection.height

        track.confidence = detection.confidence
        track.hits += 1
        track.time_since_update = 0

        # Calculate displacement from initial position
        cx, cy = track.center
        displacement = ((cx - track.initial_x) ** 2 + (cy - track.initial_y) ** 2) ** 0.5
        track.max_displacement = max(track.max_displacement, displacement)

        # Confirm track if enough hits AND sufficient displacement (if required)
        if track.is_tentative and track.hits >= self.min_hits:
            if self.min_displacement <= 0 or track.max_displacement >= self.min_displacement:
                track.state = TrackState.CONFIRMED
                logger.debug(
                    f"Track {track.track_id} CONFIRMED: {track.class_name} hits={track.hits}"
                )
            # If min_displacement is set but not met, track stays tentative
            # It will eventually be removed if it never moves

    def _match(
        self,
        tracks: list[TrackedObject],
        detections: list[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Match tracks to detections using Hungarian algorithm.

        Args:
            tracks: List of tracks
            detections: List of detections

        Returns:
            Tuple of (matches, unmatched_track_indices, unmatched_detection_indices)
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Calculate IoU distance matrix
        cost_matrix = iou_distance(tracks, detections)

        # Apply threshold
        cost_matrix[cost_matrix > self.match_thresh] = self.match_thresh + 0.1

        # Hungarian matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= self.match_thresh:
                matches.append((row, col))

        matched_tracks = set(m[0] for m in matches)
        matched_dets = set(m[1] for m in matches)

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]

        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """Update tracker with new detections.

        Args:
            detections: List of detections for current frame

        Returns:
            List of active tracked objects
        """
        self._frame_count += 1

        # Debug: Log track state before update
        tentative_count = sum(1 for t in self._tracks if t.is_tentative)
        confirmed_count = sum(1 for t in self._tracks if t.is_confirmed)
        if detections:
            det_summary = ", ".join(f"{d.class_name}:{d.confidence:.2f}" for d in detections[:3])
            logger.debug(
                f"Tracker update: dets={len(detections)} [{det_summary}], "
                f"tracks={len(self._tracks)} (tent={tentative_count}, conf={confirmed_count})"
            )

        # Run prediction step
        self._predict_tracks()

        # Increment age for all tracks
        for track in self._tracks + self._lost_tracks:
            track.age += 1
            track.time_since_update += 1

        # Split detections by confidence
        high_dets = [d for d in detections if d.confidence >= self.track_high_thresh]
        low_dets = [d for d in detections if self.track_low_thresh <= d.confidence < self.track_high_thresh]

        # === First association: confirmed tracks with high-confidence detections ===
        confirmed = [t for t in self._tracks if t.is_confirmed]
        matches1, unmatched_tracks1, unmatched_dets1 = self._match(confirmed, high_dets)

        for track_idx, det_idx in matches1:
            self._update_track(confirmed[track_idx], high_dets[det_idx])

        # === Second association: remaining confirmed with low-confidence detections ===
        remaining_confirmed = [confirmed[i] for i in unmatched_tracks1]
        matches2, unmatched_tracks2, _ = self._match(remaining_confirmed, low_dets)

        for track_idx, det_idx in matches2:
            self._update_track(remaining_confirmed[track_idx], low_dets[det_idx])

        # === Third association: tentative tracks with remaining high-confidence ===
        tentative = [t for t in self._tracks if t.is_tentative]
        remaining_high_dets = [high_dets[i] for i in unmatched_dets1]
        matches3, unmatched_tent, unmatched_high = self._match(tentative, remaining_high_dets)

        if tentative and remaining_high_dets:
            logger.debug(
                f"Tentative matching: {len(tentative)} tracks, {len(remaining_high_dets)} dets, "
                f"{len(matches3)} matches"
            )

        for track_idx, det_idx in matches3:
            track = tentative[track_idx]
            det = remaining_high_dets[det_idx]
            logger.debug(
                f"Match tentative track {track.track_id} ({track.class_name}) with "
                f"{det.class_name}:{det.confidence:.2f}"
            )
            self._update_track(track, det)

        # === Fourth association: lost tracks with remaining detections ===
        remaining_dets = [remaining_high_dets[i] for i in unmatched_high]
        matches4, unmatched_lost, unmatched_final = self._match(self._lost_tracks, remaining_dets)

        for track_idx, det_idx in matches4:
            track = self._lost_tracks[track_idx]
            self._update_track(track, remaining_dets[det_idx])
            track.state = TrackState.CONFIRMED
            self._tracks.append(track)

        # Remove re-activated tracks from lost
        reactivated = set(m[0] for m in matches4)
        self._lost_tracks = [t for i, t in enumerate(self._lost_tracks) if i not in reactivated]

        # === Create new tracks from unmatched high-confidence detections ===
        for det_idx in unmatched_final:
            new_track = self._create_track(remaining_dets[det_idx])
            self._tracks.append(new_track)

        # === Handle unmatched tracks ===
        # Mark unmatched confirmed tracks as lost
        for track_idx in unmatched_tracks2:
            track = remaining_confirmed[track_idx]
            if track.time_since_update > 1:
                track.state = TrackState.LOST
                self._lost_tracks.append(track)
                self._tracks.remove(track)

        # Remove tentative tracks that weren't matched
        for track_idx in unmatched_tent:
            track = tentative[track_idx]
            if track.time_since_update > 2:
                self._tracks.remove(track)

        # Remove old lost tracks
        self._lost_tracks = [
            t for t in self._lost_tracks
            if t.time_since_update <= self.track_buffer
        ]

        # Return confirmed tracks
        return [t for t in self._tracks if t.is_confirmed]

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracks.clear()
        self._lost_tracks.clear()
        self._next_id = 1
        self._frame_count = 0

    @property
    def stats(self) -> dict:
        """Get tracker statistics."""
        return {
            "frame_count": self._frame_count,
            "active_tracks": len([t for t in self._tracks if t.is_confirmed]),
            "tentative_tracks": len([t for t in self._tracks if t.is_tentative]),
            "lost_tracks": len(self._lost_tracks),
            "total_ids": self._next_id - 1,
        }
