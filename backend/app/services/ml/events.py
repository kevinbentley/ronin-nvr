"""ML Event Service for real-time notifications."""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import AsyncIterator, Optional, Callable, Any
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of ML events."""

    JOB_CREATED = "job_created"
    JOB_STARTED = "job_started"
    JOB_PROGRESS = "job_progress"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    DETECTION = "detection"
    LIVE_DETECTION = "live_detection"  # Real-time detection from live stream


@dataclass
class MLEvent:
    """An ML event for SSE streaming."""

    event_type: EventType
    data: dict
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        event_data = {
            "type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            **self.data,
        }
        return f"event: {self.event_type.value}\ndata: {json.dumps(event_data)}\n\n"


class MLEventService:
    """Service for emitting and subscribing to ML events.

    Provides a pub/sub mechanism for real-time ML event notifications
    via Server-Sent Events (SSE).
    """

    def __init__(self, max_subscribers: int = 100):
        """Initialize event service.

        Args:
            max_subscribers: Maximum number of concurrent subscribers
        """
        self.max_subscribers = max_subscribers
        self._subscribers: list[asyncio.Queue[MLEvent]] = []
        self._lock = asyncio.Lock()

    async def subscribe(self) -> AsyncIterator[MLEvent]:
        """Subscribe to ML events.

        Yields:
            MLEvent objects as they occur
        """
        if len(self._subscribers) >= self.max_subscribers:
            logger.warning("Max subscribers reached, rejecting new subscription")
            return

        queue: asyncio.Queue[MLEvent] = asyncio.Queue(maxsize=100)

        async with self._lock:
            self._subscribers.append(queue)
            logger.debug(f"New subscriber, total: {len(self._subscribers)}")

        try:
            while True:
                event = await queue.get()
                yield event
        except asyncio.CancelledError:
            pass
        finally:
            async with self._lock:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)
                    logger.debug(f"Subscriber removed, total: {len(self._subscribers)}")

    async def emit(self, event: MLEvent) -> None:
        """Emit an event to all subscribers.

        Args:
            event: Event to emit
        """
        async with self._lock:
            dead_queues = []
            for queue in self._subscribers:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    # Slow subscriber, mark for removal
                    dead_queues.append(queue)
                    logger.warning("Dropping slow subscriber")

            for queue in dead_queues:
                self._subscribers.remove(queue)

    async def emit_job_created(self, job_id: int, recording_id: int, model_name: str) -> None:
        """Emit job created event."""
        await self.emit(MLEvent(
            event_type=EventType.JOB_CREATED,
            data={
                "job_id": job_id,
                "recording_id": recording_id,
                "model_name": model_name,
            },
        ))

    async def emit_job_started(self, job_id: int, total_frames: int) -> None:
        """Emit job started event."""
        await self.emit(MLEvent(
            event_type=EventType.JOB_STARTED,
            data={
                "job_id": job_id,
                "total_frames": total_frames,
            },
        ))

    async def emit_job_progress(
        self,
        job_id: int,
        progress_percent: float,
        frames_processed: int,
        detections_count: int,
    ) -> None:
        """Emit job progress event."""
        await self.emit(MLEvent(
            event_type=EventType.JOB_PROGRESS,
            data={
                "job_id": job_id,
                "progress_percent": progress_percent,
                "frames_processed": frames_processed,
                "detections_count": detections_count,
            },
        ))

    async def emit_job_completed(
        self,
        job_id: int,
        frames_processed: int,
        detections_count: int,
        processing_time_seconds: float,
    ) -> None:
        """Emit job completed event."""
        await self.emit(MLEvent(
            event_type=EventType.JOB_COMPLETED,
            data={
                "job_id": job_id,
                "frames_processed": frames_processed,
                "detections_count": detections_count,
                "processing_time_seconds": processing_time_seconds,
            },
        ))

    async def emit_job_failed(self, job_id: int, error_message: str) -> None:
        """Emit job failed event."""
        await self.emit(MLEvent(
            event_type=EventType.JOB_FAILED,
            data={
                "job_id": job_id,
                "error_message": error_message,
            },
        ))

    async def emit_detection(
        self,
        detection_id: int,
        job_id: int,
        class_name: str,
        confidence: float,
        camera_id: int,
    ) -> None:
        """Emit detection event (for high-confidence detections)."""
        await self.emit(MLEvent(
            event_type=EventType.DETECTION,
            data={
                "detection_id": detection_id,
                "job_id": job_id,
                "class_name": class_name,
                "confidence": confidence,
                "camera_id": camera_id,
            },
        ))

    async def emit_live_detection(
        self,
        camera_id: int,
        camera_name: str,
        class_name: str,
        confidence: float,
        snapshot_url: Optional[str] = None,
    ) -> None:
        """Emit live detection event for real-time alerts.

        Args:
            camera_id: Camera that captured the detection
            camera_name: Human-readable camera name
            class_name: Detected object class (person, car, etc.)
            confidence: Detection confidence (0.0-1.0)
            snapshot_url: Optional URL to snapshot image with bounding box
        """
        await self.emit(MLEvent(
            event_type=EventType.LIVE_DETECTION,
            data={
                "camera_id": camera_id,
                "camera_name": camera_name,
                "class_name": class_name,
                "confidence": confidence,
                "snapshot_url": snapshot_url,
            },
        ))

    @property
    def subscriber_count(self) -> int:
        """Get number of active subscribers."""
        return len(self._subscribers)


# Global event service instance
ml_event_service = MLEventService()
