"""Live detection PostgreSQL NOTIFY listener.

Listens for live_detection notifications from the live_detection_worker
and forwards them to SSE subscribers via the MLEventService.
"""

import asyncio
import json
import logging
from typing import Optional

import asyncpg

from app.config import get_settings
from app.services.ml.events import ml_event_service

logger = logging.getLogger(__name__)
settings = get_settings()

# PostgreSQL NOTIFY channel name
LIVE_DETECTION_CHANNEL = "live_detection"


class LiveDetectionListener:
    """Listens for live detection notifications and forwards to SSE."""

    def __init__(self):
        self._connection: Optional[asyncpg.Connection] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start listening for live detection notifications."""
        if self._running:
            return

        try:
            # Get database URL from settings
            database_url = settings.database_url.replace(
                "postgresql+asyncpg://", "postgresql://"
            )

            self._connection = await asyncpg.connect(database_url)

            # Set up notification handler
            await self._connection.add_listener(
                LIVE_DETECTION_CHANNEL, self._handle_notification
            )

            self._running = True
            logger.info("Live detection listener started")

        except Exception as e:
            logger.error(f"Failed to start live detection listener: {e}")
            raise

    async def stop(self) -> None:
        """Stop listening for notifications."""
        self._running = False

        if self._connection:
            try:
                await self._connection.remove_listener(
                    LIVE_DETECTION_CHANNEL, self._handle_notification
                )
                await self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing listener connection: {e}")
            finally:
                self._connection = None

        logger.info("Live detection listener stopped")

    def _handle_notification(
        self,
        connection: asyncpg.Connection,
        pid: int,
        channel: str,
        payload: str,
    ) -> None:
        """Handle incoming PostgreSQL notification.

        This runs in the asyncpg event loop context, so we schedule
        the async event emission.
        """
        try:
            data = json.loads(payload)

            # Schedule the SSE emission (this is sync, so use ensure_future)
            asyncio.ensure_future(self._emit_event(data))

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid live detection payload: {e}")
        except Exception as e:
            logger.error(f"Error handling live detection notification: {e}")

    async def _emit_event(self, data: dict) -> None:
        """Emit live detection event to SSE subscribers."""
        try:
            await ml_event_service.emit_live_detection(
                camera_id=data["camera_id"],
                camera_name=data["camera_name"],
                class_name=data["class_name"],
                confidence=data["confidence"],
                snapshot_url=f"/api/snapshots/{data['snapshot_path']}"
                if data.get("snapshot_path")
                else None,
            )

            logger.debug(
                f"Emitted live detection: {data['class_name']} on {data['camera_name']}"
            )

        except Exception as e:
            logger.error(f"Error emitting live detection event: {e}")


# Global instance
live_detection_listener = LiveDetectionListener()
