#!/usr/bin/env python3
"""ONVIF event worker for receiving camera motion/analytics events.

This worker subscribes to PullPoint events from ONVIF-enabled cameras and
converts them to Detection records for the unified timeline.

Key features:
- Subscribes to cameras with onvif_events_enabled=true
- Polls for motion, tamper, line crossing, and analytics events
- Converts events to Detection records with event_source="onvif_motion"
- Uses PostgreSQL NOTIFY for real-time SSE
- Automatically renews subscriptions before expiry
- Handles camera reconnection on failures

Usage:
    ./onvif_event_worker.py
"""

import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add the backend directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.config import get_settings
from app.services.onvif.event_subscriber import ONVIFEvent, ONVIFEventSubscriber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("onvif_events")


@dataclass
class DebounceTracker:
    """Track event cooldowns per camera/class to prevent notification spam."""

    cooldown_seconds: float = 30.0
    _last_notified: dict[tuple[int, str], datetime] = field(default_factory=dict)

    def should_notify(self, camera_id: int, class_name: str) -> bool:
        """Check if enough time has passed since last notification."""
        key = (camera_id, class_name)
        last = self._last_notified.get(key)
        if last is None:
            return True
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return elapsed >= self.cooldown_seconds

    def mark_notified(self, camera_id: int, class_name: str) -> None:
        """Record notification time for debouncing."""
        self._last_notified[(camera_id, class_name)] = datetime.now(timezone.utc)


class ONVIFEventWorker:
    """Worker that manages ONVIF event subscriptions for all enabled cameras."""

    def __init__(
        self,
        database_url: str,
        poll_interval: float = 5.0,
        refresh_interval: float = 60.0,
        cooldown_seconds: float = 30.0,
    ):
        """Initialize the ONVIF event worker.

        Args:
            database_url: PostgreSQL connection URL
            poll_interval: Seconds between event polls
            refresh_interval: Seconds between camera list refresh
            cooldown_seconds: Debounce cooldown for duplicate events
        """
        # Convert asyncpg URL format for raw asyncpg connection
        self.database_url = database_url.replace(
            "postgresql+asyncpg://", "postgresql://"
        )
        self._pool = None
        self._running = False
        self._subscribers: dict[int, ONVIFEventSubscriber] = {}
        self._poll_interval = poll_interval
        self._refresh_interval = refresh_interval
        self._debounce = DebounceTracker(cooldown_seconds=cooldown_seconds)

    async def start(self) -> None:
        """Start the event worker."""
        import asyncpg

        logger.info("Starting ONVIF event worker...")
        self._pool = await asyncpg.create_pool(
            self.database_url, min_size=2, max_size=5, ssl=False
        )

        self._running = True
        await self._run_loop()

    async def stop(self) -> None:
        """Stop the event worker."""
        logger.info("Stopping ONVIF event worker...")
        self._running = False

        # Disconnect all subscribers
        for subscriber in self._subscribers.values():
            await subscriber.disconnect()
        self._subscribers.clear()

        if self._pool:
            await self._pool.close()

        logger.info("ONVIF event worker stopped")

    async def _load_cameras(self) -> None:
        """Load cameras with ONVIF events enabled and update subscriptions."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, host, onvif_port, username, password
                FROM cameras
                WHERE onvif_enabled = true
                AND onvif_events_enabled = true
                """
            )

        current_ids = set()
        for row in rows:
            camera_id = row["id"]
            current_ids.add(camera_id)

            if camera_id not in self._subscribers:
                # New camera - create subscriber
                subscriber = ONVIFEventSubscriber(
                    camera_id=camera_id,
                    camera_name=row["name"],
                    host=row["host"],
                    port=row["onvif_port"] or 80,
                    username=row["username"] or "",
                    password=row["password"] or "",
                )
                if await subscriber.connect():
                    self._subscribers[camera_id] = subscriber
                    logger.info(f"Subscribed to ONVIF events: {row['name']}")
                else:
                    logger.warning(
                        f"Failed to subscribe to ONVIF events: {row['name']}"
                    )

        # Remove cameras that are no longer enabled
        removed_ids = set(self._subscribers.keys()) - current_ids
        for camera_id in removed_ids:
            subscriber = self._subscribers.pop(camera_id)
            await subscriber.disconnect()
            logger.info(f"Unsubscribed from ONVIF events: camera_id={camera_id}")

    async def _run_loop(self) -> None:
        """Main event loop."""
        last_refresh = 0.0

        while self._running:
            now = asyncio.get_event_loop().time()

            # Refresh camera list periodically
            if now - last_refresh >= self._refresh_interval:
                try:
                    await self._load_cameras()
                except Exception as e:
                    logger.error(f"Failed to load cameras: {e}")
                last_refresh = now

            # Check for subscription renewals
            for subscriber in list(self._subscribers.values()):
                if subscriber.needs_renewal():
                    try:
                        await subscriber.renew_subscription()
                    except Exception as e:
                        logger.warning(
                            f"Failed to renew subscription for camera "
                            f"{subscriber.camera_id}: {e}"
                        )

            # Poll all subscribers in parallel
            if self._subscribers:
                tasks = [
                    self._poll_subscriber(sub)
                    for sub in self._subscribers.values()
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

            await asyncio.sleep(self._poll_interval)

    async def _poll_subscriber(self, subscriber: ONVIFEventSubscriber) -> None:
        """Poll a single subscriber for events and process them."""
        try:
            events = await subscriber.poll_events(timeout=2.0)

            for event in events:
                await self._process_event(event)

        except Exception as e:
            logger.debug(
                f"Error polling camera {subscriber.camera_id}: {e}"
            )

    async def _process_event(self, event: ONVIFEvent) -> None:
        """Process an ONVIF event and save as detection."""
        # Check debounce
        if not self._debounce.should_notify(event.camera_id, event.class_name):
            logger.debug(
                f"Debounced {event.class_name} from {event.camera_name}"
            )
            return

        logger.info(
            f"ONVIF event from {event.camera_name}: {event.class_name}"
        )

        # Save detection to database
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO detections (
                    recording_id, camera_id, class_name, confidence,
                    timestamp_ms, frame_number, bbox_x, bbox_y,
                    bbox_width, bbox_height, model_name, detected_at,
                    event_source
                ) VALUES (
                    NULL, $1, $2, $3, 0, 0, 0.0, 0.0, 1.0, 1.0,
                    'onvif', $4, 'onvif_motion'
                )
                """,
                event.camera_id,
                event.class_name,
                1.0,  # Confidence is always 1.0 for camera events
                event.timestamp,
            )

            # Notify listeners for real-time SSE
            payload = json.dumps(
                {
                    "camera_id": event.camera_id,
                    "camera_name": event.camera_name,
                    "class_name": event.class_name,
                    "confidence": 1.0,
                    "detected_at": event.timestamp.isoformat(),
                    "event_source": "onvif_motion",
                }
            )
            await conn.execute("SELECT pg_notify('live_detection', $1)", payload)

        # Mark as notified for debouncing
        self._debounce.mark_notified(event.camera_id, event.class_name)


def main() -> None:
    """Main entry point."""
    settings = get_settings()

    if not settings.database_url:
        logger.error("No DATABASE_URL configured")
        sys.exit(1)

    # Configuration from environment
    poll_interval = float(os.environ.get("ONVIF_POLL_INTERVAL", "5.0"))
    refresh_interval = float(os.environ.get("ONVIF_REFRESH_INTERVAL", "60.0"))
    cooldown = float(os.environ.get("ONVIF_EVENT_COOLDOWN", "30.0"))

    worker = ONVIFEventWorker(
        database_url=settings.database_url,
        poll_interval=poll_interval,
        refresh_interval=refresh_interval,
        cooldown_seconds=cooldown,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        loop.call_soon_threadsafe(shutdown_event.set)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    async def run():
        worker_task = asyncio.create_task(worker.start())
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        done, pending = await asyncio.wait(
            [worker_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        await worker.stop()

    try:
        loop.run_until_complete(run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
