"""ONVIF event subscription service.

Handles PullPoint event subscriptions from ONVIF cameras and converts
motion/analytics events to Detection records for the unified timeline.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ONVIFEvent:
    """Parsed ONVIF event."""

    camera_id: int
    camera_name: str
    topic: str
    timestamp: datetime
    class_name: str  # Mapped event class (motion, tamper, etc.)
    data: dict


class ONVIFEventSubscriber:
    """Manages PullPoint event subscription for a single camera.

    ONVIF events are received via PullPoint subscription, a polling-based
    approach where we periodically request pending events from the camera.
    """

    def __init__(
        self,
        camera_id: int,
        camera_name: str,
        host: str,
        port: int,
        username: str,
        password: str,
        wsdl_dir: str = "/app/wsdl",
    ):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.wsdl_dir = wsdl_dir

        self._camera = None
        self._pullpoint = None
        self._subscription_reference = None
        self._connected = False
        self._subscription_expiry: Optional[datetime] = None

    async def connect(self, timeout: float = 10.0) -> bool:
        """Connect to camera and create PullPoint subscription."""
        try:
            from onvif import ONVIFCamera

            self._camera = ONVIFCamera(
                self.host,
                self.port,
                self.username,
                self.password,
                self.wsdl_dir,
            )

            await asyncio.wait_for(self._camera.update_xaddrs(), timeout=timeout)

            # Create events service and PullPoint subscription
            events_service = await self._camera.create_events_service()

            # Check if events are supported
            caps = await events_service.GetServiceCapabilities()
            if not caps:
                logger.warning(
                    f"Camera {self.camera_name} does not support events service"
                )
                return False

            # Create PullPoint subscription
            # InitialTerminationTime is how long the subscription lives
            subscription_time = "PT1H"  # 1 hour
            try:
                subscription = await events_service.CreatePullPointSubscription(
                    {"InitialTerminationTime": subscription_time}
                )

                self._subscription_reference = subscription.SubscriptionReference
                self._subscription_expiry = datetime.now(timezone.utc) + timedelta(
                    hours=1
                )

                # Create pullpoint service from subscription reference
                self._pullpoint = await self._camera.create_pullpoint_service()

                self._connected = True
                logger.info(
                    f"ONVIF event subscription created for {self.camera_name}"
                )
                return True

            except Exception as e:
                logger.warning(
                    f"Failed to create PullPoint subscription for {self.camera_name}: {e}"
                )
                return False

        except asyncio.TimeoutError:
            logger.warning(f"ONVIF connection timeout for {self.camera_name}")
            return False
        except ImportError as e:
            logger.error(f"ONVIF library not installed: {e}")
            return False
        except Exception as e:
            logger.warning(
                f"Failed to connect to ONVIF events for {self.camera_name}: {e}"
            )
            return False

    async def poll_events(
        self, timeout: float = 5.0, message_limit: int = 10
    ) -> list[ONVIFEvent]:
        """Poll for pending events from the camera.

        Returns list of parsed events. This is a blocking call that waits
        up to `timeout` seconds for events.
        """
        if not self._pullpoint or not self._connected:
            return []

        events = []
        try:
            # PullMessages waits for events up to timeout
            response = await self._pullpoint.PullMessages(
                {
                    "Timeout": timedelta(seconds=timeout),
                    "MessageLimit": message_limit,
                }
            )

            if hasattr(response, "NotificationMessage"):
                for msg in response.NotificationMessage:
                    event = self._parse_notification(msg)
                    if event:
                        events.append(event)

        except asyncio.TimeoutError:
            # Normal timeout, no events pending
            pass
        except Exception as e:
            # Log but don't fail - connection may need renewal
            logger.debug(f"PullMessages error for {self.camera_name}: {e}")

        return events

    def _parse_notification(self, msg) -> Optional[ONVIFEvent]:
        """Parse ONVIF notification message into event object."""
        try:
            # Extract topic
            topic = ""
            if hasattr(msg, "Topic") and msg.Topic:
                topic = (
                    str(msg.Topic._value_1)
                    if hasattr(msg.Topic, "_value_1")
                    else str(msg.Topic)
                )

            # Extract timestamp
            timestamp = datetime.now(timezone.utc)
            if hasattr(msg, "Message") and hasattr(msg.Message, "UtcTime"):
                timestamp = msg.Message.UtcTime
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)

            # Extract data items
            data = {}
            if hasattr(msg, "Message") and hasattr(msg.Message, "Data"):
                data_elem = msg.Message.Data
                if hasattr(data_elem, "SimpleItem"):
                    for item in data_elem.SimpleItem:
                        if hasattr(item, "Name") and hasattr(item, "Value"):
                            data[item.Name] = item.Value

            # Map topic to detection class
            class_name = self._map_topic_to_class(topic, data)
            if not class_name:
                return None  # Skip events we don't care about

            return ONVIFEvent(
                camera_id=self.camera_id,
                camera_name=self.camera_name,
                topic=topic,
                timestamp=timestamp,
                class_name=class_name,
                data=data,
            )

        except Exception as e:
            logger.warning(f"Failed to parse ONVIF notification: {e}")
            return None

    def _map_topic_to_class(self, topic: str, data: dict) -> Optional[str]:
        """Map ONVIF event topic to detection class name.

        Returns None for events we should skip.
        """
        topic_lower = topic.lower()

        # Motion detection events
        if "motion" in topic_lower or "motiondetector" in topic_lower:
            # Check if motion is active (some cameras send start/stop)
            is_motion = data.get("IsMotion", data.get("State", "true"))
            if str(is_motion).lower() in ("true", "1", "yes"):
                return "motion"
            return None  # Motion ended, skip

        # Tamper detection
        if "tamper" in topic_lower:
            return "tamper"

        # Line crossing
        if "linecrossing" in topic_lower or "line_crossing" in topic_lower:
            return "line_crossing"

        # Intrusion/region detection
        if "intrusion" in topic_lower or "regiondetector" in topic_lower:
            return "intrusion"

        # Face detection
        if "face" in topic_lower:
            return "face"

        # Vehicle detection
        if "vehicle" in topic_lower:
            return "vehicle"

        # Person detection (analytics)
        if "human" in topic_lower or "pedestrian" in topic_lower:
            return "person"

        # Object detection
        if "object" in topic_lower:
            return "object"

        # Cell motion (simplified motion detection)
        if "cellmotion" in topic_lower:
            state = data.get("IsMotion", data.get("State", "true"))
            if str(state).lower() in ("true", "1"):
                return "motion"
            return None

        # Unknown event type - log for debugging but skip
        logger.debug(f"Unhandled ONVIF event topic: {topic}")
        return None

    async def renew_subscription(self) -> bool:
        """Renew the event subscription before it expires."""
        if not self._subscription_reference or not self._camera:
            return False

        try:
            # Try to renew using subscription manager
            # Note: Not all cameras support this, some require new subscription
            events_service = await self._camera.create_events_service()
            await events_service.Renew(
                {
                    "TerminationTime": "PT1H",
                    "SubscriptionReference": self._subscription_reference,
                }
            )
            self._subscription_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
            logger.debug(f"Renewed ONVIF subscription for {self.camera_name}")
            return True

        except Exception as e:
            logger.warning(
                f"Failed to renew subscription for {self.camera_name}: {e}"
            )
            # Try to create new subscription
            return await self.connect()

    def needs_renewal(self, buffer_minutes: int = 10) -> bool:
        """Check if subscription needs renewal."""
        if not self._subscription_expiry:
            return True
        buffer = timedelta(minutes=buffer_minutes)
        return datetime.now(timezone.utc) + buffer >= self._subscription_expiry

    async def disconnect(self) -> None:
        """Disconnect and cleanup subscription."""
        try:
            if self._subscription_reference and self._camera:
                # Try to unsubscribe
                events_service = await self._camera.create_events_service()
                await events_service.Unsubscribe({})
        except Exception:
            pass
        finally:
            self._camera = None
            self._pullpoint = None
            self._subscription_reference = None
            self._subscription_expiry = None
            self._connected = False
            logger.debug(f"ONVIF subscription disconnected for {self.camera_name}")

    @property
    def is_connected(self) -> bool:
        """Check if subscriber is connected."""
        return self._connected
