"""ONVIF client wrapper for camera communication.

Provides async methods for:
- Connecting to ONVIF-enabled cameras
- Discovering media profiles and RTSP stream URLs
- Querying device information and capabilities
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class MediaProfile:
    """Represents an ONVIF media profile with stream information."""

    token: str
    name: str
    rtsp_url: str
    encoding: Optional[str] = None
    resolution: Optional[tuple[int, int]] = None
    fps: Optional[float] = None


@dataclass
class ONVIFCapabilities:
    """Camera ONVIF capabilities summary."""

    device_info: dict = field(default_factory=dict)
    profiles: list[MediaProfile] = field(default_factory=list)
    has_events: bool = False
    has_analytics: bool = False
    has_ptz: bool = False


class ONVIFClient:
    """Async ONVIF client for a single camera.

    Example usage:
        client = ONVIFClient("192.168.1.100", 80, "admin", "password")
        if await client.connect():
            profiles = await client.get_media_profiles()
            for p in profiles:
                print(f"{p.name}: {p.rtsp_url}")
            await client.disconnect()
    """

    def __init__(
        self,
        host: str,
        port: int = 80,
        username: Optional[str] = None,
        password: Optional[str] = None,
        wsdl_dir: Optional[Path] = None,
    ):
        """Initialize ONVIF client.

        Args:
            host: Camera IP address or hostname
            port: ONVIF service port (typically 80 or 8080)
            username: Camera username for authentication
            password: Camera password for authentication
            wsdl_dir: Path to WSDL files directory (optional, uses library default)
        """
        self.host = host
        self.port = port
        self.username = username or ""
        self.password = password or ""
        self.wsdl_dir = wsdl_dir
        self._camera = None
        self._media_service = None
        self._connected = False

    async def connect(self, timeout: float = 10.0) -> bool:
        """Connect to camera and initialize ONVIF services.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connection successful, False otherwise
        """
        try:
            from onvif import ONVIFCamera
            import onvif

            # Determine WSDL directory - the library default is incorrect
            # WSDL files are in the onvif package's wsdl subdirectory
            wsdl_dir = self.wsdl_dir
            if not wsdl_dir:
                onvif_package_dir = Path(onvif.__file__).parent
                wsdl_dir = onvif_package_dir / "wsdl"

            # Create camera instance
            kwargs = {
                "host": self.host,
                "port": self.port,
                "user": self.username,
                "passwd": self.password,
                "wsdl_dir": str(wsdl_dir),
            }

            self._camera = ONVIFCamera(**kwargs)

            # Update service addresses with timeout
            await asyncio.wait_for(
                self._camera.update_xaddrs(),
                timeout=timeout,
            )

            self._connected = True
            logger.info(f"ONVIF connected to {self.host}:{self.port}")
            return True

        except asyncio.TimeoutError:
            logger.warning(f"ONVIF connection timeout: {self.host}:{self.port}")
            return False
        except ImportError as e:
            logger.error(f"ONVIF library not installed: {e}")
            return False
        except Exception as e:
            logger.warning(f"ONVIF connection failed to {self.host}:{self.port}: {e}")
            return False

    async def get_media_profiles(self) -> list[MediaProfile]:
        """Get available media profiles with RTSP URLs.

        Returns:
            List of MediaProfile objects with stream information
        """
        if not self._camera or not self._connected:
            return []

        profiles = []
        try:
            # Try Media2 service first (newer cameras, ONVIF Profile T)
            profiles = await self._get_media2_profiles()
            if profiles:
                return profiles

            # Fall back to Media1 service (ONVIF Profile S)
            profiles = await self._get_media1_profiles()

        except Exception as e:
            logger.error(f"Failed to get media profiles from {self.host}: {e}")

        return profiles

    async def _get_media2_profiles(self) -> list[MediaProfile]:
        """Get profiles using Media2 service (ONVIF Profile T)."""
        profiles = []
        try:
            media2 = await self._camera.create_media2_service()
            raw_profiles = await media2.GetProfiles()

            for p in raw_profiles:
                try:
                    # Get stream URI for this profile
                    uri_req = media2.create_type("GetStreamUri")
                    uri_req.ProfileToken = p.token
                    uri_req.Protocol = "RTSP"
                    uri_response = await media2.GetStreamUri(uri_req)

                    rtsp_url = (
                        uri_response.Uri
                        if hasattr(uri_response, "Uri")
                        else str(uri_response)
                    )

                    # Extract video config info if available
                    resolution = None
                    fps = None
                    encoding = None

                    if hasattr(p, "Configurations"):
                        configs = p.Configurations
                        if hasattr(configs, "VideoEncoder"):
                            vec = configs.VideoEncoder
                            if hasattr(vec, "Resolution"):
                                resolution = (
                                    vec.Resolution.Width,
                                    vec.Resolution.Height,
                                )
                            if hasattr(vec, "RateControl") and hasattr(
                                vec.RateControl, "FrameRateLimit"
                            ):
                                fps = float(vec.RateControl.FrameRateLimit)
                            if hasattr(vec, "Encoding"):
                                encoding = str(vec.Encoding)

                    profiles.append(
                        MediaProfile(
                            token=p.token,
                            name=p.Name if hasattr(p, "Name") else p.token,
                            rtsp_url=self._sanitize_rtsp_url(rtsp_url),
                            encoding=encoding,
                            resolution=resolution,
                            fps=fps,
                        )
                    )
                except Exception as e:
                    logger.debug(f"Failed to get stream URI for profile {p.token}: {e}")
                    continue

        except Exception as e:
            logger.debug(f"Media2 service not available: {e}")

        return profiles

    async def _get_media1_profiles(self) -> list[MediaProfile]:
        """Get profiles using Media1 service (ONVIF Profile S)."""
        profiles = []
        try:
            media = await self._camera.create_media_service()
            self._media_service = media
            raw_profiles = await media.GetProfiles()

            for p in raw_profiles:
                try:
                    # Get stream URI
                    stream_setup = {
                        "Stream": "RTP-Unicast",
                        "Transport": {"Protocol": "RTSP"},
                    }
                    uri_response = await media.GetStreamUri(
                        {
                            "StreamSetup": stream_setup,
                            "ProfileToken": p.token,
                        }
                    )

                    rtsp_url = uri_response.Uri

                    # Extract video encoder configuration
                    resolution = None
                    fps = None
                    encoding = None

                    if hasattr(p, "VideoEncoderConfiguration"):
                        vec = p.VideoEncoderConfiguration
                        if hasattr(vec, "Resolution"):
                            resolution = (vec.Resolution.Width, vec.Resolution.Height)
                        if hasattr(vec, "RateControl") and hasattr(
                            vec.RateControl, "FrameRateLimit"
                        ):
                            fps = float(vec.RateControl.FrameRateLimit)
                        if hasattr(vec, "Encoding"):
                            encoding = str(vec.Encoding)

                    profiles.append(
                        MediaProfile(
                            token=p.token,
                            name=p.Name if hasattr(p, "Name") else p.token,
                            rtsp_url=self._sanitize_rtsp_url(rtsp_url),
                            encoding=encoding,
                            resolution=resolution,
                            fps=fps,
                        )
                    )
                except Exception as e:
                    logger.debug(f"Failed to get stream URI for profile {p.token}: {e}")
                    continue

        except Exception as e:
            logger.debug(f"Media1 service error: {e}")

        return profiles

    def _sanitize_rtsp_url(self, url: str) -> str:
        """Remove credentials from RTSP URL if present.

        The URL returned by ONVIF may include credentials. We strip them
        since we store credentials separately in the camera model.
        """
        try:
            parsed = urlparse(url)
            if parsed.username or parsed.password:
                # Rebuild URL without credentials
                netloc = parsed.hostname
                if parsed.port:
                    netloc = f"{netloc}:{parsed.port}"
                return parsed._replace(netloc=netloc).geturl()
        except Exception:
            pass
        return url

    async def get_device_info(self) -> dict:
        """Get device information (manufacturer, model, firmware, serial).

        Returns:
            Dictionary with device information
        """
        if not self._camera or not self._connected:
            return {}

        try:
            devicemgmt = await self._camera.create_devicemgmt_service()
            info = await devicemgmt.GetDeviceInformation()
            return {
                "manufacturer": getattr(info, "Manufacturer", ""),
                "model": getattr(info, "Model", ""),
                "firmware": getattr(info, "FirmwareVersion", ""),
                "serial": getattr(info, "SerialNumber", ""),
                "hardware_id": getattr(info, "HardwareId", ""),
            }
        except Exception as e:
            logger.error(f"Failed to get device info from {self.host}: {e}")
            return {}

    async def get_capabilities(self) -> ONVIFCapabilities:
        """Get full camera capabilities including profiles and features.

        Returns:
            ONVIFCapabilities object with all discovered information
        """
        profiles = await self.get_media_profiles()
        device_info = await self.get_device_info()

        has_events = False
        has_analytics = False
        has_ptz = False

        if self._camera and self._connected:
            try:
                devicemgmt = await self._camera.create_devicemgmt_service()
                caps = await devicemgmt.GetCapabilities({"Category": "All"})
                has_events = hasattr(caps, "Events") and caps.Events is not None
                has_analytics = (
                    hasattr(caps, "Analytics") and caps.Analytics is not None
                )
                has_ptz = hasattr(caps, "PTZ") and caps.PTZ is not None
            except Exception as e:
                logger.debug(f"Failed to get capabilities: {e}")

        return ONVIFCapabilities(
            device_info=device_info,
            profiles=profiles,
            has_events=has_events,
            has_analytics=has_analytics,
            has_ptz=has_ptz,
        )

    async def disconnect(self) -> None:
        """Disconnect from camera and cleanup resources."""
        if self._camera:
            try:
                await self._camera.close()
            except Exception:
                pass
        self._camera = None
        self._media_service = None
        self._connected = False
        logger.debug(f"ONVIF disconnected from {self.host}")

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected
