"""Pydantic schemas for ONVIF API endpoints."""

from typing import Optional

from pydantic import BaseModel, Field


class ONVIFProfile(BaseModel):
    """A single ONVIF media profile with stream information."""

    token: str = Field(..., description="Profile token identifier")
    name: str = Field(..., description="Human-readable profile name")
    rtsp_url: str = Field(..., description="RTSP stream URL")
    encoding: Optional[str] = Field(None, description="Video encoding (H264, H265)")
    resolution: Optional[str] = Field(None, description="Resolution (e.g., 1920x1080)")
    fps: Optional[float] = Field(None, description="Frame rate limit")


class ONVIFDeviceInfo(BaseModel):
    """Camera device information from ONVIF."""

    manufacturer: Optional[str] = None
    model: Optional[str] = None
    firmware: Optional[str] = None
    serial: Optional[str] = None
    hardware_id: Optional[str] = None


class ONVIFProbeRequest(BaseModel):
    """Request to probe a camera for ONVIF capabilities."""

    host: str = Field(..., description="Camera IP or hostname")
    onvif_port: int = Field(80, ge=1, le=65535, description="ONVIF port (usually 80)")
    username: Optional[str] = Field(None, description="Camera username")
    password: Optional[str] = Field(None, description="Camera password")
    timeout: float = Field(
        10.0, ge=1.0, le=60.0, description="Connection timeout in seconds"
    )


class ONVIFProbeResponse(BaseModel):
    """Response from ONVIF probe operation."""

    success: bool = Field(..., description="Whether probe was successful")
    host: str = Field(..., description="Probed host address")
    device_info: ONVIFDeviceInfo = Field(
        default_factory=ONVIFDeviceInfo, description="Device information"
    )
    profiles: list[ONVIFProfile] = Field(
        default_factory=list, description="Available media profiles"
    )
    has_events: bool = Field(False, description="Camera supports ONVIF events")
    has_analytics: bool = Field(False, description="Camera supports video analytics")
    has_ptz: bool = Field(False, description="Camera supports PTZ control")
    error: Optional[str] = Field(None, description="Error message if probe failed")


class ONVIFProfilesResponse(BaseModel):
    """Response for camera profiles query."""

    camera_id: int
    profiles: list[ONVIFProfile]


class ONVIFApplyProfileRequest(BaseModel):
    """Request to apply an ONVIF profile to a camera."""

    profile_token: str = Field(..., description="Profile token to use")
    rtsp_url: str = Field(..., description="RTSP URL from the profile")


class ONVIFApplyProfileResponse(BaseModel):
    """Response after applying a profile."""

    success: bool
    camera_id: int
    new_path: str = Field(..., description="Extracted RTSP path")
    new_port: int = Field(..., description="Extracted RTSP port")
    profile_token: str


class ONVIFEventsResponse(BaseModel):
    """Response for event subscription changes."""

    success: bool
    camera_id: int
    events_enabled: bool
