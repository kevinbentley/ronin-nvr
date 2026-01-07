"""ONVIF API endpoints for camera discovery and configuration."""

import logging
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_admin_user
from app.models.user import User
from app.schemas.onvif import (
    ONVIFApplyProfileRequest,
    ONVIFApplyProfileResponse,
    ONVIFDeviceInfo,
    ONVIFEventsResponse,
    ONVIFProbeRequest,
    ONVIFProbeResponse,
    ONVIFProfile,
    ONVIFProfilesResponse,
)
from app.services.camera import CameraService
from app.services.onvif import ONVIFClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/onvif", tags=["onvif"])


@router.post("/probe", response_model=ONVIFProbeResponse)
async def probe_camera(
    request: ONVIFProbeRequest,
    current_user: User = Depends(get_admin_user),
) -> ONVIFProbeResponse:
    """Probe a camera for ONVIF capabilities and media profiles.

    Use this endpoint to discover available RTSP stream paths before adding
    a camera. The probe will return device information, available media
    profiles with their RTSP URLs, and capability flags.

    Requires admin privileges.
    """
    client = ONVIFClient(
        host=request.host,
        port=request.onvif_port,
        username=request.username,
        password=request.password,
    )

    connected = await client.connect(timeout=request.timeout)
    if not connected:
        return ONVIFProbeResponse(
            success=False,
            host=request.host,
            error=f"Could not connect to ONVIF service at {request.host}:{request.onvif_port}",
        )

    try:
        capabilities = await client.get_capabilities()

        profiles = []
        for p in capabilities.profiles:
            resolution = None
            if p.resolution:
                resolution = f"{p.resolution[0]}x{p.resolution[1]}"

            profiles.append(
                ONVIFProfile(
                    token=p.token,
                    name=p.name,
                    rtsp_url=p.rtsp_url,
                    encoding=p.encoding,
                    resolution=resolution,
                    fps=p.fps,
                )
            )

        return ONVIFProbeResponse(
            success=True,
            host=request.host,
            device_info=ONVIFDeviceInfo(**capabilities.device_info),
            profiles=profiles,
            has_events=capabilities.has_events,
            has_analytics=capabilities.has_analytics,
            has_ptz=capabilities.has_ptz,
        )
    except Exception as e:
        logger.error(f"ONVIF probe failed for {request.host}: {e}")
        return ONVIFProbeResponse(
            success=False,
            host=request.host,
            error=str(e),
        )
    finally:
        await client.disconnect()


@router.get("/cameras/{camera_id}/profiles", response_model=ONVIFProfilesResponse)
async def get_camera_profiles(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_admin_user),
) -> ONVIFProfilesResponse:
    """Get ONVIF media profiles for an existing camera.

    The camera must have ONVIF enabled (`onvif_enabled=true`).
    Returns available profiles that can be applied to update the camera's
    RTSP stream path.

    Requires admin privileges.
    """
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not camera.onvif_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ONVIF is not enabled for this camera. "
            "Enable ONVIF in camera settings first.",
        )

    client = ONVIFClient(
        host=camera.host,
        port=camera.onvif_port or 80,
        username=camera.username,
        password=camera.password,
    )

    connected = await client.connect()
    if not connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect to camera ONVIF service at "
            f"{camera.host}:{camera.onvif_port or 80}",
        )

    try:
        raw_profiles = await client.get_media_profiles()
        profiles = []
        for p in raw_profiles:
            resolution = None
            if p.resolution:
                resolution = f"{p.resolution[0]}x{p.resolution[1]}"

            profiles.append(
                ONVIFProfile(
                    token=p.token,
                    name=p.name,
                    rtsp_url=p.rtsp_url,
                    encoding=p.encoding,
                    resolution=resolution,
                    fps=p.fps,
                )
            )

        return ONVIFProfilesResponse(camera_id=camera_id, profiles=profiles)
    finally:
        await client.disconnect()


@router.post(
    "/cameras/{camera_id}/apply-profile", response_model=ONVIFApplyProfileResponse
)
async def apply_profile(
    camera_id: int,
    request: ONVIFApplyProfileRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_admin_user),
) -> ONVIFApplyProfileResponse:
    """Apply an ONVIF profile's RTSP path to a camera.

    This updates the camera's RTSP path and port based on the selected
    profile's stream URL. The camera's stream will need to be restarted
    to use the new settings.

    Requires admin privileges.
    """
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Parse RTSP URL to extract path and port
    try:
        parsed = urlparse(request.rtsp_url)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid RTSP URL format",
        )

    # Extract path (default to "/" if empty)
    new_path = parsed.path or "/"
    if parsed.query:
        new_path = f"{new_path}?{parsed.query}"

    # Extract port (default to 554 for RTSP)
    new_port = parsed.port or 554

    # Update camera
    camera.path = new_path
    camera.port = new_port
    camera.onvif_profile_token = request.profile_token

    await db.commit()

    logger.info(
        f"Applied ONVIF profile {request.profile_token} to camera {camera.name}: "
        f"path={new_path}, port={new_port}"
    )

    return ONVIFApplyProfileResponse(
        success=True,
        camera_id=camera_id,
        new_path=new_path,
        new_port=new_port,
        profile_token=request.profile_token,
    )


@router.post(
    "/cameras/{camera_id}/events/subscribe", response_model=ONVIFEventsResponse
)
async def subscribe_events(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_admin_user),
) -> ONVIFEventsResponse:
    """Enable ONVIF event subscription for a camera.

    When enabled, the ONVIF event worker will subscribe to motion and
    analytics events from this camera. Events will appear in the unified
    detection timeline alongside ML detections.

    The camera must have ONVIF enabled and support events.

    Requires admin privileges.
    """
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not camera.onvif_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ONVIF is not enabled for this camera",
        )

    camera.onvif_events_enabled = True
    await db.commit()

    logger.info(f"Enabled ONVIF events for camera {camera.name}")

    return ONVIFEventsResponse(
        success=True,
        camera_id=camera_id,
        events_enabled=True,
    )


@router.post(
    "/cameras/{camera_id}/events/unsubscribe", response_model=ONVIFEventsResponse
)
async def unsubscribe_events(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_admin_user),
) -> ONVIFEventsResponse:
    """Disable ONVIF event subscription for a camera.

    The ONVIF event worker will stop receiving events from this camera.

    Requires admin privileges.
    """
    service = CameraService(db)
    camera = await service.get_by_id(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera.onvif_events_enabled = False
    await db.commit()

    logger.info(f"Disabled ONVIF events for camera {camera.name}")

    return ONVIFEventsResponse(
        success=True,
        camera_id=camera_id,
        events_enabled=False,
    )
