"""Camera management service."""

import asyncio
import logging
import shutil
import time
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Camera
from app.models.camera import CameraStatus
from app.schemas.camera import CameraCreate, CameraTestResult, CameraUpdate

logger = logging.getLogger(__name__)


class CameraService:
    """Service for camera CRUD operations and testing."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all(self) -> list[Camera]:
        """Get all cameras."""
        result = await self.db.execute(select(Camera).order_by(Camera.name))
        return list(result.scalars().all())

    async def get_by_id(self, camera_id: int) -> Optional[Camera]:
        """Get camera by ID."""
        result = await self.db.execute(
            select(Camera).where(Camera.id == camera_id)
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Optional[Camera]:
        """Get camera by name."""
        result = await self.db.execute(
            select(Camera).where(Camera.name == name)
        )
        return result.scalar_one_or_none()

    async def create(self, camera_data: CameraCreate) -> Camera:
        """Create a new camera."""
        camera = Camera(
            name=camera_data.name,
            host=camera_data.host,
            port=camera_data.port,
            path=camera_data.path,
            username=camera_data.username,
            password=camera_data.password,
            transport=camera_data.transport,
            recording_enabled=camera_data.recording_enabled,
            status=CameraStatus.UNKNOWN.value,
        )
        self.db.add(camera)
        await self.db.flush()
        await self.db.refresh(camera)
        return camera

    async def update(self, camera: Camera, camera_data: CameraUpdate) -> Camera:
        """Update an existing camera."""
        update_data = camera_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(camera, field, value)
        await self.db.flush()
        await self.db.refresh(camera)
        return camera

    async def delete(self, camera: Camera) -> None:
        """Delete a camera."""
        await self.db.delete(camera)
        await self.db.flush()

    async def update_status(
        self,
        camera: Camera,
        status: CameraStatus,
        error_message: Optional[str] = None,
    ) -> Camera:
        """Update camera status."""
        camera.status = status.value
        camera.error_message = error_message
        if status == CameraStatus.ONLINE:
            camera.last_seen = datetime.utcnow()
        await self.db.flush()
        await self.db.refresh(camera)
        return camera


async def test_camera_connection(camera: Camera) -> CameraTestResult:
    """Test camera RTSP connection using ffprobe."""
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        return CameraTestResult(
            success=False,
            message="ffprobe not found. Please install FFmpeg.",
        )

    rtsp_url = camera.rtsp_url
    start_time = time.monotonic()

    try:
        # Run ffprobe to test connection and get stream info
        proc = await asyncio.create_subprocess_exec(
            ffprobe_path,
            "-v", "error",
            "-rtsp_transport", camera.transport,
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate",
            "-of", "csv=p=0",
            "-timeout", "5000000",  # 5 second timeout in microseconds
            rtsp_url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=10.0,
        )

        duration_ms = int((time.monotonic() - start_time) * 1000)

        if proc.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Connection failed"
            return CameraTestResult(
                success=False,
                message=f"Connection failed: {error_msg}",
                duration_ms=duration_ms,
            )

        # Parse ffprobe output: codec,width,height,fps
        output = stdout.decode().strip()
        if not output:
            return CameraTestResult(
                success=True,
                message="Connected but no video stream found",
                duration_ms=duration_ms,
            )

        parts = output.split(",")
        codec = parts[0] if len(parts) > 0 else None
        width = parts[1] if len(parts) > 1 else None
        height = parts[2] if len(parts) > 2 else None
        fps_str = parts[3] if len(parts) > 3 else None

        resolution = f"{width}x{height}" if width and height else None

        # Parse frame rate (e.g., "30/1" -> 30.0)
        fps = None
        if fps_str and "/" in fps_str:
            try:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) != 0 else None
            except (ValueError, ZeroDivisionError):
                pass

        return CameraTestResult(
            success=True,
            message="Connection successful",
            codec=codec,
            resolution=resolution,
            fps=round(fps, 2) if fps else None,
            duration_ms=duration_ms,
        )

    except asyncio.TimeoutError:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        return CameraTestResult(
            success=False,
            message="Connection timed out after 10 seconds",
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        logger.exception("Error testing camera connection")
        return CameraTestResult(
            success=False,
            message=f"Error: {str(e)}",
            duration_ms=duration_ms,
        )
