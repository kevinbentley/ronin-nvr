"""Integration tests for HLS playlist and segment serving."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient

from app.services.camera_stream import CameraStream, StreamState, stream_manager


@pytest.fixture
async def camera_with_hls(
    client: AsyncClient, tmp_path: Path, admin_headers: dict[str, str]
) -> tuple[dict, CameraStream]:
    """Create camera and prepare mock HLS stream (requires admin)."""
    # Create camera via API
    camera_data = {
        "name": "HLS Test Camera",
        "host": "192.168.1.100",
        "port": 554,
        "path": "/stream",
    }
    response = await client.post(
        "/api/cameras", json=camera_data, headers=admin_headers
    )
    camera = response.json()

    # Create mock camera object for stream
    mock_cam = MagicMock()
    mock_cam.id = camera["id"]
    mock_cam.name = camera["name"]
    mock_cam.host = camera["host"]
    mock_cam.port = camera["port"]
    mock_cam.path = camera["path"]
    mock_cam.username = None
    mock_cam.password = None
    mock_cam.transport = "tcp"
    mock_cam.rtsp_url = f"rtsp://{camera['host']}:{camera['port']}{camera['path']}"

    # Create stream with temp storage
    stream = CameraStream(mock_cam, storage_root=tmp_path)
    stream._state = StreamState.RUNNING
    stream._recording_enabled = True

    # Create mock HLS files
    stream.hls_directory.mkdir(parents=True, exist_ok=True)

    playlist_content = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:2
#EXT-X-MEDIA-SEQUENCE:0
#EXTINF:2.000,
segment000.ts
#EXTINF:2.000,
segment001.ts
#EXTINF:2.000,
segment002.ts
"""
    stream.playlist_path.write_text(playlist_content)

    # Create mock segment files with some data
    for i in range(3):
        segment = stream.hls_directory / f"segment{i:03d}.ts"
        segment.write_bytes(b"\x47" * 188 * 10)  # TS packets start with 0x47

    # Register stream with global manager
    stream_manager._streams[camera["id"]] = stream

    yield camera, stream

    # Cleanup
    if camera["id"] in stream_manager._streams:
        del stream_manager._streams[camera["id"]]


class TestHLSPlaylist:
    """Tests for HLS playlist endpoint."""

    @pytest.mark.asyncio
    async def test_get_playlist_returns_m3u8_content_type(
        self,
        client: AsyncClient,
        camera_with_hls: tuple[dict, CameraStream],
        auth_headers: dict[str, str],
    ) -> None:
        """GET playlist returns correct content type."""
        camera, stream = camera_with_hls
        camera_id = camera["id"]

        response = await client.get(
            f"/api/cameras/{camera_id}/stream/hls/playlist.m3u8",
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert "application/vnd.apple.mpegurl" in response.headers["content-type"]

    @pytest.mark.asyncio
    async def test_get_playlist_has_no_cache_headers(
        self,
        client: AsyncClient,
        camera_with_hls: tuple[dict, CameraStream],
        auth_headers: dict[str, str],
    ) -> None:
        """Playlist response has no-cache headers for live streaming."""
        camera, stream = camera_with_hls
        camera_id = camera["id"]

        response = await client.get(
            f"/api/cameras/{camera_id}/stream/hls/playlist.m3u8",
            headers=auth_headers,
        )

        assert response.status_code == 200
        cache_control = response.headers.get("cache-control", "")
        assert "no-cache" in cache_control

    @pytest.mark.asyncio
    async def test_get_playlist_has_cors_headers(
        self,
        client: AsyncClient,
        camera_with_hls: tuple[dict, CameraStream],
        auth_headers: dict[str, str],
    ) -> None:
        """Playlist response has CORS headers."""
        camera, stream = camera_with_hls
        camera_id = camera["id"]

        response = await client.get(
            f"/api/cameras/{camera_id}/stream/hls/playlist.m3u8",
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"

    @pytest.mark.asyncio
    async def test_get_playlist_contains_segments(
        self,
        client: AsyncClient,
        camera_with_hls: tuple[dict, CameraStream],
        auth_headers: dict[str, str],
    ) -> None:
        """Playlist contains segment references."""
        camera, stream = camera_with_hls
        camera_id = camera["id"]

        response = await client.get(
            f"/api/cameras/{camera_id}/stream/hls/playlist.m3u8",
            headers=auth_headers,
        )

        assert response.status_code == 200
        content = response.text
        assert "#EXTM3U" in content
        assert "segment000.ts" in content

    @pytest.mark.asyncio
    async def test_get_playlist_for_nonexistent_camera(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """GET playlist for missing camera returns 404."""
        response = await client.get(
            "/api/cameras/99999/stream/hls/playlist.m3u8",
            headers=auth_headers,
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_playlist_when_stream_not_ready(
        self, client: AsyncClient, admin_headers: dict[str, str], auth_headers: dict[str, str]
    ) -> None:
        """GET playlist returns 503 when stream isn't ready."""
        # Create a camera but don't set up stream (requires admin)
        camera_data = {
            "name": "No Stream Camera",
            "host": "192.168.1.200",
            "port": 554,
            "path": "/stream",
        }
        response = await client.post(
            "/api/cameras", json=camera_data, headers=admin_headers
        )
        camera = response.json()

        # Request playlist - stream will try to start but fail (no real camera)
        response = await client.get(
            f"/api/cameras/{camera['id']}/stream/hls/playlist.m3u8",
            headers=auth_headers,
        )
        # Should be 503 because FFmpeg can't connect
        assert response.status_code in (500, 503)


class TestHLSSegments:
    """Tests for HLS segment endpoint."""

    @pytest.mark.asyncio
    async def test_get_segment_returns_video_content_type(
        self,
        client: AsyncClient,
        camera_with_hls: tuple[dict, CameraStream],
        auth_headers: dict[str, str],
    ) -> None:
        """GET segment returns video/mp2t content type."""
        camera, stream = camera_with_hls
        camera_id = camera["id"]

        response = await client.get(
            f"/api/cameras/{camera_id}/stream/hls/segment000.ts",
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp2t"

    @pytest.mark.asyncio
    async def test_get_segment_has_cache_headers(
        self,
        client: AsyncClient,
        camera_with_hls: tuple[dict, CameraStream],
        auth_headers: dict[str, str],
    ) -> None:
        """Segment response has cache headers (segments are immutable)."""
        camera, stream = camera_with_hls
        camera_id = camera["id"]

        response = await client.get(
            f"/api/cameras/{camera_id}/stream/hls/segment000.ts",
            headers=auth_headers,
        )

        assert response.status_code == 200
        cache_control = response.headers.get("cache-control", "")
        assert "max-age" in cache_control

    @pytest.mark.asyncio
    async def test_get_segment_has_cors_headers(
        self,
        client: AsyncClient,
        camera_with_hls: tuple[dict, CameraStream],
        auth_headers: dict[str, str],
    ) -> None:
        """Segment response has CORS headers."""
        camera, stream = camera_with_hls
        camera_id = camera["id"]

        response = await client.get(
            f"/api/cameras/{camera_id}/stream/hls/segment000.ts",
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"

    @pytest.mark.asyncio
    async def test_get_missing_segment_returns_404(
        self,
        client: AsyncClient,
        camera_with_hls: tuple[dict, CameraStream],
        auth_headers: dict[str, str],
    ) -> None:
        """GET missing segment returns 404."""
        camera, stream = camera_with_hls
        camera_id = camera["id"]

        response = await client.get(
            f"/api/cameras/{camera_id}/stream/hls/segment999.ts",
            headers=auth_headers,
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_segment_with_invalid_extension_rejected(
        self, client: AsyncClient, admin_headers: dict[str, str], auth_headers: dict[str, str]
    ) -> None:
        """Non-.ts segment requests are rejected."""
        # First create a camera (requires admin)
        camera_data = {"name": "Segment Test", "host": "192.168.1.1"}
        response = await client.post(
            "/api/cameras", json=camera_data, headers=admin_headers
        )
        camera_id = response.json()["id"]

        response = await client.get(
            f"/api/cameras/{camera_id}/stream/hls/malicious.exe",
            headers=auth_headers,
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_get_multiple_segments(
        self,
        client: AsyncClient,
        camera_with_hls: tuple[dict, CameraStream],
        auth_headers: dict[str, str],
    ) -> None:
        """Can retrieve multiple different segments."""
        camera, stream = camera_with_hls
        camera_id = camera["id"]

        for i in range(3):
            response = await client.get(
                f"/api/cameras/{camera_id}/stream/hls/segment{i:03d}.ts",
                headers=auth_headers,
            )
            assert response.status_code == 200
            assert len(response.content) > 0


class TestConcurrentPlaylistRequests:
    """Tests for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_playlist_requests_dont_race(
        self,
        client: AsyncClient,
        camera_with_hls: tuple[dict, CameraStream],
        auth_headers: dict[str, str],
    ) -> None:
        """Multiple concurrent playlist requests are handled correctly."""
        camera, stream = camera_with_hls
        camera_id = camera["id"]

        # Fire multiple concurrent requests
        tasks = [
            client.get(
                f"/api/cameras/{camera_id}/stream/hls/playlist.m3u8",
                headers=auth_headers,
            )
            for _ in range(5)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        for resp in responses:
            if isinstance(resp, Exception):
                pytest.fail(f"Concurrent request caused exception: {resp}")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_concurrent_segment_requests(
        self,
        client: AsyncClient,
        camera_with_hls: tuple[dict, CameraStream],
        auth_headers: dict[str, str],
    ) -> None:
        """Multiple concurrent segment requests are handled correctly."""
        camera, stream = camera_with_hls
        camera_id = camera["id"]

        # Request different segments concurrently
        tasks = [
            client.get(
                f"/api/cameras/{camera_id}/stream/hls/segment{i:03d}.ts",
                headers=auth_headers,
            )
            for i in range(3)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for resp in responses:
            if isinstance(resp, Exception):
                pytest.fail(f"Concurrent request caused exception: {resp}")
            assert resp.status_code == 200
