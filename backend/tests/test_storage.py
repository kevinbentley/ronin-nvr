"""Tests for storage API endpoints and retention service."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import AsyncClient

from app.services.retention import FileInfo, RetentionService, StorageStats


@pytest.mark.asyncio
async def test_get_storage_stats_empty(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Test getting storage stats when storage is empty."""
    # Create a mock RetentionService with empty storage
    empty_stats = StorageStats(
        total_size_bytes=0,
        total_files=0,
        oldest_file=None,
        newest_file=None,
        cameras={},
    )
    with patch("app.api.storage.retention_service") as mock_service:
        mock_service.get_stats.return_value = empty_stats

        response = await client.get("/api/storage/stats", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["total_size_bytes"] == 0
        assert data["total_files"] == 0
        assert data["cameras"] == []


@pytest.mark.asyncio
async def test_run_cleanup_empty(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    """Test running cleanup when storage is empty (admin only)."""
    with patch("app.api.storage.retention_service") as mock_service:
        mock_service.enforce_retention.return_value = {
            "files_scanned": 0,
            "files_deleted": 0,
            "bytes_freed": 0,
            "gb_freed": 0.0,
            "storage_before_gb": 0.0,
            "storage_after_gb": 0.0,
        }

        response = await client.post("/api/storage/cleanup", headers=admin_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["files_scanned"] == 0
        assert data["files_deleted"] == 0


@pytest.mark.asyncio
async def test_storage_unauthorized(client: AsyncClient) -> None:
    """Test that storage endpoints require authentication."""
    response = await client.get("/api/storage/stats")
    assert response.status_code == 401


class TestRetentionService:
    """Tests for RetentionService."""

    def test_scan_empty_storage(self) -> None:
        """Test scanning non-existent storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = RetentionService(Path(tmpdir) / "nonexistent")
            files, stats = service.scan_storage()

            assert files == []
            assert stats.total_files == 0
            assert stats.total_size_bytes == 0

    def test_scan_storage_with_files(self) -> None:
        """Test scanning storage with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_root = Path(tmpdir)

            # Create camera directory structure
            camera_dir = storage_root / "TestCamera" / "2024-01-15"
            camera_dir.mkdir(parents=True)

            # Create test files
            (camera_dir / "12-00-00.mp4").write_bytes(b"x" * 1000)
            (camera_dir / "12-15-00.mp4").write_bytes(b"x" * 2000)

            service = RetentionService(storage_root)
            files, stats = service.scan_storage()

            assert len(files) == 2
            assert stats.total_files == 2
            assert stats.total_size_bytes == 3000
            assert "TestCamera" in stats.cameras
            assert stats.cameras["TestCamera"]["file_count"] == 2

    def test_get_files_to_delete_by_age(self) -> None:
        """Test getting files to delete by age."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=40)
        recent_time = now - timedelta(days=5)

        files = [
            FileInfo(Path("/old.mp4"), 1000, old_time, "cam1"),
            FileInfo(Path("/recent.mp4"), 1000, recent_time, "cam1"),
        ]

        service = RetentionService()
        to_delete = service.get_files_to_delete(
            files, retention_days=30, max_size_gb=None
        )

        assert len(to_delete) == 1
        assert to_delete[0].path == Path("/old.mp4")

    def test_get_files_to_delete_by_size(self) -> None:
        """Test getting files to delete by size."""
        now = datetime.now(timezone.utc)

        files = [
            FileInfo(
                Path("/file1.mp4"), 500 * 1024 * 1024, now - timedelta(days=3), "cam1"
            ),
            FileInfo(
                Path("/file2.mp4"), 500 * 1024 * 1024, now - timedelta(days=2), "cam1"
            ),
            FileInfo(
                Path("/file3.mp4"), 500 * 1024 * 1024, now - timedelta(days=1), "cam1"
            ),
        ]

        service = RetentionService()
        # 1GB limit, but we have 1.5GB - should delete oldest
        to_delete = service.get_files_to_delete(
            files, retention_days=None, max_size_gb=1.0
        )

        assert len(to_delete) == 1
        assert to_delete[0].path == Path("/file1.mp4")

    def test_get_files_to_delete_combined(self) -> None:
        """Test getting files to delete with both policies."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=40)

        files = [
            FileInfo(Path("/old.mp4"), 100 * 1024 * 1024, old_time, "cam1"),
            FileInfo(
                Path("/file1.mp4"), 600 * 1024 * 1024, now - timedelta(days=2), "cam1"
            ),
            FileInfo(
                Path("/file2.mp4"), 600 * 1024 * 1024, now - timedelta(days=1), "cam1"
            ),
        ]

        service = RetentionService()
        # Old file deleted by age, then oldest remaining deleted by size
        to_delete = service.get_files_to_delete(
            files, retention_days=30, max_size_gb=1.0
        )

        assert len(to_delete) == 2
        paths = [f.path for f in to_delete]
        assert Path("/old.mp4") in paths
        assert Path("/file1.mp4") in paths

    def test_delete_files(self) -> None:
        """Test deleting files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_root = Path(tmpdir)
            camera_dir = storage_root / "TestCamera" / "2024-01-15"
            camera_dir.mkdir(parents=True)

            test_file = camera_dir / "12-00-00.mp4"
            test_file.write_bytes(b"x" * 1000)

            files = [FileInfo(test_file, 1000, datetime.now(timezone.utc), "TestCamera")]

            service = RetentionService(storage_root)
            deleted, freed = service.delete_files(files)

            assert deleted == 1
            assert freed == 1000
            assert not test_file.exists()

    def test_cleanup_empty_dirs(self) -> None:
        """Test that empty directories are cleaned up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_root = Path(tmpdir)
            camera_dir = storage_root / "TestCamera" / "2024-01-15"
            camera_dir.mkdir(parents=True)

            test_file = camera_dir / "12-00-00.mp4"
            test_file.write_bytes(b"x" * 1000)

            files = [FileInfo(test_file, 1000, datetime.now(timezone.utc), "TestCamera")]

            service = RetentionService(storage_root)
            service.delete_files(files)

            # Date directory should be removed (was empty after delete)
            assert not camera_dir.exists()
            # Camera directory should also be removed
            assert not (storage_root / "TestCamera").exists()
