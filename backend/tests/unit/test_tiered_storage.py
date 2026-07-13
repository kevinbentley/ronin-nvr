"""Tests for tiered storage system."""

import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.recording import Recording, RecordingStatus, StorageTier
from app.services.playback import PlaybackInfo, PlaybackService, RecordingFile
from app.services import tier_migration as tm
from app.services.tier_migration import TierMigrationService


class TestStorageTier:
    """Tests for StorageTier enum."""

    def test_tier_values(self):
        """Test that tier enum has expected values."""
        assert StorageTier.HOT.value == "hot"
        assert StorageTier.WARM.value == "warm"
        assert StorageTier.COLD.value == "cold"


class TestRecordingFile:
    """Tests for RecordingFile with storage tier support."""

    def test_default_storage_tier(self):
        """Test that RecordingFile defaults to hot tier."""
        rf = RecordingFile(
            path=Path("/storage/Camera1/2024-01-01/12-00-00.mp4"),
            camera_name="Camera1",
            date=datetime(2024, 1, 1).date(),
            start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            size=1000000,
        )
        assert rf.storage_tier == StorageTier.HOT.value

    def test_storage_tier_can_be_set(self):
        """Test that storage tier can be set."""
        rf = RecordingFile(
            path=Path("/storage/Camera1/2024-01-01/12-00-00.mp4"),
            camera_name="Camera1",
            date=datetime(2024, 1, 1).date(),
            start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            size=1000000,
            storage_tier=StorageTier.WARM.value,
            storage_path="/warm/Camera1/2024-01-01/12-00-00.mp4",
        )
        assert rf.storage_tier == StorageTier.WARM.value
        assert rf.storage_path == "/warm/Camera1/2024-01-01/12-00-00.mp4"


class TestPlaybackInfo:
    """Tests for PlaybackInfo dataclass."""

    def test_hot_playback_info(self):
        """Test playback info for hot storage."""
        info = PlaybackInfo(
            url="/videos/Camera1/2024-01-01/12-00-00.mp4",
            tier=StorageTier.HOT.value,
            requires_loading=False,
        )
        assert info.url == "/videos/Camera1/2024-01-01/12-00-00.mp4"
        assert info.tier == "hot"
        assert info.requires_loading is False
        assert info.expires_in is None

    def test_cold_playback_info(self):
        """Test playback info for cold storage with presigned URL."""
        info = PlaybackInfo(
            url="https://s3.example.com/bucket/key?signature=xyz",
            tier=StorageTier.COLD.value,
            requires_loading=True,
            expires_in=3600,
        )
        assert "s3.example.com" in info.url
        assert info.tier == "cold"
        assert info.requires_loading is True
        assert info.expires_in == 3600


class TestPlaybackServiceTiers:
    """Tests for PlaybackService tier handling."""

    @pytest.fixture
    def playback_service(self, tmp_path):
        """Create a playback service with temp storage."""
        return PlaybackService(storage_root=tmp_path)

    def test_get_playback_info_hot(self, playback_service, tmp_path):
        """Test getting playback info for hot storage."""
        # Create a test file
        camera_dir = tmp_path / "Camera1" / "2024-01-01"
        camera_dir.mkdir(parents=True)
        test_file = camera_dir / "12-00-00.mp4"
        test_file.write_bytes(b"test video content")

        rf = RecordingFile(
            path=test_file,
            camera_name="Camera1",
            date=datetime(2024, 1, 1).date(),
            start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            size=test_file.stat().st_size,
            storage_tier=StorageTier.HOT.value,
        )

        info = playback_service.get_playback_info(rf)

        assert info.tier == "hot"
        assert info.requires_loading is False
        assert "/videos/" in info.url

    def test_get_playback_info_warm(self, playback_service, tmp_path):
        """Test getting playback info for warm storage."""
        warm_path = tmp_path / "warm" / "Camera1" / "2024-01-01" / "12-00-00.mp4"
        warm_path.parent.mkdir(parents=True)
        warm_path.write_bytes(b"test video content")

        rf = RecordingFile(
            path=Path("/original/path.mp4"),
            camera_name="Camera1",
            date=datetime(2024, 1, 1).date(),
            start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            size=warm_path.stat().st_size,
            storage_tier=StorageTier.WARM.value,
            storage_path=str(warm_path),
        )

        info = playback_service.get_playback_info(rf)

        assert info.tier == "warm"
        assert info.requires_loading is False
        assert "/videos-warm/" in info.url

    @patch("app.services.playback.get_s3_client")
    def test_get_playback_info_cold(self, mock_get_s3, playback_service):
        """Test getting playback info for cold storage generates presigned URL."""
        mock_s3 = MagicMock()
        mock_s3.is_configured.return_value = True
        mock_s3.generate_presigned_url.return_value = (
            "https://s3.example.com/bucket/key?signature=xyz"
        )
        mock_get_s3.return_value = mock_s3

        rf = RecordingFile(
            path=Path("/original/path.mp4"),
            camera_name="Camera1",
            date=datetime(2024, 1, 1).date(),
            start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            size=1000000,
            storage_tier=StorageTier.COLD.value,
            storage_path="ronin-nvr/Camera1/2024-01-01/12-00-00.mp4",
        )

        info = playback_service.get_playback_info(rf)

        assert info.tier == "cold"
        assert info.requires_loading is True
        assert "s3.example.com" in info.url
        mock_s3.generate_presigned_url.assert_called_once()


class TestS3StorageClient:
    """Tests for S3StorageClient."""

    @patch("boto3.client")
    def test_client_creation(self, mock_boto_client):
        """Test S3 client is created with correct parameters."""
        from app.services.s3_storage import S3StorageClient

        client = S3StorageClient(
            endpoint_url="http://minio:9000",
            bucket_name="test-bucket",
            access_key="test-key",
            secret_key="test-secret",
            region="us-west-2",
        )

        # Access client property to trigger creation
        _ = client.client

        mock_boto_client.assert_called_once()
        call_kwargs = mock_boto_client.call_args[1]
        assert call_kwargs["service_name"] == "s3"
        assert call_kwargs["region_name"] == "us-west-2"
        assert call_kwargs["endpoint_url"] == "http://minio:9000"

    def test_is_configured_missing_bucket(self):
        """Test is_configured returns False when bucket is missing."""
        from app.services.s3_storage import S3StorageClient

        client = S3StorageClient(
            bucket_name=None,
            access_key="test-key",
            secret_key="test-secret",
        )

        assert client.is_configured() is False

    def test_is_configured_all_present(self):
        """Test is_configured returns True when all required settings present."""
        from app.services.s3_storage import S3StorageClient

        client = S3StorageClient(
            bucket_name="test-bucket",
            access_key="test-key",
            secret_key="test-secret",
        )

        assert client.is_configured() is True

    def test_get_full_key_adds_prefix(self):
        """Test that keys are prefixed correctly."""
        from app.services.s3_storage import S3StorageClient

        client = S3StorageClient(
            bucket_name="test-bucket",
            access_key="test-key",
            secret_key="test-secret",
            prefix="ronin/",
        )

        full_key = client._get_full_key("Camera1/2024-01-01/12-00-00.mp4")
        assert full_key == "ronin/Camera1/2024-01-01/12-00-00.mp4"

    def test_get_full_key_already_prefixed(self):
        """Test that already-prefixed keys are not double-prefixed."""
        from app.services.s3_storage import S3StorageClient

        client = S3StorageClient(
            bucket_name="test-bucket",
            access_key="test-key",
            secret_key="test-secret",
            prefix="ronin/",
        )

        full_key = client._get_full_key("ronin/Camera1/2024-01-01/12-00-00.mp4")
        assert full_key == "ronin/Camera1/2024-01-01/12-00-00.mp4"


GB = 1024 ** 3


def _make_warm_recording(rec_id: int, warm_root: Path, size: int = 100) -> Recording:
    """Create a Recording backed by a real file on the warm filesystem."""
    fs_path = warm_root / f"Camera1/2024-01-01/{rec_id:02d}.mp4"
    fs_path.parent.mkdir(parents=True, exist_ok=True)
    fs_path.write_bytes(b"x" * size)
    rec = Recording(
        camera_id=1,
        file_path=f"/hot/Camera1/2024-01-01/{rec_id:02d}.mp4",
        file_size=size,
        start_time=datetime(2024, 1, 1, rec_id, 0, 0, tzinfo=timezone.utc),
        status=RecordingStatus.COMPLETED.value,
        storage_tier=StorageTier.WARM.value,
        storage_path=str(fs_path),
    )
    rec.id = rec_id
    return rec


@pytest.fixture
def warm_service(tmp_path, monkeypatch):
    """A migration service pointed at temp hot/warm dirs, cold disabled."""
    warm_root = tmp_path / "warm"
    warm_root.mkdir()
    monkeypatch.setattr(tm.settings, "cold_storage_enabled", False)
    monkeypatch.setattr(tm.settings, "warm_storage_enabled", True)
    monkeypatch.setattr(tm.settings, "tier_migration_headroom_gb", 0.0)
    svc = TierMigrationService(
        hot_storage_path=tmp_path / "hot",
        warm_storage_path=warm_root,
    )
    return svc


class TestShouldMigrateDiskFree:
    """should_migrate_from_tier honors the filesystem free-space threshold."""

    async def test_triggers_when_free_below_threshold(self, warm_service):
        warm_service._disk_free_bytes = MagicMock(return_value=10 * GB)
        result = await warm_service.should_migrate_from_tier(
            AsyncMock(), StorageTier.HOT.value,
            max_gb=None, retention_days=None,
            min_free_gb=50, disk_path=Path("/hot"),
        )
        assert result is True

    async def test_no_trigger_when_free_above_threshold(self, warm_service):
        warm_service._disk_free_bytes = MagicMock(return_value=100 * GB)
        result = await warm_service.should_migrate_from_tier(
            AsyncMock(), StorageTier.HOT.value,
            max_gb=None, retention_days=None,
            min_free_gb=50, disk_path=Path("/hot"),
        )
        assert result is False


class TestEvictOneFromWarm:
    """Eviction deletes the warm file and DB row when cold is disabled."""

    async def test_deletes_file_and_row(self, warm_service):
        rec = _make_warm_recording(1, warm_service.warm_storage_path, size=100)
        fs_path = Path(rec.storage_path)
        db = AsyncMock()

        reclaimed = await warm_service._evict_one_from_warm(rec, db)

        assert reclaimed == 100
        assert not fs_path.exists()
        db.delete.assert_awaited_once_with(rec)
        db.commit.assert_awaited()

    async def test_missing_file_still_removes_row(self, warm_service):
        rec = _make_warm_recording(1, warm_service.warm_storage_path, size=100)
        Path(rec.storage_path).unlink()
        db = AsyncMock()

        reclaimed = await warm_service._evict_one_from_warm(rec, db)

        assert reclaimed == 100  # falls back to recorded file_size
        db.delete.assert_awaited_once_with(rec)


class TestEnsureWarmRoom:
    """ensure_warm_room clears oldest warm data until the target is free."""

    async def test_evicts_until_enough_free(self, warm_service):
        recs = [_make_warm_recording(i, warm_service.warm_storage_path, 100)
                for i in (1, 2, 3)]
        warm_service.get_migration_candidates = AsyncMock(return_value=recs)

        def free_space(*_args):
            # Free space grows by 100 for each recording evicted from disk.
            evicted = sum(1 for r in recs if not Path(r.storage_path).exists())
            return 100 + evicted * 100

        warm_service._disk_free_bytes = MagicMock(side_effect=free_space)
        db = AsyncMock()

        freed = await warm_service.ensure_warm_room(db, needed_bytes=250)

        assert freed == 200  # two of three recordings evicted
        assert not Path(recs[0].storage_path).exists()
        assert not Path(recs[1].storage_path).exists()
        assert Path(recs[2].storage_path).exists()
        assert db.delete.await_count == 2

    async def test_noop_when_already_enough_free(self, warm_service):
        warm_service.get_migration_candidates = AsyncMock(return_value=[])
        warm_service._disk_free_bytes = MagicMock(return_value=500)
        db = AsyncMock()

        freed = await warm_service.ensure_warm_room(db, needed_bytes=250)

        assert freed == 0
        warm_service.get_migration_candidates.assert_not_awaited()

    async def test_stops_when_no_candidates_remain(self, warm_service):
        warm_service.get_migration_candidates = AsyncMock(return_value=[])
        warm_service._disk_free_bytes = MagicMock(return_value=10)
        db = AsyncMock()

        freed = await warm_service.ensure_warm_room(db, needed_bytes=250)

        assert freed == 0  # nothing to evict, does not loop forever


class TestMaintainWarm:
    """maintain_warm enforces the standing size cap."""

    async def test_evicts_until_under_cap(self, warm_service, monkeypatch):
        monkeypatch.setattr(tm.settings, "warm_max_gb", 1.0)
        monkeypatch.setattr(tm.settings, "warm_min_free_gb", None)
        recs = [_make_warm_recording(i, warm_service.warm_storage_path, 100)
                for i in (1, 2)]
        warm_service.get_migration_candidates = AsyncMock(return_value=recs)
        warm_service.get_tier_size = AsyncMock(
            side_effect=[2 * GB, 2 * GB, 0, 0]
        )
        db = AsyncMock()

        freed = await warm_service.maintain_warm(db)

        assert freed == 200
        assert db.delete.await_count == 2


def _make_hot_recording(rec_id: int, hot_root: Path, size: int = 100) -> Recording:
    """Create a HOT Recording backed by a real file on the hot filesystem."""
    fs_path = hot_root / f"Camera1/2024-01-01/{rec_id:02d}.mp4"
    fs_path.parent.mkdir(parents=True, exist_ok=True)
    fs_path.write_bytes(b"x" * size)
    rec = Recording(
        camera_id=1,
        file_path=str(fs_path),
        file_size=size,
        start_time=datetime(2024, 1, 1, rec_id, 0, 0, tzinfo=timezone.utc),
        status=RecordingStatus.COMPLETED.value,
        storage_tier=StorageTier.HOT.value,
    )
    rec.id = rec_id
    return rec


class TestMigrateToWarmDurability:
    """A committed hot->warm migration must survive the source vanishing."""

    async def test_source_deleted_after_commit_keeps_warm_copy(self, warm_service):
        """If the source disappears (e.g. transcode) right after the commit,
        the migration still succeeds and the warm copy is preserved."""
        hot_root = warm_service.hot_storage_path
        rec = _make_hot_recording(1, hot_root, size=128)
        source = Path(rec.file_path)
        dest = warm_service.warm_storage_path / "Camera1/2024-01-01/01.mp4"

        db = AsyncMock()

        def drop_source_on_commit():
            # Simulate a concurrent process removing the hot file during commit.
            source.unlink()

        db.commit.side_effect = drop_source_on_commit

        result = await warm_service.migrate_to_warm(rec, db)

        assert result is True
        assert rec.storage_tier == StorageTier.WARM.value
        assert dest.exists() and dest.read_bytes() == b"x" * 128
        assert not source.exists()

    async def test_failed_retry_keeps_existing_warm_copy(self, warm_service):
        """If the copy fails while a valid warm copy from a prior successful
        migration already exists at the destination, the failure cleanup must
        not delete that copy."""
        hot_root = warm_service.hot_storage_path
        rec = _make_hot_recording(2, hot_root, size=64)
        # Prior pass already produced a good warm copy at the destination.
        dest = warm_service.warm_storage_path / "Camera1/2024-01-01/02.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * 64)

        db = AsyncMock()
        # The source vanishes mid-copy, so the copy raises after the
        # pre-existence check (open(src) fails before dest is touched).
        with patch.object(
            tm.shutil, "copy2", side_effect=FileNotFoundError("source gone")
        ):
            result = await warm_service.migrate_to_warm(rec, db)

        assert result is False
        assert dest.exists() and dest.read_bytes() == b"x" * 64
