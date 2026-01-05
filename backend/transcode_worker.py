#!/usr/bin/env python3
"""Standalone transcode worker for re-encoding video files to save storage.

This script runs independently of the main backend server, scanning the
storage directory for completed recordings and re-encoding them with H.265
(HEVC) codec for better compression.

Supports both CPU encoding (libx265) and GPU encoding (NVENC) with automatic
detection. NVENC is 10-20x faster when an NVIDIA GPU is available.

Usage:
    ./transcode_worker.py                      # Run with defaults (auto-detect GPU)
    ./transcode_worker.py --gpu                # Force GPU encoding
    ./transcode_worker.py --no-gpu             # Force CPU encoding
    ./transcode_worker.py --crf 28             # Use CRF/CQ 28 (default)
    ./transcode_worker.py --dry-run            # Show what would be transcoded

The worker:
- Scans storage for .mp4 files not yet transcoded
- Auto-detects NVENC GPU support and uses it when available
- Re-encodes video to H.265 with configurable quality
- Copies audio stream without re-encoding
- Replaces original file atomically (write temp, then rename)
- Tracks completed files to avoid reprocessing
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

# Add the backend directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("transcode_worker")


class EncoderType(str, Enum):
    """Video encoder type."""

    CPU = "cpu"      # libx265 (software)
    NVENC = "nvenc"  # NVIDIA hardware encoder


@dataclass
class EncoderInfo:
    """Information about the selected encoder."""

    encoder_type: EncoderType
    encoder_name: str  # FFmpeg encoder name (libx265 or hevc_nvenc)
    quality_param: str  # Parameter name (-crf or -cq)
    preset_values: list[str]  # Valid preset values

    @property
    def is_gpu(self) -> bool:
        """Check if this is a GPU encoder."""
        return self.encoder_type == EncoderType.NVENC


# Encoder configurations
CPU_ENCODER = EncoderInfo(
    encoder_type=EncoderType.CPU,
    encoder_name="libx265",
    quality_param="-crf",
    preset_values=["ultrafast", "superfast", "veryfast", "faster", "fast",
                   "medium", "slow", "slower", "veryslow"],
)

NVENC_ENCODER = EncoderInfo(
    encoder_type=EncoderType.NVENC,
    encoder_name="hevc_nvenc",
    quality_param="-cq",
    preset_values=["p1", "p2", "p3", "p4", "p5", "p6", "p7"],  # p1=fastest, p7=best
)

# Preset mapping from CPU presets to NVENC presets
NVENC_PRESET_MAP = {
    "ultrafast": "p1",
    "superfast": "p2",
    "veryfast": "p3",
    "faster": "p3",
    "fast": "p4",
    "medium": "p4",
    "slow": "p5",
    "slower": "p6",
    "veryslow": "p7",
}


def detect_nvenc_support() -> bool:
    """Check if NVENC (NVIDIA GPU encoding) is available.

    Returns:
        True if hevc_nvenc encoder is available in FFmpeg
    """
    try:
        # Check if ffmpeg has hevc_nvenc encoder
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "hevc_nvenc" in result.stdout:
            # Try a quick encode test to verify GPU is accessible
            # This catches cases where the encoder exists but no GPU is available
            # Note: NVENC requires minimum 256x256 resolution
            test_result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
                    "-c:v", "hevc_nvenc", "-f", "null", "-"
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return test_result.returncode == 0
        return False
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        return False


def get_encoder(force_gpu: Optional[bool] = None) -> EncoderInfo:
    """Get the appropriate encoder based on availability and preference.

    Args:
        force_gpu: If True, require GPU. If False, use CPU. If None, auto-detect.

    Returns:
        EncoderInfo for the selected encoder

    Raises:
        RuntimeError: If GPU is forced but not available
    """
    nvenc_available = detect_nvenc_support()

    if force_gpu is True:
        if not nvenc_available:
            raise RuntimeError(
                "GPU encoding requested but NVENC is not available. "
                "Ensure NVIDIA drivers and CUDA are installed."
            )
        return NVENC_ENCODER

    if force_gpu is False:
        return CPU_ENCODER

    # Auto-detect: prefer GPU if available
    if nvenc_available:
        logger.info("NVENC GPU encoding detected and enabled")
        return NVENC_ENCODER
    else:
        logger.info("Using CPU encoding (libx265)")
        return CPU_ENCODER


@dataclass
class TranscodeResult:
    """Result of a transcode operation."""

    success: bool
    original_size: int
    new_size: int
    duration_seconds: float
    encoder_used: str = "unknown"
    error_message: Optional[str] = None

    @property
    def savings_percent(self) -> float:
        """Calculate percentage of space saved."""
        if self.original_size == 0:
            return 0.0
        return (1 - self.new_size / self.original_size) * 100

    @property
    def savings_mb(self) -> float:
        """Calculate MB saved."""
        return (self.original_size - self.new_size) / (1024 * 1024)


class TranscodeTracker:
    """Track which files have been transcoded."""

    def __init__(self, storage_root: Path):
        """Initialize tracker.

        Args:
            storage_root: Root storage directory
        """
        self.tracking_file = storage_root / ".transcode_status.json"
        self._data: dict = {"transcoded": {}, "failed": {}}
        self._load()

    def _load(self) -> None:
        """Load tracking data from file."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file) as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load tracking file: {e}")
                self._data = {"transcoded": {}, "failed": {}}

    def _save(self) -> None:
        """Save tracking data to file."""
        try:
            with open(self.tracking_file, "w") as f:
                json.dump(self._data, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Could not save tracking file: {e}")

    def is_transcoded(self, file_path: Path) -> bool:
        """Check if a file has been transcoded."""
        return str(file_path) in self._data.get("transcoded", {})

    def is_failed(self, file_path: Path) -> bool:
        """Check if a file failed transcoding."""
        return str(file_path) in self._data.get("failed", {})

    def mark_transcoded(
        self,
        file_path: Path,
        original_size: int,
        new_size: int,
        duration: float,
        encoder: str = "unknown",
    ) -> None:
        """Mark a file as transcoded."""
        self._data["transcoded"][str(file_path)] = {
            "original_size": original_size,
            "new_size": new_size,
            "savings_percent": round((1 - new_size / original_size) * 100, 1) if original_size > 0 else 0,
            "duration_seconds": round(duration, 1),
            "encoder": encoder,
            "transcoded_at": datetime.now(timezone.utc).isoformat(),
        }
        # Remove from failed if it was there
        self._data.get("failed", {}).pop(str(file_path), None)
        self._save()

    def mark_failed(self, file_path: Path, error: str) -> None:
        """Mark a file as failed."""
        self._data["failed"][str(file_path)] = {
            "error": error,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def is_locked(self, file_path: Path) -> bool:
        """Check if a file is currently being processed by another worker."""
        lock_file = file_path.parent / f".lock_{file_path.name}"
        if not lock_file.exists():
            return False
        # Check if lock is stale (older than 2 hours)
        try:
            lock_age = datetime.now(timezone.utc) - datetime.fromtimestamp(
                lock_file.stat().st_mtime, tz=timezone.utc
            )
            if lock_age.total_seconds() > 7200:  # 2 hours
                logger.warning(f"Removing stale lock: {lock_file}")
                lock_file.unlink()
                return False
        except OSError:
            pass
        return True

    def lock_file(self, file_path: Path) -> bool:
        """Try to acquire a lock on a file. Returns True if successful."""
        lock_file = file_path.parent / f".lock_{file_path.name}"
        try:
            # Use O_EXCL for atomic creation - fails if file exists
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            # Lock exists - check if it's stale (older than 30 minutes)
            # This handles workers that crashed or were restarted
            try:
                lock_age = datetime.now(timezone.utc) - datetime.fromtimestamp(
                    lock_file.stat().st_mtime, tz=timezone.utc
                )
                if lock_age.total_seconds() > 1800:  # 30 minutes
                    logger.warning(f"Removing stale lock: {lock_file}")
                    lock_file.unlink()
                    # Try again after removing stale lock
                    fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(os.getpid()).encode())
                    os.close(fd)
                    return True
            except (OSError, FileExistsError):
                # Another worker beat us to it, or other error
                pass
            return False
        except OSError as e:
            logger.warning(f"Could not create lock file: {e}")
            return False

    def unlock_file(self, file_path: Path) -> None:
        """Release the lock on a file."""
        lock_file = file_path.parent / f".lock_{file_path.name}"
        try:
            lock_file.unlink()
        except OSError:
            pass

    def get_stats(self) -> dict:
        """Get transcoding statistics."""
        transcoded = self._data.get("transcoded", {})
        total_original = sum(v.get("original_size", 0) for v in transcoded.values())
        total_new = sum(v.get("new_size", 0) for v in transcoded.values())

        # Count by encoder type
        encoder_counts: dict[str, int] = {}
        for v in transcoded.values():
            enc = v.get("encoder", "unknown")
            encoder_counts[enc] = encoder_counts.get(enc, 0) + 1

        return {
            "files_transcoded": len(transcoded),
            "files_failed": len(self._data.get("failed", {})),
            "total_original_gb": round(total_original / (1024**3), 2),
            "total_new_gb": round(total_new / (1024**3), 2),
            "total_savings_gb": round((total_original - total_new) / (1024**3), 2),
            "average_savings_percent": (
                round((1 - total_new / total_original) * 100, 1)
                if total_original > 0
                else 0
            ),
            "by_encoder": encoder_counts,
        }

    def clear_failed(self) -> int:
        """Clear failed entries to allow retry. Returns count cleared."""
        count = len(self._data.get("failed", {}))
        self._data["failed"] = {}
        self._save()
        return count

    def cleanup_stale_locks(self, max_age_seconds: int = 1800) -> int:
        """Remove all stale lock files from storage.

        This should be called on worker startup to clean up locks from
        workers that crashed or were restarted.

        Args:
            max_age_seconds: Consider locks older than this as stale (default 30 min)

        Returns:
            Number of stale locks removed
        """
        removed = 0
        now = datetime.now(timezone.utc)

        for lock_file in self.tracking_file.parent.rglob(".lock_*.mp4"):
            try:
                lock_age = now - datetime.fromtimestamp(
                    lock_file.stat().st_mtime, tz=timezone.utc
                )
                if lock_age.total_seconds() > max_age_seconds:
                    logger.info(f"Removing stale lock: {lock_file}")
                    lock_file.unlink()
                    removed += 1
            except OSError:
                pass

        return removed


class TranscodeWorker:
    """Worker that transcodes video files to save storage."""

    def __init__(
        self,
        storage_root: Path,
        crf: int = 28,
        preset: str = "medium",
        min_age_minutes: int = 20,
        dry_run: bool = False,
        force_gpu: Optional[bool] = None,
    ):
        """Initialize worker.

        Args:
            storage_root: Root storage directory
            crf: Quality value (CRF for CPU, CQ for GPU; 18-32 typical)
            preset: Encoding preset (auto-mapped for GPU)
            min_age_minutes: Minimum file age before transcoding
            dry_run: If True, only show what would be done
            force_gpu: True=require GPU, False=use CPU, None=auto-detect
        """
        self.storage_root = storage_root
        self.crf = crf
        self.preset = preset
        self.min_age = timedelta(minutes=min_age_minutes)
        self.dry_run = dry_run

        # Detect and configure encoder
        self.encoder = get_encoder(force_gpu)
        self._effective_preset = self._get_effective_preset()

        self.tracker = TranscodeTracker(storage_root)
        self._running = False
        self._current_file: Optional[Path] = None

    def _get_effective_preset(self) -> str:
        """Get the effective preset for the current encoder."""
        if self.encoder.is_gpu:
            # Map CPU preset to NVENC preset
            return NVENC_PRESET_MAP.get(self.preset, "p4")
        return self.preset

    def _get_video_info(self, file_path: Path) -> Optional[dict]:
        """Get video codec, bitrate, resolution, and duration.

        Returns:
            Dict with codec, bitrate, width, height, duration, or None on error
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name,bit_rate,width,height",
                    "-show_entries", "format=duration,bit_rate",
                    "-of", "json",
                    str(file_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return None

            import json
            data = json.loads(result.stdout)

            stream = data.get("streams", [{}])[0]
            fmt = data.get("format", {})

            # Get bitrate from stream or fall back to format bitrate
            bitrate = stream.get("bit_rate") or fmt.get("bit_rate")
            if bitrate:
                bitrate = int(bitrate)

            return {
                "codec": stream.get("codec_name", "").lower(),
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "bitrate": bitrate,
                "duration": float(fmt.get("duration", 0)),
            }
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
            return None

    def _should_skip_transcode(self, file_path: Path) -> tuple[bool, str]:
        """Check if file should be skipped based on codec and efficiency.

        Returns:
            Tuple of (should_skip, reason)
        """
        info = self._get_video_info(file_path)
        if not info:
            return False, ""  # Can't determine, try transcoding

        codec = info["codec"]
        bitrate = info["bitrate"]
        width = info["width"]
        height = info["height"]

        # If we can't determine bitrate, fall back to simple codec check
        if not bitrate or not width or not height:
            if codec in ("hevc", "h265"):
                return True, "already_hevc"
            return False, ""

        # Calculate bits per pixel per frame (assuming 30fps as baseline)
        # This normalizes bitrate across different resolutions
        pixels = width * height
        bpp = bitrate / (pixels * 30) if pixels > 0 else 0

        # HEVC at reasonable quality is typically 0.05-0.15 bpp
        # H.264 at reasonable quality is typically 0.1-0.3 bpp
        # Higher bpp = more bits = higher quality/less compression

        if codec in ("hevc", "h265"):
            # Skip HEVC files that are already efficiently encoded
            # Threshold: ~0.1 bpp is good quality HEVC, above that might benefit
            # from re-encode but risk is high. Be conservative.
            if bpp < 0.15:
                return True, "hevc_efficient"
            else:
                # High bitrate HEVC - could potentially save space but risky
                # Log it but still skip to avoid making files larger
                logger.info(
                    f"Skipping high-bitrate HEVC {file_path.name}: "
                    f"{bitrate/1_000_000:.1f} Mbps, {bpp:.3f} bpp"
                )
                return True, "hevc_high_bitrate"

        # H.264 - always worth transcoding to HEVC
        return False, ""

    def _is_already_hevc(self, file_path: Path) -> bool:
        """Check if a video file is already encoded with HEVC/H.265.

        Deprecated: Use _should_skip_transcode for smarter checks.
        """
        should_skip, _ = self._should_skip_transcode(file_path)
        return should_skip

    def _get_video_duration(self, file_path: Path) -> Optional[float]:
        """Get video duration in seconds."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    str(file_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
            return None

    def _build_ffmpeg_command(
        self,
        input_path: Path,
        output_path: Path,
    ) -> list[str]:
        """Build the FFmpeg command for transcoding.

        Args:
            input_path: Source video file
            output_path: Destination file

        Returns:
            FFmpeg command as list of arguments
        """
        cmd = ["ffmpeg"]

        if self.encoder.is_gpu:
            # Enable hardware-accelerated decoding with CUDA
            # This keeps frames on GPU: decode (NVDEC) -> encode (NVENC)
            cmd.extend([
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
            ])

        cmd.extend(["-i", str(input_path), "-c:v", self.encoder.encoder_name])

        if self.encoder.is_gpu:
            # NVENC-specific options
            cmd.extend([
                "-preset", self._effective_preset,
                "-rc", "vbr",  # Variable bitrate for quality-based encoding
                self.encoder.quality_param, str(self.crf),
                "-b_ref_mode", "middle",  # Better B-frame compression
            ])
        else:
            # libx265 options
            cmd.extend([
                "-preset", self._effective_preset,
                self.encoder.quality_param, str(self.crf),
            ])

        # Common options
        cmd.extend([
            "-vsync", "vfr",  # Preserve variable frame rate (prevents frame duplication)
            "-c:a", "copy",  # Copy audio without re-encoding
            "-tag:v", "hvc1",  # Apple compatibility tag
            "-y",  # Overwrite output
            str(output_path),
        ])

        return cmd

    def transcode_file(self, file_path: Path) -> TranscodeResult:
        """Transcode a single video file.

        Args:
            file_path: Path to the video file

        Returns:
            TranscodeResult with details of the operation
        """
        start_time = datetime.now()
        original_size = file_path.stat().st_size

        if self.dry_run:
            return TranscodeResult(
                success=True,
                original_size=original_size,
                new_size=original_size,
                duration_seconds=0,
                encoder_used=self.encoder.encoder_name,
            )

        # Create temp file in same directory (for atomic rename)
        temp_fd, temp_path = tempfile.mkstemp(
            suffix=".mp4",
            prefix=".transcode_",
            dir=file_path.parent,
        )
        os.close(temp_fd)
        temp_path = Path(temp_path)

        try:
            # Build FFmpeg command
            cmd = self._build_ffmpeg_command(file_path, temp_path)

            encoder_label = "GPU" if self.encoder.is_gpu else "CPU"
            logger.info(f"Transcoding ({encoder_label}): {file_path.name}")
            logger.debug(f"Command: {' '.join(cmd)}")

            # Run FFmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                raise RuntimeError(f"FFmpeg failed: {error_msg}")

            # Verify output file
            if not temp_path.exists():
                raise RuntimeError("Output file was not created")

            new_size = temp_path.stat().st_size
            if new_size < 1000:  # Less than 1KB is suspicious
                raise RuntimeError(f"Output file too small: {new_size} bytes")

            # Verify output is playable (quick check)
            duration = self._get_video_duration(temp_path)
            if duration is None or duration < 1:
                raise RuntimeError("Output file appears corrupted")

            # Atomic replace: rename temp to original
            temp_path.rename(file_path)

            elapsed = (datetime.now() - start_time).total_seconds()

            return TranscodeResult(
                success=True,
                original_size=original_size,
                new_size=new_size,
                duration_seconds=elapsed,
                encoder_used=self.encoder.encoder_name,
            )

        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

            elapsed = (datetime.now() - start_time).total_seconds()
            return TranscodeResult(
                success=False,
                original_size=original_size,
                new_size=original_size,
                duration_seconds=elapsed,
                encoder_used=self.encoder.encoder_name,
                error_message=str(e),
            )

    def _try_claim_file(self, mp4_file: Path) -> bool:
        """Try to claim a file for transcoding with lock-first approach.

        This method attempts to acquire a lock BEFORE checking other conditions,
        preventing race conditions when multiple workers are running.

        Args:
            mp4_file: Path to the video file

        Returns:
            True if this worker should process the file, False otherwise
        """
        # Skip hidden directories and temp files
        if any(part.startswith(".") for part in mp4_file.parts):
            return False
        if mp4_file.name.startswith("."):
            return False

        # Skip if already transcoded or failed
        if self.tracker.is_transcoded(mp4_file):
            return False
        if self.tracker.is_failed(mp4_file):
            return False

        # Try to acquire lock FIRST - this is the key to preventing races
        # If another worker is checking this file, only one will succeed
        if not self.tracker.lock_file(mp4_file):
            return False

        # Now we have the lock - do remaining checks
        # If any fail, we release the lock

        # Check file age
        now = datetime.now(timezone.utc)
        try:
            mtime = datetime.fromtimestamp(mp4_file.stat().st_mtime, tz=timezone.utc)
            if now - mtime < self.min_age:
                self.tracker.unlock_file(mp4_file)
                return False
        except OSError:
            self.tracker.unlock_file(mp4_file)
            return False

        # Check if file should be skipped (already HEVC or efficiently encoded)
        should_skip, skip_reason = self._should_skip_transcode(mp4_file)
        if should_skip:
            # Mark as transcoded so we skip it next time
            try:
                file_size = mp4_file.stat().st_size
                self.tracker.mark_transcoded(
                    mp4_file, file_size, file_size, 0, skip_reason or "skipped"
                )
            except OSError:
                pass
            logger.debug(f"Skipping {mp4_file.name} - {skip_reason}")
            self.tracker.unlock_file(mp4_file)
            return False

        # File is claimed and ready for transcoding
        return True

    async def run_once(self) -> dict:
        """Run one pass of transcoding.

        Returns:
            Summary statistics
        """
        processed = 0
        total_savings_mb = 0.0
        files_found = 0

        # Scan all .mp4 files and try to claim them one at a time
        # This lock-first approach prevents multiple workers from
        # grabbing the same file
        for mp4_file in self.storage_root.rglob("*.mp4"):
            if not self._running:
                logger.info("Worker stopped, exiting")
                break

            # Try to claim this file (acquires lock if successful)
            if not self._try_claim_file(mp4_file):
                continue

            files_found += 1
            self._current_file = mp4_file

            if self.dry_run:
                size_mb = mp4_file.stat().st_size / (1024 * 1024)
                logger.info(f"[DRY RUN] Would transcode: {mp4_file} ({size_mb:.1f} MB)")
                self.tracker.unlock_file(mp4_file)
                processed += 1
                self._current_file = None
                continue

            try:
                result = self.transcode_file(mp4_file)

                if result.success:
                    self.tracker.mark_transcoded(
                        mp4_file,
                        result.original_size,
                        result.new_size,
                        result.duration_seconds,
                        result.encoder_used,
                    )
                    # Calculate speed relative to video duration
                    video_duration = self._get_video_duration(mp4_file)
                    speed_str = ""
                    if video_duration and result.duration_seconds > 0:
                        speed = video_duration / result.duration_seconds
                        speed_str = f", {speed:.1f}x realtime"

                    logger.info(
                        f"Transcoded {mp4_file.name}: "
                        f"{result.savings_mb:.1f} MB saved ({result.savings_percent:.1f}%) "
                        f"in {result.duration_seconds:.1f}s{speed_str}"
                    )
                    processed += 1
                    total_savings_mb += result.savings_mb
                else:
                    self.tracker.mark_failed(mp4_file, result.error_message or "Unknown")
                    logger.error(f"Failed to transcode {mp4_file.name}: {result.error_message}")
            finally:
                self.tracker.unlock_file(mp4_file)

            self._current_file = None

        if files_found == 0:
            logger.info("No files to transcode")

        return {
            "files_found": files_found,
            "files_processed": processed,
            "total_savings_mb": round(total_savings_mb, 1),
            "encoder": self.encoder.encoder_name,
        }

    async def run_continuous(self, check_interval: int = 300) -> None:
        """Run continuously, checking for new files periodically.

        Args:
            check_interval: Seconds between checks
        """
        self._running = True
        encoder_label = "GPU (NVENC)" if self.encoder.is_gpu else "CPU (libx265)"
        logger.info(f"Starting continuous transcode worker (check every {check_interval}s)")
        logger.info(f"Encoder: {encoder_label}, Quality: {self.crf}, Preset: {self._effective_preset}")

        while self._running:
            try:
                await self.run_once()
                await asyncio.sleep(check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in transcode loop: {e}", exc_info=True)
                await asyncio.sleep(60)

        logger.info("Transcode worker stopped")

    def stop(self) -> None:
        """Stop the worker gracefully."""
        self._running = False
        if self._current_file:
            logger.info(f"Stopping after current file: {self._current_file.name}")


def main() -> None:
    """Main entry point for transcode worker CLI."""
    parser = argparse.ArgumentParser(
        description="Standalone worker for re-encoding videos to H.265",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quality values (CRF for CPU, CQ for GPU - same scale):
  18-22: Visually lossless, larger files
  23-28: Good quality, reasonable size (recommended for DVR)
  29-32: Smaller files, some quality loss (OK for surveillance)
  33+:   Very small files, noticeable quality loss

GPU Encoding (NVENC):
  - Automatically detected and used when available
  - 10-20x faster than CPU encoding
  - Requires NVIDIA GPU with NVENC support
  - Presets auto-mapped: medium->p4, slow->p5, etc.

Examples:
  ./transcode_worker.py                    # Auto-detect GPU, run once
  ./transcode_worker.py --continuous       # Run continuously
  ./transcode_worker.py --gpu              # Force GPU encoding
  ./transcode_worker.py --no-gpu           # Force CPU encoding
  ./transcode_worker.py --crf 32           # More aggressive compression
  ./transcode_worker.py --dry-run          # Show what would be done
  ./transcode_worker.py --stats            # Show transcoding statistics
  ./transcode_worker.py --check-gpu        # Check GPU availability and exit
        """,
    )

    # Load settings for defaults
    settings = get_settings()

    parser.add_argument(
        "--crf",
        type=int,
        default=None,
        help=f"Quality value - CRF for CPU, CQ for GPU (18-51, default: {settings.transcode_crf})",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                 "medium", "slow", "slower", "veryslow"],
        help=f"Encoding preset, auto-mapped for GPU (default: {settings.transcode_preset})",
    )
    parser.add_argument(
        "--storage-root",
        type=str,
        default=None,
        help="Storage root directory (default: from settings)",
    )
    parser.add_argument(
        "--min-age",
        type=int,
        default=None,
        help=f"Minimum file age in minutes before transcoding (default: {settings.transcode_min_age_minutes})",
    )
    parser.add_argument(
        "--continuous",
        "-c",
        action="store_true",
        help="Run continuously, checking for new files",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help=f"Check interval in seconds for continuous mode (default: {settings.transcode_check_interval})",
    )

    # GPU options (mutually exclusive)
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU encoding (fail if not available)",
    )
    gpu_group.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU encoding (ignore GPU even if available)",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be transcoded without doing it",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show transcoding statistics and exit",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Clear failed entries to allow retry",
    )
    parser.add_argument(
        "--check-gpu",
        action="store_true",
        help="Check GPU availability and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle check-gpu command
    if args.check_gpu:
        print("\nGPU Encoding Check:")
        print("-" * 40)

        # Check ffmpeg
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            print(f"  FFmpeg: {ffmpeg_path}")
        else:
            print("  FFmpeg: NOT FOUND")
            sys.exit(1)

        # Check NVENC
        nvenc_available = detect_nvenc_support()
        if nvenc_available:
            print("  NVENC:  Available (hevc_nvenc)")
            print("\n  GPU encoding will be used automatically.")
        else:
            print("  NVENC:  Not available")
            print("\n  Possible reasons:")
            print("    - No NVIDIA GPU present")
            print("    - NVIDIA drivers not installed")
            print("    - FFmpeg built without NVENC support")
            print("    - GPU doesn't support NVENC")
        return

    # Resolve configuration from args or settings
    crf = args.crf if args.crf is not None else settings.transcode_crf
    preset = args.preset if args.preset is not None else settings.transcode_preset
    min_age = args.min_age if args.min_age is not None else settings.transcode_min_age_minutes
    interval = args.interval if args.interval is not None else settings.transcode_check_interval
    storage_root = Path(args.storage_root) if args.storage_root else settings.storage_root
    storage_root = storage_root.resolve()

    # Determine GPU preference
    force_gpu: Optional[bool] = None
    if args.gpu:
        force_gpu = True
    elif args.no_gpu:
        force_gpu = False

    if not storage_root.exists():
        logger.error(f"Storage root does not exist: {storage_root}")
        sys.exit(1)

    logger.info(f"Storage root: {storage_root}")

    # Handle stats command
    if args.stats:
        tracker = TranscodeTracker(storage_root)
        stats = tracker.get_stats()
        print("\nTranscoding Statistics:")
        print(f"  Files transcoded: {stats['files_transcoded']}")
        print(f"  Files failed:     {stats['files_failed']}")
        print(f"  Original size:    {stats['total_original_gb']:.2f} GB")
        print(f"  Current size:     {stats['total_new_gb']:.2f} GB")
        print(f"  Space saved:      {stats['total_savings_gb']:.2f} GB")
        print(f"  Average savings:  {stats['average_savings_percent']:.1f}%")
        if stats.get("by_encoder"):
            print(f"  By encoder:       {stats['by_encoder']}")
        return

    # Handle retry-failed command
    if args.retry_failed:
        tracker = TranscodeTracker(storage_root)
        cleared = tracker.clear_failed()
        print(f"Cleared {cleared} failed entries")
        return

    # Create worker
    try:
        worker = TranscodeWorker(
            storage_root=storage_root,
            crf=crf,
            preset=preset,
            min_age_minutes=min_age,
            dry_run=args.dry_run,
            force_gpu=force_gpu,
        )
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    # Handle shutdown signals
    def signal_handler(sig: int, frame: object) -> None:
        logger.info(f"Received signal {sig}, shutting down...")
        worker.stop()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Clean up stale locks on startup
    stale_removed = worker.tracker.cleanup_stale_locks()
    if stale_removed > 0:
        logger.info(f"Cleaned up {stale_removed} stale lock(s) from previous run")

    # Run
    if args.continuous:
        asyncio.run(worker.run_continuous(interval))
    else:
        result = asyncio.run(worker.run_once())
        print(f"\nProcessed {result['files_processed']} of {result['files_found']} files")
        print(f"Encoder used: {result.get('encoder', 'unknown')}")
        if result.get('total_savings_mb', 0) > 0:
            print(f"Total space saved: {result['total_savings_mb']:.1f} MB")


if __name__ == "__main__":
    main()
