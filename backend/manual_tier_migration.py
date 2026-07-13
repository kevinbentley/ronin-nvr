#!/usr/bin/env python3
"""Manual tier migration / cleanup script.

When the automatic tier-migration system isn't keeping up (or is broken),
this script gives you a way to move files between tiers and delete the
oldest files from a tier to free up disk space.

It updates the database record for each file in the same way the automatic
migration service does, so the rest of the system stays consistent.

Subcommands:
    stats              Show recording counts/sizes per tier and disk usage.
    migrate-hot-warm   Copy files from hot to warm storage, then delete the
                       hot copy. Stops when hot is under --target-hot-gb,
                       when --free-gb has been freed from hot, or when
                       --max-files have been processed.
    delete-warm        Delete the oldest warm recordings (file + DB row).
                       Stops on the same conditions, applied to warm.

Path mapping
------------
Recording paths in the database use the container view (e.g.
``/data/storage/...`` for hot and ``/data/warm-storage/...`` for warm).
When you run this script on the host, pass ``--hot-fs-root`` and
``--warm-fs-root`` to point at the real mount points (for this deployment:
``/data/sas1/ronin/storage`` and ``/data/sas3/ronin-warm``).

Examples
--------
    # See what's where
    python manual_tier_migration.py stats

    # Free 1 TB of warm space by deleting oldest files (dry-run first)
    python manual_tier_migration.py delete-warm --free-gb 1000 --dry-run
    python manual_tier_migration.py delete-warm --free-gb 1000

    # Migrate oldest hot files into warm until hot is <= 9000 GB
    python manual_tier_migration.py migrate-hot-warm --target-hot-gb 9000

Run inside the container (recommended) with::

    docker compose exec backend python manual_tier_migration.py ...

Or from the host with the venv::

    backend/venv/bin/python backend/manual_tier_migration.py \\
        --hot-fs-root /data/sas1/ronin/storage \\
        --warm-fs-root /data/sas3/ronin-warm \\
        ...
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

_BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BACKEND_DIR))
# Match other backend scripts: run from backend/ so app.config picks up
# backend/.env (valid keys only) instead of the project-root .env (which
# contains compose-only keys like POSTGRES_USER that pydantic rejects).
os.chdir(_BACKEND_DIR)

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.recording import Recording, RecordingStatus, StorageTier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("manual_tier_migration")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"

DEFAULT_HOT_DB_ROOT = "/data/storage"
DEFAULT_WARM_DB_ROOT = "/data/warm-storage"


@dataclass
class PathMap:
    """Maps DB-stored container paths to real filesystem paths."""

    hot_db_root: str
    hot_fs_root: Path
    warm_db_root: str
    warm_fs_root: Path

    def to_fs(self, db_path: str, tier: str) -> Path:
        """Translate a DB-stored path to a real filesystem path."""
        if tier == StorageTier.HOT.value:
            db_root, fs_root = self.hot_db_root, self.hot_fs_root
        elif tier == StorageTier.WARM.value:
            db_root, fs_root = self.warm_db_root, self.warm_fs_root
        else:
            return Path(db_path)

        p = Path(db_path)
        try:
            rel = p.relative_to(db_root)
            return fs_root / rel
        except ValueError:
            # Path doesn't start with the expected root; trust it as-is.
            return p

    def warm_db_path_for(self, hot_db_path: str) -> str:
        """Compute the warm DB path that mirrors a given hot DB path."""
        try:
            rel = Path(hot_db_path).relative_to(self.hot_db_root)
        except ValueError:
            rel = Path(hot_db_path).name
        return str(Path(self.warm_db_root) / rel)


def ensure_postgres_up(timeout: int = 60) -> None:
    """Make sure the postgres container is reachable on localhost:5432.

    If we can't connect, try to ``docker compose up -d postgres`` and wait.
    No-op when already reachable or when running inside the container
    (where the host ``postgres`` resolves over the docker network).
    """
    host = os.environ.get("PGHOST", "localhost")
    port = int(os.environ.get("PGPORT", "5432"))

    def _try_connect() -> bool:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            return False

    if _try_connect():
        return

    if not COMPOSE_FILE.exists():
        logger.warning("Cannot reach %s:%d and no compose file found", host, port)
        return

    logger.info("Postgres not reachable on %s:%d, starting via docker compose...", host, port)
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "postgres"],
            check=True,
            cwd=str(PROJECT_ROOT),
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.error("Failed to start postgres via docker compose: %s", e)
        raise

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _try_connect():
            logger.info("Postgres is up.")
            time.sleep(1)  # give it a beat to finish accepting auth
            return
        time.sleep(1)

    raise RuntimeError(f"Timed out waiting for postgres at {host}:{port}")


def get_database_url(override: Optional[str]) -> str:
    """Pick a DB URL that works from wherever we're running."""
    if override:
        return override
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    # When running on the host (not inside the container), default to localhost.
    user = os.environ.get("POSTGRES_USER", "ronin_nvr_user")
    pw = os.environ.get("POSTGRES_PASSWORD", "ronin_pass")
    db = os.environ.get("POSTGRES_DB", "ronin_nvr")
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    return f"postgresql+asyncpg://{user}:{pw}@{host}:{port}/{db}"


async def tier_summary(db: AsyncSession, tier: str) -> dict:
    """DB-side stats for a tier."""
    base = (
        select(
            func.coalesce(func.sum(Recording.file_size), 0),
            func.count(),
            func.min(Recording.start_time),
            func.max(Recording.start_time),
        )
        .where(Recording.storage_tier == tier)
        .where(Recording.status == RecordingStatus.COMPLETED.value)
    )
    total_size, count, oldest, newest = (await db.execute(base)).one()
    return {
        "bytes": int(total_size or 0),
        "gb": float(total_size or 0) / (1024**3),
        "count": int(count or 0),
        "oldest": oldest,
        "newest": newest,
    }


async def cmd_stats(db: AsyncSession, paths: PathMap) -> None:
    """Show per-tier stats and actual disk usage."""
    for tier in (StorageTier.HOT.value, StorageTier.WARM.value, StorageTier.COLD.value):
        s = await tier_summary(db, tier)
        logger.info(
            "%-4s  files=%-7d  size=%8.2f GB  oldest=%s  newest=%s",
            tier,
            s["count"],
            s["gb"],
            s["oldest"],
            s["newest"],
        )

    for label, root in (("hot", paths.hot_fs_root), ("warm", paths.warm_fs_root)):
        if not root.exists():
            logger.info("disk %-4s  (path %s missing)", label, root)
            continue
        usage = shutil.disk_usage(root)
        logger.info(
            "disk %-4s  total=%.2f GB used=%.2f GB free=%.2f GB  at %s",
            label,
            usage.total / (1024**3),
            usage.used / (1024**3),
            usage.free / (1024**3),
            root,
        )


async def oldest_candidates(
    db: AsyncSession,
    tier: str,
    batch_size: int,
    exclude_ids: Optional[set[int]] = None,
) -> list[Recording]:
    """Oldest completed recordings on the given tier."""
    q = (
        select(Recording)
        .where(Recording.storage_tier == tier)
        .where(Recording.status == RecordingStatus.COMPLETED.value)
        .order_by(Recording.start_time.asc())
        .limit(batch_size)
    )
    if exclude_ids:
        q = q.where(Recording.id.notin_(exclude_ids))
    return list((await db.execute(q)).scalars().all())


def cleanup_empty_dirs(start_dir: Path, root: Path) -> None:
    """Remove empty parent directories up to (but not including) root."""
    try:
        d = start_dir
        while d != root and root in d.parents:
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
                d = d.parent
            else:
                break
    except OSError:
        pass


@dataclass
class RunBudget:
    """Stop conditions for a migration / deletion run."""

    target_tier_gb: Optional[float]
    free_gb: Optional[float]
    max_files: Optional[int]

    bytes_processed: int = 0
    files_processed: int = 0

    def remaining_files(self) -> Optional[int]:
        if self.max_files is None:
            return None
        return max(0, self.max_files - self.files_processed)

    def file_budget_exhausted(self) -> bool:
        return self.max_files is not None and self.files_processed >= self.max_files

    def free_budget_exhausted(self) -> bool:
        if self.free_gb is None:
            return False
        return (self.bytes_processed / (1024**3)) >= self.free_gb

    def tier_under_target(self, current_tier_gb: float) -> bool:
        if self.target_tier_gb is None:
            return False
        return current_tier_gb <= self.target_tier_gb

    def record(self, size_bytes: int) -> None:
        self.files_processed += 1
        self.bytes_processed += max(0, size_bytes)


async def cmd_migrate_hot_warm(
    db: AsyncSession,
    paths: PathMap,
    budget: RunBudget,
    dry_run: bool,
) -> None:
    """Copy oldest hot recordings to warm and delete the hot copy."""
    if not paths.warm_fs_root.exists():
        raise RuntimeError(f"Warm storage root does not exist: {paths.warm_fs_root}")

    failed = 0
    skipped_missing = 0
    migrated_count = 0
    seen_ids: set[int] = set()

    while True:
        if budget.file_budget_exhausted():
            logger.info("Stopping: hit --max-files limit (%d).", budget.max_files)
            break
        if budget.free_budget_exhausted():
            logger.info(
                "Stopping: freed %.2f GB from hot (>= --free-gb %s).",
                budget.bytes_processed / (1024**3),
                budget.free_gb,
            )
            break

        hot_stats = await tier_summary(db, StorageTier.HOT.value)
        if budget.tier_under_target(hot_stats["gb"]):
            logger.info(
                "Stopping: hot is at %.2f GB (<= --target-hot-gb %s).",
                hot_stats["gb"],
                budget.target_tier_gb,
            )
            break

        batch_size = min(50, budget.remaining_files() or 50)
        candidates = await oldest_candidates(
            db, StorageTier.HOT.value, batch_size, exclude_ids=seen_ids
        )
        if not candidates:
            logger.info("Stopping: no more hot recordings to migrate.")
            break

        for rec in candidates:
            if budget.file_budget_exhausted() or budget.free_budget_exhausted():
                break
            seen_ids.add(rec.id)

            src = paths.to_fs(rec.file_path, StorageTier.HOT.value)
            warm_db_path = paths.warm_db_path_for(rec.file_path)
            dst = paths.to_fs(warm_db_path, StorageTier.WARM.value)

            size = rec.file_size or 0

            if not src.exists():
                skipped_missing += 1
                # Count toward --max-files so a sea of orphans can't trap us
                # in an infinite-feeling loop.
                budget.files_processed += 1
                logger.warning("Missing on disk: %s (recording %d)", src, rec.id)
                if not dry_run:
                    rec.status = RecordingStatus.ERROR.value
                    rec.error_message = "File missing from disk during manual migration"
                    await db.commit()
                continue

            if dry_run:
                logger.info(
                    "DRY-RUN would migrate rec=%d %s -> %s (%.2f MB)",
                    rec.id,
                    src,
                    dst,
                    size / (1024**2),
                )
                budget.record(size)
                migrated_count += 1
                continue

            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(dst.suffix + ".part")
                shutil.copy2(src, tmp)
                if tmp.stat().st_size != src.stat().st_size:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError("size mismatch after copy")
                tmp.rename(dst)

                rec.storage_tier = StorageTier.WARM.value
                rec.storage_path = warm_db_path
                rec.migrated_at = datetime.utcnow()
                await db.commit()

                src.unlink()
                cleanup_empty_dirs(src.parent, paths.hot_fs_root)

                budget.record(size)
                migrated_count += 1
                logger.info(
                    "Migrated rec=%d  %.2f MB  hot->warm  (%d migrated, %.2f GB freed)",
                    rec.id,
                    size / (1024**2),
                    migrated_count,
                    budget.bytes_processed / (1024**3),
                )
            except Exception as e:
                failed += 1
                await db.rollback()
                logger.exception("Failed to migrate rec=%d: %s", rec.id, e)
                # Best-effort cleanup of partial copy.
                for p in (dst, dst.with_suffix(dst.suffix + ".part")):
                    try:
                        if p.exists() and p.stat().st_size != (src.stat().st_size if src.exists() else -1):
                            p.unlink()
                    except OSError:
                        pass

    logger.info(
        "Done. considered=%d migrated=%d freed_from_hot=%.2f GB failed=%d missing=%d",
        budget.files_processed,
        migrated_count,
        budget.bytes_processed / (1024**3),
        failed,
        skipped_missing,
    )


async def cmd_delete_warm(
    db: AsyncSession,
    paths: PathMap,
    budget: RunBudget,
    dry_run: bool,
) -> None:
    """Delete the oldest warm recordings (DB row + file)."""
    failed = 0
    skipped_missing = 0
    seen_ids: set[int] = set()

    while True:
        if budget.file_budget_exhausted():
            logger.info("Stopping: hit --max-files limit (%d).", budget.max_files)
            break
        if budget.free_budget_exhausted():
            logger.info(
                "Stopping: freed %.2f GB from warm (>= --free-gb %s).",
                budget.bytes_processed / (1024**3),
                budget.free_gb,
            )
            break

        warm_stats = await tier_summary(db, StorageTier.WARM.value)
        if budget.tier_under_target(warm_stats["gb"]):
            logger.info(
                "Stopping: warm is at %.2f GB (<= --target-warm-gb %s).",
                warm_stats["gb"],
                budget.target_tier_gb,
            )
            break

        batch_size = min(100, budget.remaining_files() or 100)
        candidates = await oldest_candidates(
            db, StorageTier.WARM.value, batch_size, exclude_ids=seen_ids
        )
        if not candidates:
            logger.info("Stopping: no more warm recordings to delete.")
            break

        for rec in candidates:
            if budget.file_budget_exhausted() or budget.free_budget_exhausted():
                break
            seen_ids.add(rec.id)

            warm_db_path = rec.storage_path or rec.file_path
            fs_path = paths.to_fs(warm_db_path, StorageTier.WARM.value)
            size = rec.file_size or 0

            if dry_run:
                exists = "ok" if fs_path.exists() else "missing"
                logger.info(
                    "DRY-RUN would delete rec=%d %s (%.2f MB, %s, start=%s)",
                    rec.id,
                    fs_path,
                    size / (1024**2),
                    exists,
                    rec.start_time,
                )
                budget.record(size)
                continue

            try:
                if fs_path.exists():
                    fs_path.unlink()
                    cleanup_empty_dirs(fs_path.parent, paths.warm_fs_root)
                else:
                    skipped_missing += 1
                    logger.warning("File missing on disk: %s (rec=%d)", fs_path, rec.id)

                await db.delete(rec)
                await db.commit()

                budget.record(size)
                logger.info(
                    "Deleted rec=%d  %.2f MB  warm  (%d files, %.2f GB total)",
                    rec.id,
                    size / (1024**2),
                    budget.files_processed,
                    budget.bytes_processed / (1024**3),
                )
            except Exception as e:
                failed += 1
                await db.rollback()
                logger.exception("Failed to delete rec=%d: %s", rec.id, e)

    logger.info(
        "Done. deleted=%d freed_from_warm=%.2f GB failed=%d missing=%d",
        budget.files_processed,
        budget.bytes_processed / (1024**3),
        failed,
        skipped_missing,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override DATABASE_URL (default: env var or localhost).",
    )
    parser.add_argument(
        "--hot-fs-root",
        type=Path,
        default=None,
        help="Real path to hot storage on disk (default: settings.storage_root).",
    )
    parser.add_argument(
        "--warm-fs-root",
        type=Path,
        default=None,
        help="Real path to warm storage on disk (default: settings.warm_storage_path).",
    )
    parser.add_argument(
        "--hot-db-root",
        default=DEFAULT_HOT_DB_ROOT,
        help=f"Prefix stored in DB for hot paths (default: {DEFAULT_HOT_DB_ROOT}).",
    )
    parser.add_argument(
        "--warm-db-root",
        default=DEFAULT_WARM_DB_ROOT,
        help=f"Prefix stored in DB for warm paths (default: {DEFAULT_WARM_DB_ROOT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without modifying the DB or filesystem.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", help="Show per-tier counts/sizes and disk usage.")

    p_migrate = sub.add_parser(
        "migrate-hot-warm",
        help="Migrate oldest hot recordings to warm storage.",
    )
    p_migrate.add_argument("--target-hot-gb", type=float, default=None,
                          help="Stop when hot tier <= this size in GB.")
    p_migrate.add_argument("--free-gb", type=float, default=None,
                          help="Stop after freeing this many GB from hot.")
    p_migrate.add_argument("--max-files", type=int, default=None,
                          help="Stop after processing this many files.")

    p_delete = sub.add_parser(
        "delete-warm",
        help="Delete oldest warm recordings (oldest start_time first).",
    )
    p_delete.add_argument("--target-warm-gb", type=float, default=None,
                         help="Stop when warm tier <= this size in GB.")
    p_delete.add_argument("--free-gb", type=float, default=None,
                         help="Stop after freeing this many GB from warm.")
    p_delete.add_argument("--max-files", type=int, default=None,
                         help="Stop after processing this many files.")

    return parser


def resolve_paths(args: argparse.Namespace) -> PathMap:
    """Build the PathMap from CLI args and (as fallback) app settings."""
    from app.config import get_settings

    settings = get_settings()
    hot_fs = args.hot_fs_root or Path(settings.storage_root)
    warm_fs = args.warm_fs_root or settings.warm_storage_path
    if warm_fs is None:
        warm_fs = Path("/data/warm-storage")
    warm_fs = Path(warm_fs)

    return PathMap(
        hot_db_root=args.hot_db_root,
        hot_fs_root=Path(hot_fs),
        warm_db_root=args.warm_db_root,
        warm_fs_root=warm_fs,
    )


async def run(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    logger.info(
        "Paths: hot db=%s fs=%s  |  warm db=%s fs=%s",
        paths.hot_db_root, paths.hot_fs_root,
        paths.warm_db_root, paths.warm_fs_root,
    )

    db_url = get_database_url(args.database_url)
    engine = create_async_engine(db_url, pool_pre_ping=True)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with session_maker() as db:
            if args.command == "stats":
                await cmd_stats(db, paths)
                return

            if args.command == "migrate-hot-warm":
                budget = RunBudget(
                    target_tier_gb=args.target_hot_gb,
                    free_gb=args.free_gb,
                    max_files=args.max_files,
                )
                if budget.target_tier_gb is None and budget.free_gb is None and budget.max_files is None:
                    raise SystemExit(
                        "Refusing to run unbounded. Pass at least one of "
                        "--target-hot-gb, --free-gb, --max-files."
                    )
                await cmd_migrate_hot_warm(db, paths, budget, args.dry_run)
                return

            if args.command == "delete-warm":
                budget = RunBudget(
                    target_tier_gb=args.target_warm_gb,
                    free_gb=args.free_gb,
                    max_files=args.max_files,
                )
                if budget.target_tier_gb is None and budget.free_gb is None and budget.max_files is None:
                    raise SystemExit(
                        "Refusing to run unbounded. Pass at least one of "
                        "--target-warm-gb, --free-gb, --max-files."
                    )
                await cmd_delete_warm(db, paths, budget, args.dry_run)
                return
    finally:
        await engine.dispose()


def main() -> None:
    args = build_parser().parse_args()
    ensure_postgres_up()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
