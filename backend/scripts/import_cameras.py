#!/usr/bin/env python3
"""Import cameras from a JSON file.

This script imports camera configurations from a JSON file that was
previously exported using export_cameras.py.

Usage:
    python import_cameras.py -i cameras.json
    python import_cameras.py -i cameras.json --skip-existing
    python import_cameras.py -i cameras.json --update-existing
    python import_cameras.py -i cameras.json --dry-run
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.models.camera import Camera, CameraStatus


def validate_camera_data(data: dict[str, Any], index: int) -> list[str]:
    """Validate camera data from the import file.

    Args:
        data: The camera data dictionary
        index: The index in the cameras array (for error messages)

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if not data.get("name"):
        errors.append(f"Camera {index}: 'name' is required")
    if not data.get("host"):
        errors.append(f"Camera {index}: 'host' is required")

    port = data.get("port", 554)
    if not isinstance(port, int) or port < 1 or port > 65535:
        errors.append(f"Camera {index}: 'port' must be between 1 and 65535")

    transport = data.get("transport", "tcp")
    if transport not in ("tcp", "udp"):
        errors.append(f"Camera {index}: 'transport' must be 'tcp' or 'udp'")

    onvif_port = data.get("onvif_port")
    if onvif_port is not None:
        if not isinstance(onvif_port, int) or onvif_port < 1 or onvif_port > 65535:
            errors.append(f"Camera {index}: 'onvif_port' must be between 1 and 65535")

    return errors


async def import_cameras(
    input_path: Path,
    skip_existing: bool = False,
    update_existing: bool = False,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Import cameras from a JSON file.

    Args:
        input_path: Path to the input JSON file
        skip_existing: Skip cameras that already exist (by name)
        update_existing: Update cameras that already exist (by name)
        dry_run: Validate and report without making changes

    Returns:
        Tuple of (created, updated, skipped) counts
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    # Validate file format
    if "cameras" not in data:
        print("Error: Invalid file format - missing 'cameras' array")
        sys.exit(1)

    cameras_data = data["cameras"]
    version = data.get("version", "unknown")
    print(f"Import file version: {version}")
    print(f"Contains {len(cameras_data)} camera(s)")

    if data.get("includes_passwords"):
        print("Note: File includes camera passwords")
    else:
        print("Note: File does not include passwords")
    print()

    # Validate all cameras first
    all_errors = []
    for i, cam_data in enumerate(cameras_data):
        errors = validate_camera_data(cam_data, i)
        all_errors.extend(errors)

    if all_errors:
        print("Validation errors:")
        for error in all_errors:
            print(f"  - {error}")
        sys.exit(1)

    if dry_run:
        print("DRY RUN - no changes will be made\n")

    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    created = 0
    updated = 0
    skipped = 0

    async with async_session() as session:
        for cam_data in cameras_data:
            name = cam_data["name"]

            # Check if camera exists
            result = await session.execute(
                select(Camera).where(Camera.name == name)
            )
            existing = result.scalar_one_or_none()

            if existing:
                if update_existing:
                    if dry_run:
                        print(f"  Would update: {name}")
                    else:
                        # Update the existing camera
                        existing.host = cam_data["host"]
                        existing.port = cam_data.get("port", 554)
                        existing.path = cam_data.get("path", "/cam/realmonitor")
                        existing.username = cam_data.get("username")
                        if cam_data.get("password"):
                            existing.password = cam_data["password"]
                        existing.transport = cam_data.get("transport", "tcp")
                        existing.recording_enabled = cam_data.get(
                            "recording_enabled", True
                        )
                        # ONVIF settings
                        existing.onvif_port = cam_data.get("onvif_port")
                        existing.onvif_enabled = cam_data.get("onvif_enabled", False)
                        existing.onvif_profile_token = cam_data.get(
                            "onvif_profile_token"
                        )
                        existing.onvif_device_info = cam_data.get("onvif_device_info")
                        existing.onvif_events_enabled = cam_data.get(
                            "onvif_events_enabled", False
                        )
                        print(f"  Updated: {name}")
                    updated += 1
                elif skip_existing:
                    print(f"  Skipped (exists): {name}")
                    skipped += 1
                else:
                    print(f"  Error: Camera '{name}' already exists")
                    print("         Use --skip-existing or --update-existing")
                    await engine.dispose()
                    sys.exit(1)
            else:
                # Create new camera
                if dry_run:
                    print(f"  Would create: {name}")
                else:
                    path = cam_data.get("path", "/cam/realmonitor")
                    if not path.startswith("/"):
                        path = "/" + path

                    camera = Camera(
                        name=name,
                        host=cam_data["host"],
                        port=cam_data.get("port", 554),
                        path=path,
                        username=cam_data.get("username"),
                        password=cam_data.get("password"),
                        transport=cam_data.get("transport", "tcp"),
                        recording_enabled=cam_data.get("recording_enabled", True),
                        status=CameraStatus.UNKNOWN.value,
                        # ONVIF settings
                        onvif_port=cam_data.get("onvif_port"),
                        onvif_enabled=cam_data.get("onvif_enabled", False),
                        onvif_profile_token=cam_data.get("onvif_profile_token"),
                        onvif_device_info=cam_data.get("onvif_device_info"),
                        onvif_events_enabled=cam_data.get("onvif_events_enabled", False),
                    )
                    session.add(camera)
                    print(f"  Created: {name}")
                created += 1

        if not dry_run:
            await session.commit()

    await engine.dispose()

    return created, updated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import camera configurations from a JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python import_cameras.py -i cameras.json
    python import_cameras.py -i cameras.json --skip-existing
    python import_cameras.py -i cameras.json --update-existing
    python import_cameras.py -i cameras.json --dry-run

Duplicate handling:
    By default, the script will fail if a camera name already exists.
    Use --skip-existing to skip duplicates, or --update-existing to update them.
""",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input JSON file path",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cameras that already exist (by name)",
    )
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Update cameras that already exist (by name)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show what would happen without making changes",
    )

    args = parser.parse_args()

    if args.skip_existing and args.update_existing:
        print("Error: Cannot use both --skip-existing and --update-existing")
        sys.exit(1)

    if not args.input.exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    created, updated, skipped = asyncio.run(
        import_cameras(
            input_path=args.input,
            skip_existing=args.skip_existing,
            update_existing=args.update_existing,
            dry_run=args.dry_run,
        )
    )

    print()
    if args.dry_run:
        print("DRY RUN Summary:")
        print(f"  Would create: {created}")
        print(f"  Would update: {updated}")
        print(f"  Would skip: {skipped}")
    else:
        print("Import complete:")
        print(f"  Created: {created}")
        print(f"  Updated: {updated}")
        print(f"  Skipped: {skipped}")


if __name__ == "__main__":
    main()
