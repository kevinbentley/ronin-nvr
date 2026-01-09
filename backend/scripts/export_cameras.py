#!/usr/bin/env python3
"""Export all configured cameras to a JSON file.

This script exports camera configurations from the database to a JSON file
that can be imported into another RoninNVR installation.

Usage:
    python export_cameras.py -o cameras.json
    python export_cameras.py -o cameras.json --include-passwords
    python export_cameras.py -o cameras.json --pretty
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.models.camera import Camera


def camera_to_dict(camera: Camera, include_password: bool = False) -> dict[str, Any]:
    """Convert a Camera model to a dictionary for export.

    Args:
        camera: The Camera model instance
        include_password: Whether to include the password in the export

    Returns:
        Dictionary with camera configuration
    """
    data: dict[str, Any] = {
        "name": camera.name,
        "host": camera.host,
        "port": camera.port,
        "path": camera.path,
        "username": camera.username,
        "transport": camera.transport,
        "recording_enabled": camera.recording_enabled,
        # ONVIF settings
        "onvif_port": camera.onvif_port,
        "onvif_enabled": camera.onvif_enabled,
        "onvif_profile_token": camera.onvif_profile_token,
        "onvif_device_info": camera.onvif_device_info,
        "onvif_events_enabled": camera.onvif_events_enabled,
    }

    if include_password and camera.password:
        data["password"] = camera.password

    return data


async def export_cameras(
    output_path: Path,
    include_passwords: bool = False,
    pretty: bool = False,
) -> int:
    """Export all cameras to a JSON file.

    Args:
        output_path: Path to the output JSON file
        include_passwords: Whether to include passwords in the export
        pretty: Whether to format the JSON with indentation

    Returns:
        Number of cameras exported
    """
    settings = get_settings()

    engine = create_async_engine(settings.database_url, echo=False)
    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        result = await session.execute(select(Camera).order_by(Camera.name))
        cameras = result.scalars().all()

        export_data = {
            "version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "camera_count": len(cameras),
            "includes_passwords": include_passwords,
            "cameras": [
                camera_to_dict(cam, include_password=include_passwords)
                for cam in cameras
            ],
        }

    await engine.dispose()

    indent = 2 if pretty else None
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=indent, ensure_ascii=False)

    return len(cameras)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export camera configurations to a JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python export_cameras.py -o cameras.json
    python export_cameras.py -o cameras.json --include-passwords
    python export_cameras.py -o cameras.json --pretty

Note: By default, passwords are NOT included in the export for security.
      Use --include-passwords if you need to preserve credentials.
""",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("cameras.json"),
        help="Output file path (default: cameras.json)",
    )
    parser.add_argument(
        "--include-passwords",
        action="store_true",
        help="Include camera passwords in the export (security risk!)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Format JSON with indentation for readability",
    )

    args = parser.parse_args()

    if args.include_passwords:
        print("WARNING: Passwords will be included in the export file.")
        print("         Store the file securely and delete after use.")
        print()

    count = asyncio.run(
        export_cameras(
            output_path=args.output,
            include_passwords=args.include_passwords,
            pretty=args.pretty,
        )
    )

    print(f"Exported {count} camera(s) to {args.output}")

    if not args.include_passwords:
        print("\nNote: Passwords were not exported. Use --include-passwords to include them.")


if __name__ == "__main__":
    main()
