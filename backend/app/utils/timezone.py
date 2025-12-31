"""Timezone utilities for consistent UTC handling throughout the application.

All timestamps in the database and API responses should be in UTC.
The frontend is responsible for converting to the user's local timezone.
"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Get current UTC time as a timezone-aware datetime.

    Returns:
        Current time in UTC with timezone info attached.
    """
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware UTC.

    If the datetime is naive (no timezone), assume it's UTC and attach the timezone.
    If it already has a timezone, convert to UTC.

    Args:
        dt: A datetime object (naive or timezone-aware)

    Returns:
        Timezone-aware datetime in UTC
    """
    if dt.tzinfo is None:
        # Naive datetime - assume UTC
        return dt.replace(tzinfo=timezone.utc)
    else:
        # Convert to UTC
        return dt.astimezone(timezone.utc)


def to_utc_isoformat(dt: datetime) -> str:
    """Convert datetime to ISO 8601 format with Z suffix for UTC.

    Args:
        dt: A datetime object

    Returns:
        ISO 8601 string ending with 'Z' for UTC
    """
    utc_dt = ensure_utc(dt)
    # Replace +00:00 with Z for cleaner output
    return utc_dt.isoformat().replace("+00:00", "Z")
