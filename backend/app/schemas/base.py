"""Base schema with UTC datetime serialization."""

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict


def serialize_datetime(dt: datetime) -> str:
    """Serialize datetime to ISO 8601 format with Z suffix for UTC.

    Ensures all datetime values are serialized with timezone info,
    using Z suffix for UTC times.
    """
    if dt.tzinfo is None:
        # Naive datetime - assume UTC
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        # Convert to UTC
        dt = dt.astimezone(timezone.utc)

    # Format as ISO 8601 with Z suffix
    return dt.isoformat().replace("+00:00", "Z")


class UTCBaseModel(BaseModel):
    """Base model that serializes all datetimes as UTC with Z suffix.

    Use this as the base class for all API response schemas to ensure
    consistent timezone handling.
    """

    model_config = ConfigDict(from_attributes=True)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override to ensure datetime serialization in dict output."""
        data = super().model_dump(**kwargs)
        return self._serialize_datetimes(data)

    def model_dump_json(self, **kwargs: Any) -> str:
        """Override to ensure datetime serialization in JSON output."""
        data = self.model_dump(**kwargs)
        return json.dumps(data)

    def _serialize_datetimes(self, obj: Any) -> Any:
        """Recursively serialize datetime objects."""
        if isinstance(obj, datetime):
            return serialize_datetime(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_datetimes(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetimes(item) for item in obj]
        return obj
