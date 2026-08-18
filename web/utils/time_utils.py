"""UTC timestamp helpers shared by the web application."""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with an explicit offset."""
    return utc_now().isoformat()


def parse_timestamp(value: str | datetime) -> datetime:
    """Parse an ISO-8601 timestamp and normalize it to UTC.

    Legacy records without a timezone were written by a container running in
    UTC, so naive timestamps are interpreted as UTC for backward compatibility.
    """
    if isinstance(value, datetime):
        timestamp = value
    else:
        if not isinstance(value, str):
            raise TypeError("Timestamp must be a string or datetime")
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)
