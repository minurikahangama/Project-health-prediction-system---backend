"""
Time helpers.

`utcnow()` replaces the deprecated `datetime.utcnow()` (removed-track since
Python 3.12) while preserving the project's existing convention of storing
*naive* UTC timestamps. All DateTime columns in models.py are naive, so this
returns a naive value to keep comparisons (e.g. expiry checks) consistent.
"""
from datetime import datetime, timezone


def utcnow() -> datetime:
    """Current UTC time as a naive datetime (no tzinfo), like datetime.utcnow()."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
