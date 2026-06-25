from __future__ import annotations

import pandas as pd


LOCAL_TIMEZONE = "Europe/Berlin"
INTERNAL_TIMEZONE = "UTC"


def local_config_timestamp_to_utc(value: str | pd.Timestamp, *, local_tz: str = LOCAL_TIMEZONE) -> pd.Timestamp:
    """
    Interpret a config timestamp as local market time and convert it to UTC.

    Config timestamps are expected to be local wall-clock values unless they
    already contain a timezone offset. Ambiguous/nonexistent DST wall-clock
    times are rejected instead of guessed.
    """
    ts = pd.to_datetime(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(local_tz, ambiguous="raise", nonexistent="raise")
    return ts.tz_convert(INTERNAL_TIMEZONE)


def validate_regular_index(index: pd.DatetimeIndex, *, timestep_hours: float, name: str) -> None:
    """Fail if a timezone-aware index is not strictly regular in UTC."""
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have a DatetimeIndex")
    if index.tz is None:
        raise ValueError(f"{name} must be timezone-aware")
    if index.has_duplicates:
        dup = index[index.duplicated()].unique()[:5]
        raise ValueError(f"{name} contains duplicate timestamps: {list(dup)}")
    if len(index) < 2:
        return

    expected = pd.Timedelta(hours=float(timestep_hours))
    idx_utc = index.tz_convert(INTERNAL_TIMEZONE)
    diffs = idx_utc.to_series().diff().dropna()
    bad = diffs[diffs != expected]
    if not bad.empty:
        examples = [(str(ts), str(delta)) for ts, delta in bad.head(5).items()]
        raise ValueError(
            f"{name} is not regular at {expected}. Bad UTC step examples: {examples}"
        )


def localize_datetime_index(index: pd.DatetimeIndex, *, tz: str = LOCAL_TIMEZONE) -> pd.DatetimeIndex:
    """Convert a timezone-aware DatetimeIndex to the configured output timezone."""
    if index.tz is None:
        raise ValueError("Cannot localize a timezone-naive DatetimeIndex")
    return index.tz_convert(tz)
