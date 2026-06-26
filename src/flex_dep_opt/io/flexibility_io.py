from __future__ import annotations

import pandas as pd

REQUIRED_COLS: list[str] = [
    "time",
    "Power_lower_kW",
    "Power_upper_kW",
    "Capacity_lower_kWh",
    "Capacity_upper_kWh",
]


# =============================================================================
# CSV I/O: flexibility bounds
# =============================================================================
def read_flexibility_bounds_csv(path: str, tz: str = "UTC") -> pd.DataFrame:
    """
    Read flexibility bounds CSV and return a time-indexed DataFrame.

    Expected columns
    ----------------
    time, Power_lower_kW, Power_upper_kW, Capacity_lower_kWh, Capacity_upper_kWh

    Timestamp handling
    ------------------
    - Parses time as UTC (or offset-aware strings) and converts to `tz`.
    - This is robust for mixed UTC offsets in the file (+01/+02).

    Validation
    ----------
    - Fails on unparsable timestamps (NaT)
    - Fails on NaNs / non-numeric bounds
    - Fails on duplicate timestamps
    - Ensures tz-aware DatetimeIndex
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Flexibility CSV must contain columns {missing}: {path}")

    # -------------------------------------------------------------------------
    # Parse timestamps and fail fast if any cannot be parsed
    # -------------------------------------------------------------------------
    ts_utc = pd.to_datetime(df["time"], errors="coerce", utc=True)
    if ts_utc.isna().any():
        bad_n = int(ts_utc.isna().sum())
        raise ValueError(f"Found {bad_n} unparsable timestamps in {path}")

    ts = ts_utc.dt.tz_convert(tz)

    # -------------------------------------------------------------------------
    # Extract bounds and enforce numeric dtype
    # -------------------------------------------------------------------------
    cols = [
        "Power_lower_kW",
        "Power_upper_kW",
        "Capacity_lower_kWh",
        "Capacity_upper_kWh",
    ]
    bounds = df[cols].apply(pd.to_numeric, errors="coerce")

    if bounds.isna().any().any():
        bad_rows = bounds[bounds.isna().any(axis=1)].index[:5].tolist()
        raise ValueError(f"Found NaNs in flexibility bounds in {path} (row examples): {bad_rows}")

    bounds.index = pd.DatetimeIndex(ts)
    bounds = bounds.sort_index()

    # Guard against duplicate timestamps
    if bounds.index.has_duplicates:
        dup = bounds.index[bounds.index.duplicated()].unique()[:5]
        raise ValueError(f"Duplicate timestamps in flexibility bounds {path} (examples): {list(dup)}")

    # Ensure tz-awareness (defensive: should always be true after tz_convert)
    if bounds.index.tz is None:
        raise ValueError(f"Flexibility bounds index must be timezone-aware after conversion: {path}")

    return bounds.astype(float)


# =============================================================================
# Alignment + validation utilities
# =============================================================================
def align_and_validate_flexibility_bounds(
    bounds: pd.DataFrame,
    time_index: pd.DatetimeIndex,
    *,
    expected_len: int | None = None,
) -> pd.DataFrame:
    """
    Align flexibility bounds to a target time_index and validate consistency.

    Steps
    -----
    1) Ensure DatetimeIndex and required columns
    2) (Optional) ensure `time_index` has an expected length (useful for N vs N+1)
    3) Sort and reindex to the target time_index
    4) Validate completeness (no missing timestamps after reindexing)
    5) Validate band consistency:
         - Power_lower_kW <= Power_upper_kW
         - Capacity_lower_kWh <= Capacity_upper_kWh

    Parameters
    ----------
    bounds:
        Raw bounds DataFrame (time-indexed).
    time_index:
        Target index to align to. For MPC/state alignment this is often the *state index* of length N+1.
    expected_len:
        Optional length check for `time_index`. This is a lightweight guardrail to avoid accidental mixing of decision index (N) and state index (N+1).
        Example: pass expected_len=N+1 when aligning state bounds.

    Returns
    -------
    pd.DataFrame
        Reindexed bounds, aligned to `time_index`.
    """
    if not isinstance(bounds.index, pd.DatetimeIndex):
        raise ValueError("flexibility bounds must have a DatetimeIndex")
    if bounds.index.tz is None:
        raise ValueError("flexibility bounds index must be timezone-aware")

    if not isinstance(time_index, pd.DatetimeIndex):
        raise ValueError("time_index must be a DatetimeIndex")
    if time_index.tz is None:
        raise ValueError("time_index must be timezone-aware")

    if expected_len is not None and len(time_index) != int(expected_len):
        raise ValueError(f"time_index length mismatch: got {len(time_index)}, expected {int(expected_len)}")

    missing_cols = [c for c in REQUIRED_COLS if c != "time" and c not in bounds.columns]
    if missing_cols:
        raise ValueError(f"flexibility bounds missing required columns: {missing_cols}")

    # Sort first (keeps behavior deterministic)
    b = bounds.sort_index().reindex(time_index)

    # -------------------------------------------------------------------------
    # Completeness check: do we have bounds for every required timestamp?
    # -------------------------------------------------------------------------
    if b.isnull().any().any():
        missing_idx = b[b.isnull().any(axis=1)].index[:5]
        raise ValueError(f"Missing flexibility bounds for timestamps (examples): {list(missing_idx)}")

    # -------------------------------------------------------------------------
    # Physical / logical consistency: lower bounds must not exceed upper bounds
    # -------------------------------------------------------------------------
    bad_power = b["Power_lower_kW"] > b["Power_upper_kW"]
    bad_energy = b["Capacity_lower_kWh"] > b["Capacity_upper_kWh"]

    if bad_power.any():
        ex = b.index[bad_power][:5]
        raise ValueError(f"Inconsistent power band (lower > upper) at: {list(ex)}")

    if bad_energy.any():
        ex = b.index[bad_energy][:5]
        raise ValueError(f"Inconsistent energy band (lower > upper) at: {list(ex)}")

    return b
