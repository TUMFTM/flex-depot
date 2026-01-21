import pandas as pd

REQUIRED_COLS = [
    "time",
    "Power_lower_kW",
    "Power_upper_kW",
    "Capacity_lower_kWh",
    "Capacity_upper_kWh",
]

def read_flexibility_bounds_csv(path: str, tz: str = "Europe/Berlin") -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Flexibility CSV must contain columns {missing}")

    # Robust for mixed offsets (+02/+01): parse to UTC, then convert
    ts = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_convert(tz)

    bounds = df[[
        "Power_lower_kW",
        "Power_upper_kW",
        "Capacity_lower_kWh",
        "Capacity_upper_kWh",
    ]].astype(float)

    bounds.index = ts
    bounds = bounds.sort_index()
    return bounds

def align_and_validate_flexibility_bounds(
    bounds: pd.DataFrame,
    time_index: pd.DatetimeIndex,
    *,
    require_complete: bool = True,
    check_monotone: bool = True,
    check_band_consistency: bool = True,
) -> pd.DataFrame:
    """
    Align flexibility bounds to a target time_index and optionally validate consistency.

    - Reindexes to time_index
    - Ensures no missing timestamps (if require_complete)
    - Ensures E_lower <= E_upper and P_lower <= P_upper (if check_band_consistency)
    """
    if not isinstance(bounds.index, pd.DatetimeIndex):
        raise ValueError("flexibility bounds must have a DatetimeIndex")

    b = bounds.sort_index().reindex(time_index)

    if require_complete and b.isnull().any().any():
        missing_idx = b[b.isnull().any(axis=1)].index[:5]
        raise ValueError(f"Missing flexibility bounds for timestamps (examples): {list(missing_idx)}")

    if check_band_consistency:
        bad_power = (b["Power_lower_kW"] > b["Power_upper_kW"])
        bad_energy = (b["Capacity_lower_kWh"] > b["Capacity_upper_kWh"])
        if bad_power.any():
            ex = b.index[bad_power][:5]
            raise ValueError(f"Inconsistent power band (lower > upper) at: {list(ex)}")
        if bad_energy.any():
            ex = b.index[bad_energy][:5]
            raise ValueError(f"Inconsistent energy band (lower > upper) at: {list(ex)}")

    if check_monotone and not b.index.is_monotonic_increasing:
        b = b.sort_index()

    return b