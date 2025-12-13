import pandas as pd

REQUIRED_COLS = [
    "time",
    "Power_lower_kW",
    "Power_upper_kW",
    "Capacity_lower_kWh",
    "Capacity_upper_kWh",
]

def read_mobility_bounds_csv(path: str, tz: str = "Europe/Berlin") -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Mobility CSV must contain columns {missing}")

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

def slice_mobility_bounds(bounds: pd.DataFrame, window_index: pd.DatetimeIndex) -> pd.DataFrame:
    window_bounds = bounds.reindex(window_index)
    if window_bounds.isnull().any().any():
        missing_idx = window_bounds[window_bounds.isnull().any(axis=1)].index[:5]
        raise ValueError(f"Missing mobility bounds for timestamps (examples): {list(missing_idx)}")
    return window_bounds
