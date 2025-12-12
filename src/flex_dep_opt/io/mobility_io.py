import pandas as pd
from pathlib import Path


def read_mobility_bounds_csv(path: str, tz: str = "Europe/Berlin") -> pd.DataFrame:
    """
    Read a CSV file with aggregated fleet bounds into a timezone-aware DataFrame.

    Expected columns:
        time,
        Power_lower_kW,
        Power_upper_kW,
        Capacity_lower_kWh,
        Capacity_upper_kWh

    Returns:
        DataFrame indexed by tz-aware DatetimeIndex in the given timezone
        with columns [P_lower_kW, P_upper_kW, C_lower_kWh, C_upper_kWh].
    """
    df = pd.read_csv(path)

    required_cols = [
        "time",
        "Power_lower_kW",
        "Power_upper_kW",
        "Capacity_lower_kWh",
        "Capacity_upper_kWh",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Mobility CSV must contain columns {missing}")

    # Zeitspalte: immer zuerst als UTC parsen, dann in gewünschte TZ konvertieren
    # → löst das Problem mit gemischten Offsets (+02, +01)
    time_parsed = pd.to_datetime(df["time"], errors="coerce", utc=True)
    ts = time_parsed.dt.tz_convert(tz)

    bounds = pd.DataFrame(
        {
            "P_lower_kW": df["Power_lower_kW"].astype(float).values,
            "P_upper_kW": df["Power_upper_kW"].astype(float).values,
            "C_lower_kWh": df["Capacity_lower_kWh"].astype(float).values,
            "C_upper_kWh": df["Capacity_upper_kWh"].astype(float).values,
        },
        index=ts,
    )

    bounds = bounds.sort_index()
    return bounds


def slice_mobility_bounds(
    bounds: pd.DataFrame, window_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Align full-horizon mobility bounds to a given MPC window index.

    Assumes both indices are tz-aware in the same timezone.
    """
    # harte Variante: alles muss da sein
    window_bounds = bounds.reindex(window_index)

    if window_bounds.isnull().any().any():
        missing_idx = window_bounds[window_bounds.isnull().any(axis=1)].index
        raise ValueError(
            f"Missing mobility bounds for some timestamps in window "
            f"(examples: {list(missing_idx[:5])})"
        )

    return window_bounds
