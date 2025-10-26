# prices_io.py
# Utility functions for reading and writing day-ahead price data as CSV files.

import pandas as pd
from pathlib import Path

def read_prices_csv(path: str, tz: str = "Europe/Berlin") -> pd.Series:
    """Read a CSV file containing columns [time, price] into a timezone-aware pandas Series."""
    df = pd.read_csv(path)
    if "time" not in df or "price" not in df:
        raise ValueError("CSV must contain columns 'time' and 'price'")

    ts = pd.to_datetime(df["time"], errors="coerce").dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    s = pd.Series(df["price"].astype(float).values, index=ts)
    s = s.sort_index()
    return s

def write_prices_csv(prices: pd.Series, path: str) -> str:
    """Write a pandas Series (with DatetimeIndex) to a CSV file."""
    df = pd.DataFrame({"time": prices.index, "price": prices.values})
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return str(output_path.resolve())