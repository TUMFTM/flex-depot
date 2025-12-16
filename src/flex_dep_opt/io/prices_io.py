# prices_io.py
# Utility functions for reading and writing day-ahead price data as CSV files.

import pandas as pd
from pathlib import Path
from flex_dep_opt.market.dayahead import DayAheadPrices
from flex_dep_opt.market.intraday import IntradayPrices

def read_prices_csv(path: str, tz: str = "Europe/Berlin") -> pd.Series:
    """Read a CSV file containing columns [time, price] into a timezone-aware pandas Series."""
    df = pd.read_csv(path)
    if "time" not in df or "price" not in df:
        raise ValueError("CSV must contain columns 'time' and 'price'")

    ts = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_convert(tz)
    s = pd.Series(df["price"].astype(float).values, index=ts).sort_index()
    s = s[~s.index.isna()]
    if s.isna().any():
        raise ValueError(f"prices contain NaNs: {path}")
    return s

def write_prices_csv(prices: pd.Series, path: str) -> str:
    """Write a pandas Series (with DatetimeIndex) to a CSV file."""
    df = pd.DataFrame({"time": prices.index, "price": prices.values})
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return str(output_path.resolve())

def build_prices_from_settings(settings):
    prices_by_market = {}
    if settings["optimization"]["markets"]["dayahead"]["enabled"]:
        da = DayAheadPrices.from_csv(settings["optimization"]["markets"]["dayahead"]["source"])
        prices_by_market["DA"] = da.prices_eur_per_kwh
    if settings["optimization"]["markets"]["intraday"]["enabled"]:
        idp = IntradayPrices.from_csv(settings["optimization"]["markets"]["intraday"]["source"])
        prices_by_market["ID"] = idp.prices_eur_per_kwh

    # Optional: imbalance / reBAP (pos/neg)
    imb_cfg = settings["optimization"].get("imbalance", {})
    if imb_cfg.get("enabled", False):
        pos = read_prices_csv(imb_cfg["source_pos"])
        neg = read_prices_csv(imb_cfg["source_neg"])
        prices_by_market["IMB_POS"] = pos
        prices_by_market["IMB_NEG"] = neg
    return prices_by_market