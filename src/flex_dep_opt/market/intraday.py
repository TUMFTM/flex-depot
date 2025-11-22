from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import pandas as pd

Unit = Literal["eur_per_kwh", "eur_per_mwh"]

@dataclass
class IntradayPrices:
    """Container for intraday prices (typically 15-min resolution),
    stored internally in EUR/kWh.
    """
    prices_eur_per_kwh: pd.Series

    def __post_init__(self) -> None:
        if not isinstance(self.prices_eur_per_kwh.index, pd.DatetimeIndex):
            raise ValueError("prices must have a DatetimeIndex")
        if self.prices_eur_per_kwh.index.tz is None:
            raise ValueError("timestamp index must be timezone-aware")
        if self.prices_eur_per_kwh.isna().any():
            raise ValueError("prices contain NaNs")

        # ensure sorting
        self.prices_eur_per_kwh = self.prices_eur_per_kwh.sort_index()

    @classmethod
    @classmethod
    def from_csv(
            cls,
            path: str,
            time_col: str = "time",
            price_col: str = "price",
            *,
            unit: Unit = "eur_per_kwh",
            tz: str = "Europe/Berlin",
    ) -> "IntradayPrices":
        """Load intraday prices from CSV.

        Robust against strings with offsets like '2025-10-01 00:00:00+02:00'.
        Normalizes everything to the target timezone `tz`.
        """
        df = pd.read_csv(path)
        if time_col not in df or price_col not in df:
            raise ValueError(f"CSV must have columns '{time_col}' and '{price_col}'")

        # 1) Parse as UTC → works for naive and offset-aware strings
        ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)

        # 2) Convert to target timezone
        ts = ts.dt.tz_convert(tz)

        # 3) Prices
        prices = df[price_col].astype(float)
        if unit == "eur_per_mwh":
            prices = prices / 1000.0

        s = pd.Series(prices.values, index=ts)
        s = s[~s.index.isna()]

        return cls(prices_eur_per_kwh=s)

    @property
    def duration(self) -> int:
        return len(self.prices_eur_per_kwh)

    def to_frame(self) -> pd.DataFrame:
        return self.prices_eur_per_kwh.to_frame(name="price_eur_per_kwh")

    def summary(self) -> dict:
        s = self.prices_eur_per_kwh
        return {
            "points": len(s),
            "start": s.index[0].isoformat(),
            "end": s.index[-1].isoformat(),
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
        }