# dayahead.py
# Defines the DayAheadPrices class, a validated container for hourly day-ahead
# electricity prices. It ensures timezone-aware timestamps, consistent units
# (EUR/kWh), and provides helper methods to load data from CSV, summarize,
# and access key statistics.

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import pandas as pd

Unit = Literal["eur_per_kwh", "eur_per_mwh"]

@dataclass
class DayAheadPrices:
    """Container for day-ahead prices with a tz-aware hourly DatetimeIndex.

    Internally we store prices in EUR/kWh to keep optimization units simple.
    """
    prices_eur_per_kwh: pd.Series

    def __post_init__(self) -> None:
        if not isinstance(self.prices_eur_per_kwh.index, pd.DatetimeIndex):
            raise ValueError("prices must have a DatetimeIndex")
        if self.prices_eur_per_kwh.index.tz is None:
            raise ValueError("prices index must be timezone-aware (e.g., Europe/Berlin)")
        if self.prices_eur_per_kwh.isna().any():
            raise ValueError("prices contain NaNs")
        # Ensure chronological order
        self.prices_eur_per_kwh = self.prices_eur_per_kwh.sort_index()

    @classmethod
    def from_series(
        cls,
        s: pd.Series,
        unit: Unit = "eur_per_kwh",
    ) -> "DayAheadPrices":
        """Create from a Series. Converts to EUR/kWh if needed."""
        if unit == "eur_per_mwh":
            s = s / 1000.0  # 1 MWh = 1000 kWh
        elif unit != "eur_per_kwh":
            raise ValueError(f"unknown unit: {unit}")
        return cls(prices_eur_per_kwh=s)

    @classmethod
    def from_csv(
            cls,
            path: str,
            time_col: str = "time",
            price_col: str = "price",
            *,
            unit: Unit = "eur_per_kwh",
            tz: str = "Europe/Berlin",
            parse_utc: bool = False,
    ) -> "DayAheadPrices":
        """Load from CSV with columns [time_col, price_col].

        - Robust gegen Strings mit Offset (z.B. '2025-10-01 00:00:00+02:00')
        - Normalisiert alles in eine einheitliche Ziel-TZ (tz).
        """
        df = pd.read_csv(path)
        if time_col not in df or price_col not in df:
            raise ValueError(f"CSV must have columns '{time_col}' and '{price_col}'")

        # 1) Immer als UTC parsen – das verträgt sowohl naive als auch offset-aware Strings
        ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)

        # 2) In gewünschte Ziel-TZ umrechnen
        ts = ts.dt.tz_convert(tz)

        # 3) Preis-Spalte
        prices = df[price_col].astype(float)

        # 4) Series bauen + NaT-Zeilen droppen
        s = pd.Series(prices.values, index=ts)
        s = s[~s.index.isna()]

        return cls.from_series(s, unit=unit)

    # Convenience helpers
    @property
    def hours(self) -> int:
        return len(self.prices_eur_per_kwh)

    @property
    def start(self) -> pd.Timestamp:
        return self.prices_eur_per_kwh.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.prices_eur_per_kwh.index[-1]

    def to_frame(self) -> pd.DataFrame:
        return self.prices_eur_per_kwh.to_frame(name="price_eur_per_kwh")

    def summary(self) -> dict:
        s = self.prices_eur_per_kwh
        return {
            "hours": len(s),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
        }
