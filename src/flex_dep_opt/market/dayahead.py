from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Unit = Literal["eur_per_kwh", "eur_per_mwh"]


# =============================================================================
# Day-Ahead price container
# =============================================================================
@dataclass
class DayAheadPrices:
    """
    Container for day-ahead prices (typically 15-min resolution).
    -------------------
    - Parsing, validation, and normalization are centralized in `io/prices_io.py`.
    - This class is therefore a lightweight container that enforces only basic
      invariants (index type, tz-awareness) and provides convenience methods.

    Units
    -----
    - Internally stored in EUR/kWh.
    """
    prices_eur_per_kwh: pd.Series

    def __post_init__(self) -> None:
        if not isinstance(self.prices_eur_per_kwh.index, pd.DatetimeIndex):
            raise ValueError("prices must have a DatetimeIndex")
        if self.prices_eur_per_kwh.index.tz is None:
            raise ValueError("prices index must be timezone-aware")

        # Keep deterministic ordering; deeper validation happens in prices_io
        self.prices_eur_per_kwh = self.prices_eur_per_kwh.sort_index()

    @classmethod
    def from_series(cls, s: pd.Series, unit: Unit = "eur_per_kwh") -> "DayAheadPrices":
        """
        Build from a Series and convert units if needed.

        Note: Numeric/NaN/duplicate validation is expected to be handled upstream
        (prices_io) for data loaded from files.
        """
        if unit == "eur_per_mwh":
            s = s / 1000.0
        elif unit != "eur_per_kwh":
            raise ValueError(f"unknown unit: {unit}")
        return cls(prices_eur_per_kwh=s.astype(float))

    # Convenience helpers
    @property
    def hours(self) -> int:
        return int(len(self.prices_eur_per_kwh))

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
            "hours": int(len(s)),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
        }