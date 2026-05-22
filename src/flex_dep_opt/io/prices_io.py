from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import pandas as pd

from flex_dep_opt.config.settings import Settings

PathLike = Union[str, Path]


# =============================================================================
# CSV I/O: generic price series (single source of truth for validation)
# =============================================================================
def read_prices_csv(path: PathLike, tz: str = "Europe/Berlin") -> pd.Series:
    """
    Read a CSV file containing columns [time, price] into a timezone-aware Series.

    This function is the single source of truth for:
      - timestamp parsing (UTC -> tz conversion)
      - NaT detection (unparsable timestamps)
      - NaN / non-numeric price detection
      - duplicate timestamp detection
      - sorting by time

    Assumptions
    -----------
    - Timestamps are parsed with `utc=True`. This is robust for offset-aware strings.

    Parameters
    ----------
    path:
        Path to CSV file.
    tz:
        Target timezone for the resulting DatetimeIndex.

    Returns
    -------
    pd.Series
        A float Series indexed by tz-aware timestamps.

    Raises
    ------
    ValueError
        If required columns are missing, timestamps cannot be parsed, prices
        contain NaNs, or duplicate timestamps exist.
    """
    path = Path(path)
    df = pd.read_csv(path)

    required = {"time", "price"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns {sorted(missing)}: {path}")

    # -------------------------------------------------------------------------
    # Parse timestamps (UTC) and fail fast on any unparsable entries
    # -------------------------------------------------------------------------
    ts_utc = pd.to_datetime(df["time"], errors="coerce", utc=True)
    if ts_utc.isna().any():
        bad_n = int(ts_utc.isna().sum())
        raise ValueError(f"Found {bad_n} unparsable timestamps in {path}")

    ts = ts_utc.dt.tz_convert(tz)

    # -------------------------------------------------------------------------
    # Parse prices and fail fast on non-numeric entries
    # -------------------------------------------------------------------------
    prices = pd.to_numeric(df["price"], errors="coerce")
    if prices.isna().any():
        bad_n = int(prices.isna().sum())
        raise ValueError(f"Found {bad_n} non-numeric / NaN prices in {path}")

    s = pd.Series(prices.astype(float).to_numpy(), index=pd.DatetimeIndex(ts)).sort_index()
    s.name = "price"

    # -------------------------------------------------------------------------
    # Guard against duplicate timestamps (often caused by DST or bad merges)
    # -------------------------------------------------------------------------
    if s.index.has_duplicates:
        dup = s.index[s.index.duplicated()].unique()[:5]
        raise ValueError(f"Duplicate timestamps in {path} (examples): {list(dup)}")

    return s


# =============================================================================
# Settings-based builders
# =============================================================================
def build_prices_from_settings(settings: Settings, *, tz: str = "Europe/Berlin") -> Dict[str, pd.Series]:
    """
    Build price series by market from settings.

    Market codes
    ------------
    - "DA": day-ahead
    - "ID": intraday
    - "IMB_POS", "IMB_NEG": optional imbalance prices

    """
    prices_by_market: Dict[str, pd.Series] = {}

    mk_cfg = settings.optimization.markets

    if mk_cfg.dayahead.enabled:
        prices_by_market["DA"] = read_prices_csv(mk_cfg.dayahead.source, tz=tz)

    if mk_cfg.intraday.enabled:
        prices_by_market["ID"] = read_prices_csv(mk_cfg.intraday.source, tz=tz)

    # Optional: imbalance / reBAP (pos/neg)
    imb_cfg = settings.optimization.imbalance
    if imb_cfg.enabled:
        prices_by_market["IMB_POS"] = read_prices_csv(imb_cfg.source_pos, tz=tz)
        prices_by_market["IMB_NEG"] = read_prices_csv(imb_cfg.source_neg, tz=tz)

    return prices_by_market


def build_fees_from_settings(settings: Settings) -> Dict[str, float]:
    """
    Build a dict of per-market transaction fees [EUR/kWh] keyed by market code.
    """
    fees_by_market: Dict[str, float] = {}

    mk_cfg = settings.optimization.markets

    if mk_cfg.dayahead.enabled:
        fees_by_market["DA"] = mk_cfg.dayahead.fee_eur_per_kwh

    if mk_cfg.intraday.enabled:
        fees_by_market["ID"] = mk_cfg.intraday.fee_eur_per_kwh

    return fees_by_market
