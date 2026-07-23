from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from flex_dep_opt.config.settings import Settings

PathLike = str | Path


# =============================================================================
# CSV I/O: generic price series (single source of truth for validation)
# =============================================================================
def read_prices_csv(path: PathLike, tz: str = "UTC") -> pd.Series:
    """
    Read a CSV file containing columns [time, price] into a timezone-aware Series.

    This function is the single source of truth for:
      - timestamp parsing (offset-aware input -> UTC -> tz conversion)
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
def build_prices_from_settings(settings: Settings, *, tz: str = "UTC") -> dict[str, pd.Series]:
    """
    Build price series by market from settings.

    Market codes
    ------------
    - "DA": day-ahead
    - "ID": intraday
    - "IMB_POS", "IMB_NEG": optional imbalance prices

    """
    prices_by_market: dict[str, pd.Series] = {}

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


def build_forecast_prices_from_settings(settings: Settings, *, tz: str = "UTC") -> dict[str, pd.Series]:
    """
    Build DECISION price series by market from settings (prices the MPC optimizes on).

    Starts from the realized series of `build_prices_from_settings()` and overrides
    the "DA" / "ID" entries with `read_prices_csv(forecast_source)` where a
    `forecast_source` is configured, so all realized-series validation (NaT, NaN,
    duplicates, sorting) applies to forecasts as well.

    Fallback semantics
    ------------------
    Markets without a configured `forecast_source` keep the realized series,
    i.e. perfect price foresight. Settlement (cashflows / KPIs) always uses
    `build_prices_from_settings()`; this builder only affects the prices seen
    inside the MPC optimization window.
    """
    prices_by_market = build_prices_from_settings(settings, tz=tz)

    mk_cfg = settings.optimization.markets

    if mk_cfg.dayahead.enabled and mk_cfg.dayahead.forecast_source:
        prices_by_market["DA"] = read_prices_csv(mk_cfg.dayahead.forecast_source, tz=tz)

    if mk_cfg.intraday.enabled and mk_cfg.intraday.forecast_source:
        prices_by_market["ID"] = read_prices_csv(mk_cfg.intraday.forecast_source, tz=tz)

    return prices_by_market


def build_fees_from_settings(settings: Settings) -> dict[str, float]:
    """
    Build a dict of per-market transaction fees [EUR/kWh] keyed by market code.
    """
    fees_by_market: dict[str, float] = {}

    mk_cfg = settings.optimization.markets

    if mk_cfg.dayahead.enabled:
        fees_by_market["DA"] = mk_cfg.dayahead.fee_eur_per_kwh

    if mk_cfg.intraday.enabled:
        fees_by_market["ID"] = mk_cfg.intraday.fee_eur_per_kwh

    return fees_by_market


# =============================================================================
# FCR I/O: capacity prices (XLSX) and grid-frequency data (CSV)
# =============================================================================
FCR_FREQ_DATETIME_COL = "DATETIME"
FCR_DROOP_COL = "FREQ_DROOP_MEAN"
FCR_DROOP_ABS_COL = "FREQ_DROOP_ABS_MEAN"


def get_fcr_prices(file_path: str) -> pd.Series:
    df = pd.read_excel(file_path)

    if "DATETIME_UTC" in df.columns:
        df["datetime"] = pd.to_datetime(df["DATETIME_UTC"], errors="coerce", utc=True)
        if df["datetime"].isna().any():
            raise ValueError(f"Unparsable timestamps in 'DATETIME_UTC' in {file_path}")
        df = df.set_index("datetime")
    else:
        df["start_hour"] = df["PRODUCTNAME"].str.split("_").str[1].astype(int)
        df["datetime"] = pd.to_datetime(df["DATE_FROM"]) + pd.to_timedelta(df["start_hour"], unit="h")
        df = df.set_index("datetime").tz_localize("Europe/Berlin", ambiguous="infer", nonexistent="shift_forward")

    price_col = "GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"
    prices = (
        df[price_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .rename("fcr_price")
        .sort_index()
    )

    if prices.index.has_duplicates:
        prices = prices.groupby(level=0).first()

    return prices


def get_fcr_frequency_data(file_path: str, tz: str = "Europe/Berlin") -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if FCR_DROOP_COL not in df.columns:
        raise ValueError(f"FCR frequency CSV missing required column: {FCR_DROOP_COL}")

    if FCR_FREQ_DATETIME_COL not in df.columns:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"FCR frequency CSV missing column: {FCR_FREQ_DATETIME_COL}")
    else:
        ts = pd.to_datetime(df[FCR_FREQ_DATETIME_COL], errors="coerce", utc=False)
        if ts.isna().any():
            raise ValueError(f"Unparsable timestamps in '{FCR_FREQ_DATETIME_COL}'")

        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        else:
            ts = ts.dt.tz_convert(tz)

        df.index = pd.DatetimeIndex(ts)
        df = df.drop(columns=[FCR_FREQ_DATETIME_COL])

    if not pd.api.types.is_numeric_dtype(df[FCR_DROOP_COL]):
        df[FCR_DROOP_COL] = df[FCR_DROOP_COL].astype(str).str.replace(",", ".", regex=False)

    df[FCR_DROOP_COL] = pd.to_numeric(df[FCR_DROOP_COL], errors="coerce")

    cols = [FCR_DROOP_COL]
    if FCR_DROOP_ABS_COL in df.columns:
        if not pd.api.types.is_numeric_dtype(df[FCR_DROOP_ABS_COL]):
            df[FCR_DROOP_ABS_COL] = df[FCR_DROOP_ABS_COL].astype(str).str.replace(",", ".", regex=False)
        df[FCR_DROOP_ABS_COL] = pd.to_numeric(df[FCR_DROOP_ABS_COL], errors="coerce")
        cols.append(FCR_DROOP_ABS_COL)

    df = df.sort_index()
    if df.index.has_duplicates:
        warnings.warn("Duplicate timestamps found; keeping first occurrence.", UserWarning)
        df = df[~df.index.duplicated(keep="first")]

    return df[cols]
