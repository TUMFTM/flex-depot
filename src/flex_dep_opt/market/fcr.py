import warnings
from zoneinfo import ZoneInfo

import pandas as pd

FCR_FREQ_DATETIME_COL = "DATETIME"
FCR_DROOP_COL         = "FREQ_DROOP_MEAN"


def fcr_gate_closure_timestamp(
    slot_start: pd.Timestamp,
    *,
    hour: str = "08:00",
    closes_previous_day: bool = True,
    timezone: str = "Europe/Berlin",
) -> pd.Timestamp:
    """
    Capacity-market gate-closure timestamp for an FCR 4 h slot.
    """
    hh, mm = (int(x) for x in hour.split(":"))
    day = slot_start.normalize()
    if closes_previous_day:
        day = day - pd.Timedelta(days=1)
    return day.replace(hour=hh, minute=mm, tzinfo=ZoneInfo(timezone))


def get_fcr_frequency_data(file_path: str, tz: str = "Europe/Berlin") -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if FCR_DROOP_COL not in df.columns:
        raise ValueError(f"FCR frequency CSV missing required column: {FCR_DROOP_COL}")

    column = FCR_DROOP_COL

    if FCR_FREQ_DATETIME_COL not in df.columns:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"FCR frequency CSV missing column: {FCR_FREQ_DATETIME_COL}")
    else:
        ts = pd.to_datetime(df[FCR_FREQ_DATETIME_COL], errors="coerce", utc=False)
        if ts.isna().any():
            raise ValueError(f"Unparsable timestamps in '{FCR_FREQ_DATETIME_COL}'")

        if ts.dt.tz is None:
            try:
                ts = ts.dt.tz_localize(tz, ambiguous=False, nonexistent="shift_forward")
            except Exception:
                ts = ts.dt.tz_localize("UTC").dt.tz_convert(tz)
        else:
            ts = ts.dt.tz_convert(tz)

        df.index = pd.DatetimeIndex(ts)
        df = df.drop(columns=[FCR_FREQ_DATETIME_COL])

    if not pd.api.types.is_numeric_dtype(df[column]):
        df[column] = (
            df[column].astype(str)
            .str.replace(",", ".", regex=False)
        )

    df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_index()
    if df.index.has_duplicates:
        warnings.warn("Duplicate timestamps found; keeping first occurrence.", UserWarning)
        df = df[~df.index.duplicated(keep="first")]

    return df[[column]]


def get_fcr_prices(file_path: str) -> pd.Series:
    df = pd.read_excel(file_path)

    df["start_hour"] = df["PRODUCTNAME"].str.split("_").str[1].astype(int)
    df["datetime"] = pd.to_datetime(df["DATE_FROM"]) + pd.to_timedelta(df["start_hour"], unit="h")

    df = df.set_index("datetime").tz_localize(
        "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
    )

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
