import warnings

import pandas as pd

FREQUENCY_NOMINAL_HZ = 50.0
FREQUENCY_DEADBAND_HZ = 0.010
FREQUENCY_FULL_ACTIVATION_HZ = 0.200

FCR_FREQ_DATETIME_COL = "DATETIME"
FCR_FREQ_COL          = "FREQ_WORST_DEV_HZ"
_FREQ_MIN_COL         = "FREQ_MIN_HZ"
_FREQ_MAX_COL         = "FREQ_MAX_HZ"


def droop_signal(
    freq_hz: float,
    nominal_hz: float = FREQUENCY_NOMINAL_HZ,
    deadband_hz: float = FREQUENCY_DEADBAND_HZ,
    full_activation_hz: float = FREQUENCY_FULL_ACTIVATION_HZ,
) -> float:
    delta_f = float(freq_hz) - nominal_hz
    if abs(delta_f) < deadband_hz:
        return 0.0
    return max(-1.0, min(1.0, -delta_f / full_activation_hz))


def get_fcr_frequency_data(file_path: str, tz: str = "Europe/Berlin") -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if FCR_FREQ_COL not in df.columns:
        if _FREQ_MIN_COL in df.columns and _FREQ_MAX_COL in df.columns:
            dev_max = (df[_FREQ_MAX_COL] - FREQUENCY_NOMINAL_HZ).abs()
            dev_min = (df[_FREQ_MIN_COL] - FREQUENCY_NOMINAL_HZ).abs()
            df[FCR_FREQ_COL] = df[_FREQ_MAX_COL].where(dev_max >= dev_min, df[_FREQ_MIN_COL])
        else:
            freq_col = next((c for c in df.columns if "FREQ" in c.upper()), None)
            if freq_col:
                df = df.rename(columns={freq_col: FCR_FREQ_COL})
            else:
                raise ValueError(f"FCR frequency CSV missing required column: {FCR_FREQ_COL}")

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

    if df[FCR_FREQ_COL].dtype == object:
        df[FCR_FREQ_COL] = (
            df[FCR_FREQ_COL].astype(str)
            .str.replace(",", ".", regex=False)
        )

    df[FCR_FREQ_COL] = pd.to_numeric(df[FCR_FREQ_COL], errors="coerce")

    df = df.sort_index()
    if df.index.has_duplicates:
        warnings.warn("Duplicate timestamps found; keeping first occurrence.", UserWarning)
        df = df[~df.index.duplicated(keep="first")]

    return df[[FCR_FREQ_COL]]


def get_fcr_prices(file_path: str) -> pd.Series:
    df = pd.read_excel(file_path)

    df["start_hour"] = df["PRODUCTNAME"].str.split("_").str[1].astype(int)
    df["datetime"] = pd.to_datetime(df["DATE_FROM"]) + pd.to_timedelta(df["start_hour"], unit="h")

    df = df.set_index("datetime").tz_localize(
        "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
    )

    price_col = "GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"
    return (
        df[price_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .rename("fcr_price")
        .sort_index()
    )
