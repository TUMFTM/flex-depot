import sys
import pathlib
import pandas as pd

from flex_dep_opt.config.settings import Settings

def load_frequency_file(path: pathlib.Path) -> pd.DataFrame:
    df = None
    for encoding in ("utf-8-sig", "latin-1", "utf-8"):
        try:
            _df = pd.read_csv(path, sep=";", dtype=str, encoding=encoding)
            if len(_df.columns) > 1:
                df = _df
                break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError("Could not decode file with utf-8-sig, latin-1, or utf-8.")

    df.columns = [c.strip().upper() for c in df.columns]

    col_map = {}
    for col in df.columns:
        stripped = col.replace(" ", "_")
        if stripped == "DATE":
            col_map["date"] = col
        elif stripped == "TIME":
            col_map["time"] = col
        elif "FREQ" in stripped:
            col_map["freq"] = col

    missing = [k for k in ("date", "time", "freq") if k not in col_map]
    if missing:
        raise ValueError(
            f"Could not find columns for: {missing}.\n"
            f"Columns found: {list(df.columns)}"
        )

    datetime_str = df[col_map["date"]].str.strip() + " " + df[col_map["time"]].str.strip()
    df["DATETIME"] = pd.to_datetime(datetime_str, format="%d.%m.%Y %H:%M:%S")

    df["FREQUENCY_HZ"] = (
        df[col_map["freq"]]
        .str.strip()
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    df = df[["DATETIME", "FREQUENCY_HZ"]].set_index("DATETIME").sort_index()
    return df

_fcr = Settings.load().optimization.trading.fcr
_FREQ_NOMINAL_HZ = _fcr.frequency_nominal_hz
_FREQ_DEADBAND_HZ = _fcr.deadband_hz
_FREQ_FULL_ACTIVATION_HZ = _fcr.full_activation_hz

def droop_signal_vec(
    freq: pd.Series,
    nominal_hz: float = _FREQ_NOMINAL_HZ,
    deadband_hz: float = _FREQ_DEADBAND_HZ,
    full_activation_hz: float = _FREQ_FULL_ACTIVATION_HZ,
) -> pd.Series:
    delta_f = freq - nominal_hz
    droop = (-delta_f / full_activation_hz).clip(-1.0, 1.0)
    return droop.where(delta_f.abs() >= deadband_hz, 0.0)


def resample_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    resampled = df["FREQUENCY_HZ"].resample("15min").agg(
        FREQ_MEAN_HZ="mean",
        FREQ_MIN_HZ="min",
        FREQ_MAX_HZ="max",
        FREQ_STD_HZ="std",
        SAMPLE_COUNT="count",
    )
    dev_max = (resampled["FREQ_MAX_HZ"] - _FREQ_NOMINAL_HZ).abs()
    dev_min = (resampled["FREQ_MIN_HZ"] - _FREQ_NOMINAL_HZ).abs()
    resampled["FREQ_WORST_DEV_HZ"] = resampled["FREQ_MAX_HZ"].where(
        dev_max >= dev_min, resampled["FREQ_MIN_HZ"]
    )
    droop = droop_signal_vec(df["FREQUENCY_HZ"])
    resampled["FREQ_DROOP_MEAN"] = droop.resample("15min").mean()
    resampled["FREQ_DROOP_ABS_MEAN"] = droop.abs().resample("15min").mean()
    resampled = resampled.round(6)
    resampled.index.name = "DATETIME"
    return resampled.reset_index()


def save_output(df: pd.DataFrame, path: pathlib.Path) -> None:
    df.to_csv(path, index=False, sep=",")
    print(f"Saved {len(df):,} rows → {path}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = pathlib.Path(sys.argv[1]).expanduser().resolve()

    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_path = pathlib.Path(sys.argv[2]).expanduser().resolve()
    else:
        output_path = input_path.with_name(input_path.stem + "_15min.csv")

    print(f"Reading  : {input_path}")
    df_raw = load_frequency_file(input_path)
    print(f"Loaded   : {len(df_raw):,} samples  "
          f"({df_raw.index[0]} → {df_raw.index[-1]})")

    df_15min = resample_to_15min(df_raw)
    save_output(df_15min, output_path)

if __name__ == "__main__":
    main()