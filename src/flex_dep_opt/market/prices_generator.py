# prices_generator.py
# Utility for generating example (dummy) day-ahead price data or preparation of real historic EPEX data.
# Creates a simple 24-hour price profile with base and peak hours,
# saves it as a CSV file under /data, and returns the output path.
# Used for testing and demonstration purposes.

import pandas as pd
import numpy as np
from pathlib import Path


############### DAY-AHEAD ##################
def write_example_prices_DA_csv(
    path: str = "data/example_prices_DA.csv",
    base_price_eur_per_kwh: float = 0.10,
    peak_addition_eur_per_kwh: float = 0.05,
    tz: str = "Europe/Berlin",
) -> str:
    """Generate a simple 24h dummy day-ahead price profile and write it to CSV.

    - Base price: flat base value (default 0.10 €/kWh)
    - Peak hours (8–11h, 17–20h): base + peak_addition (default +0.05 €/kWh)
    - Returns the absolute path of the generated file.
    """
    # Create a DatetimeIndex for 24 hours starting at today's 00:00
    start = pd.Timestamp.now(tz=tz).normalize()
    idx = pd.date_range(start, periods=24*4*7, freq="15min")

    # Simple price shape: base + peak addition at typical hours
    prices = []
    for t in idx:
        if 8 <= t.hour <= 11 or 17 <= t.hour <= 20:
            prices.append(base_price_eur_per_kwh + peak_addition_eur_per_kwh)
        else:
            prices.append(base_price_eur_per_kwh)

    df = pd.DataFrame({"time": idx, "price": prices})

    # Ensure data directory exists
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    return str(output_path.resolve())


def write_from_epex_DA_csv(
        src_path: str,                                          # €/MWh
        dst_path: str = "data/epex_prices_DA.csv",               # €/kWh
        *,
        tz: str = "Europe/Berlin",
        price_col: str = "Deutschland/Luxemburg [€/MWh] Originalauflösungen",
) -> str:
    """Read and prepare historic EPEX dayahead data
    """
    df = pd.read_csv(src_path, sep=";", decimal=",")

    # Create a DatetimeIndex
    ts = pd.to_datetime(df["Datum von"],dayfirst=True,format="%d.%m.%Y %H:%M",errors="coerce")

    # Timezone handling
    ts = ts.dt.tz_localize(tz,nonexistent="shift_forward",ambiguous="NaT")

    df["time"] = ts

    # Extract price data
    prices = df[price_col].astype(float) / 1000                                     # from €/MWh to €/kWh

    # Standardized output format
    cleaned = pd.DataFrame({
        "time": df["time"],
        "price": prices,
    })

    cleaned = cleaned.dropna()

    # Ensure data directory exists
    output_path = Path(dst_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(output_path, index=False)
    return str(output_path.resolve())


############### INTRADAY ##################
def write_example_prices_ID_csv(
    path: str = "data/example_prices_ID.csv",
    da_price_path: str = "data/epex_prices_DA.csv",
    tz: str = "Europe/Berlin",
    spread_mean: float = 0.0,          # durchschnittliche Abweichung vom DA (€/kWh)
    spread_std: float = 0.01,          # typische Intraday-Volatilität (€/kWh)
    spike_prob: float = 0.03,          # Wahrscheinlichkeit für einen Preisspike
    spike_magnitude: float = 0.05      # Größe eines Preisspikes (€/kWh)
) -> str:
    """
    Generate a synthetic intraday price series based on:
    - existing day-ahead price curve (15-min resolution)
    - random noise (Gaussian)
    - occasional price spikes for added realism

    output: CSV in unified format (time, price)
    """

    # Load DA prices to use as basis
    da = pd.read_csv(da_price_path, parse_dates=["time"])
    da = da.sort_values("time").reset_index(drop=True)

    # Generate base spread and volatility
    np.random.seed(42)
    noise = np.random.normal(loc=spread_mean, scale=spread_std, size=len(da))

    # Occasional spikes
    spikes = np.random.rand(len(da))
    spikes = np.where(spikes < spike_prob, spike_magnitude, 0.0)

    # Build intraday prices
    id_price = da["price"] + noise + spikes
    id_price = id_price.clip(lower=0.0)  # no negative prices here unless you want them

    df_id = pd.DataFrame({
        "time": da["time"],
        "price": id_price
    })

    # Save
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_id.to_csv(output_path, index=False)

    return str(output_path.resolve())


############### reBAP ##################
def write_from_rebap_csv(
        src_path: str,
        dst_dir: str = "data",
        *,
        tz: str = "Europe/Berlin",
) -> str:
    """
    Read and prepare historic reBAP data.

    The function inspects the source file name to determine whether it
    contains data for "unterdeckt" (under-supplied) or "überdeckt/ueberdeckt"
    (over-supplied), automatically selects the corresponding price column,
    and writes a standardized CSV.

    Expected input CSV structure (column names may have additional units):
    - A date column, e.g. 'Datum'          (01.10.2025)
    - A time column, e.g. 'von'            (00:00)
    - A price column containing either:
        * 'unterdeckt'
        * 'überdeckt' or 'ueberdeckt'
      in its header text, in EUR/MWh.

    Output:
    - A CSV in `dst_dir` with:
        * 'time'  (timezone-aware timestamp)
        * 'price' (in EUR/kWh)
      named:
        * reBAP_prices_pos.csv  for "unterdeckt"
        * reBAP_prices_neg.csv  for "überdeckt"/"ueberdeckt"

    Returns
    -------
    str
        Absolute path of the written output file.
    """
    src_path = Path(src_path)
    file_name_lower = src_path.name.lower()

    # Determine whether this is "unterdeckt" (positive) or "überdeckt" (negative)
    if "unterdeckt" in file_name_lower:
        mode = "unterdeckt"
        sign_label = "pos"
    elif "überdeckt" in file_name_lower or "ueberdeckt" in file_name_lower:
        mode = "ueberdeckt"
        sign_label = "neg"
    else:
        raise ValueError(
            f"Could not determine reBAP mode from file name '{src_path.name}'. "
            "Expected 'unterdeckt' or 'überdeckt/ueberdeckt' in the file name."
        )

    # Read the raw CSV (semicolon separated, German decimal comma)
    df = pd.read_csv(src_path, sep=";", decimal=",")

    # Try to detect the date and time columns
    # Assumes typical column names like 'Datum' and 'von'
    date_col_candidates = [c for c in df.columns if "datum" in c.lower()]
    time_col_candidates = [c for c in df.columns if c.lower() in ("von", "start", "time", "uhrzeit")]

    if not date_col_candidates:
        raise ValueError("Could not find a date column (e.g. 'Datum') in the input file.")
    if not time_col_candidates:
        raise ValueError("Could not find a time column (e.g. 'von') in the input file.")

    date_col = date_col_candidates[0]
    time_col = time_col_candidates[0]

    # Build a datetime column from date + time
    ts = pd.to_datetime(
        df[date_col].astype(str) + " " + df[time_col].astype(str),
        dayfirst=True,
        format="%d.%m.%Y %H:%M",
        errors="coerce",
    )

    # Apply timezone localization
    ts = ts.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    df["time"] = ts

    # Detect the price column based on the mode
    if mode == "unterdeckt":
        price_cols = [c for c in df.columns if "unterdeckt" in c.lower()]
    else:  # "ueberdeckt"
        price_cols = [c for c in df.columns if "überdeckt" in c.lower() or "ueberdeckt" in c.lower()]

    if not price_cols:
        raise ValueError(
            f"Could not find a price column for mode '{mode}' in the input file. "
            "Expected a column containing 'unterdeckt' or 'überdeckt/ueberdeckt'."
        )

    price_col = price_cols[0]

    # Convert EUR/MWh to EUR/kWh
    prices = df[price_col].astype(float) / 1000.0

    # Standardized output format
    cleaned = pd.DataFrame({
        "time": df["time"],
        "price": prices,
    }).dropna()

    # Build destination path based on sign label
    dst_dir_path = Path(dst_dir)
    dst_dir_path.mkdir(parents=True, exist_ok=True)

    output_path = dst_dir_path / f"reBAP_prices_{sign_label}.csv"

    cleaned.to_csv(output_path, index=False)
    return str(output_path.resolve())

