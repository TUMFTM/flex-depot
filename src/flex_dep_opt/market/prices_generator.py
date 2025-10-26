# prices_generator.py
# Utility for generating example (dummy) day-ahead price data.
# Creates a simple 24-hour price profile with base and peak hours,
# saves it as a CSV file under /data, and returns the output path.
# Used for testing and demonstration purposes.

import pandas as pd
from pathlib import Path

def write_example_prices_csv(
    path: str = "data/example_prices.csv",
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
    idx = pd.date_range(start, periods=24*7, freq="h")

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