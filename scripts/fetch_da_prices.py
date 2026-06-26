"""
Fetch day-ahead prices from Energy-Charts (Fraunhofer ISE) into a DA CSV.

Usage:
    python fetch_da_prices.py 2026-01-01 2026-05-31 [out.csv] [--bzn DE-LU]
"""

import argparse
import json
import urllib.request

import pandas as pd

API = "https://api.energy-charts.info/price"

def fetch_da_prices(start: str, end: str, bzn: str = "DE-LU") -> pd.Series:
    url = f"{API}?bzn={bzn}&start={start}&end={end}"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)

    idx = pd.to_datetime(data["unix_seconds"], unit="s", utc=True)
    prices = (pd.Series(data["price"], index=idx, name="price") / 1000.0).round(5)
    prices.index.name = "time"
    return prices.dropna()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("start", help="YYYY-MM-DD inclusive")
    p.add_argument("end", help="YYYY-MM-DD inclusive")
    p.add_argument("out", nargs="?", default=None)
    p.add_argument("--bzn", default="DE-LU", help="bidding zone (default DE-LU)")
    args = p.parse_args()

    out = args.out or f"data/prices/prices_DA_{args.start}_{args.end}.csv"
    prices = fetch_da_prices(args.start, args.end, args.bzn)
    assert not prices.empty, "API returned no prices for the requested range"
    prices.to_csv(out)
    print(f"Wrote {out}: {len(prices)} rows, {prices.index.min()} -> {prices.index.max()}")


if __name__ == "__main__":
    main()
