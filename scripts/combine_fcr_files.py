import glob
import os

import pandas as pd

SRC_GLOB = "data/prices/RESULT_OVERVIEW_CAPACITY_MARKET_FCR_2026-*.xlsx"
OUT_PATH = "data/prices/RESULT_OVERVIEW_CAPACITY_MARKET_FCR_2026-01-01_2026-05-31.xlsx"


def combine(src_glob: str = SRC_GLOB, out_path: str = OUT_PATH) -> str:
    files = sorted(f for f in glob.glob(src_glob))
    if not files:
        raise FileNotFoundError(f"No FCR files matched {src_glob!r}")

    frames = [pd.read_excel(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["DATE_FROM", "PRODUCTNAME"])
    combined.to_excel(out_path, index=False)
    return out_path


if __name__ == "__main__":
    out = combine()
    from flex_dep_opt.market.fcr import get_fcr_prices

    prices = get_fcr_prices(out)
    assert not prices.empty, "combined file produced no prices"
    print(f"Wrote {os.path.basename(out)}: {len(prices)} slot prices, "
          f"{prices.index.min()} -> {prices.index.max()}")
