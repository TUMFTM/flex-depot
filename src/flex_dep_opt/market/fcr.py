import pandas as pd
import yaml


def load_settings(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_fcr_price_file(year: int = 2025) -> pd.Series:
    path = f"data/prices/RESULT_OVERVIEW_CAPACITY_MARKET_FCR_{year}-01-01_{year}-12-31.xlsx"
    df = pd.read_excel(path)

    df["start_hour"] = df["PRODUCTNAME"].str.split("_").str[1].astype(int)
    df["datetime"] = pd.to_datetime(df["DATE_FROM"]) + pd.to_timedelta(df["start_hour"], unit="h")
    df = df.set_index("datetime").tz_localize(
        "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
    )

    price_col = "GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"
    df[price_col] = (
        df[price_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    return df[price_col].rename("fcr_price")

def get_fcr_prices(years: list[int] = (2025)) -> pd.Series:
    fcr_prices = (
        pd.concat([_load_fcr_price_file(y) for y in years])
        .pipe(lambda s: s[~s.index.duplicated(keep="first")])
        .sort_index()
    )
    return fcr_prices