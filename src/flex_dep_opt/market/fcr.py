import pandas as pd

def get_fcr_prices(file_path: str) -> pd.Series:
    df = pd.read_excel(file_path)

    df["start_hour"] = df["PRODUCTNAME"].str.split("_").str[1].astype(int)
    df["datetime"] = pd.to_datetime(df["DATE_FROM"]) + pd.to_timedelta(df["start_hour"], unit="h")
    
    df = df.set_index("datetime").tz_localize(
        "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
    )

    price_col = "GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"
    fcr_series = (
        df[price_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .rename("fcr_price")
        .sort_index()
    )

    return fcr_series