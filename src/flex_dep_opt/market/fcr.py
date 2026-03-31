
import pandas as pd
import yaml
from flex_dep_opt.io.flexibility_io import (
    read_flexibility_bounds_csv,
)

def load_settings(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def generate_fcr_availability_df():
    cfg = load_settings("src/flex_dep_opt/config/settings_example.yaml")
    opt_cfg = cfg["optimization"]
    flex_cfg = opt_cfg.get("flexibility", {})
    flexibility_bounds_full = read_flexibility_bounds_csv(flex_cfg["bounds_file"])

    input_df = pd.read_excel(
        "data/prices/RESULT_OVERVIEW_CAPACITY_MARKET_FCR_2025-01-01_2025-12-31.xlsx"
    )

    # NEGPOS_00_04 -> 0, NEGPOS_04_08 -> 4, ...
    input_df['start_hour'] = input_df['PRODUCTNAME'].str.split('_').str[1].astype(int)

    input_df['datetime'] = (
        pd.to_datetime(input_df['DATE_FROM'])
        + pd.to_timedelta(input_df['start_hour'], unit='h')
    )
    input_df = input_df.set_index('datetime').tz_localize(
        "Europe/Berlin", ambiguous='infer'
    )

    price_col = 'GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'
    input_df[price_col] = (
        input_df[price_col]
        .astype(str)
        .str.replace(',', '.', regex=False)
        .pipe(pd.to_numeric, errors='coerce')
    )

    fcr_prices = input_df[price_col].rename('fcr_price')

    symmetric_limit = flexibility_bounds_full.copy()

    if symmetric_limit.index.tz is None:
        symmetric_limit.index = symmetric_limit.index.tz_localize(
            "Europe/Berlin", ambiguous='infer'
        )
    else:
        symmetric_limit.index = symmetric_limit.index.tz_convert("Europe/Berlin")

    symmetric_limit['inst_symmetric_limit'] = (
        symmetric_limit[['Capacity_upper_kWh', 'Capacity_lower_kWh']]
        .abs()
        .min(axis=1)
    )

    inst_limit_utc = symmetric_limit['inst_symmetric_limit'].tz_convert("UTC")

    fcr_grouped_capacity = (
        inst_limit_utc
        .resample('4h', label='left', closed='left')
        .min()
        .rename('fcr_capacity_kWh')
    )

    fcr_grouped_capacity = fcr_grouped_capacity.where(
        fcr_grouped_capacity >= 1000, other=0.0
    )

    fcr_prices = input_df[price_col].rename('fcr_price')  
    fcr_prices = fcr_prices[~fcr_prices.index.duplicated(keep='first')]
    fcr_prices = fcr_prices.tz_convert("UTC")

    fcr_prices_df = fcr_prices.to_frame()
    fcr_prices_df.index = fcr_prices_df.index.floor('4h')
    fcr_prices_df = fcr_prices_df[~fcr_prices_df.index.duplicated(keep='first')]

    fcr_result = fcr_grouped_capacity.to_frame().join(fcr_prices_df, how='inner')

    fcr_result['fcr_revenue_eur'] = (
        fcr_result['fcr_capacity_kWh'] / 1000.0
    ) * fcr_result['fcr_price']

    fcr_result.index = fcr_result.index.tz_convert("Europe/Berlin")
    fcr_grouped_capacity.index = fcr_grouped_capacity.index.tz_convert("Europe/Berlin")

    return symmetric_limit, fcr_grouped_capacity, fcr_result