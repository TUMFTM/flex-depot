
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
    input_df = (
        input_df
        .set_index('datetime')
        .tz_localize("Europe/Berlin", ambiguous='infer', nonexistent='shift_forward')
    )

    price_col = 'GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'
    input_df[price_col] = (
        input_df[price_col]
        .astype(str)
        .str.replace(',', '.', regex=False)
        .pipe(pd.to_numeric, errors='coerce')
    )

    symmetric_limit = flexibility_bounds_full.copy()
    if symmetric_limit.index.tz is None:
        symmetric_limit.index = symmetric_limit.index.tz_localize(
            "Europe/Berlin", ambiguous='infer', nonexistent='shift_forward'
        )
    else:
        symmetric_limit.index = symmetric_limit.index.tz_convert("Europe/Berlin")

    symmetric_limit['inst_symmetric_limit'] = (
        symmetric_limit[['Capacity_upper_kWh', 'Capacity_lower_kWh']]
        .abs()
        .min(axis=1)
    )

    fcr_prices = (
        input_df[price_col]
        .rename('fcr_price')
        .pipe(lambda s: s[~s.index.duplicated(keep='first')])
    )

    def min_capacity_in_slot(slot_start, capacity_series):
        slot_end = slot_start + pd.Timedelta(hours=4)
        mask = (capacity_series.index >= slot_start) & (capacity_series.index < slot_end)
        values = capacity_series[mask]
        return values.min() if not values.empty else 0.0

    fcr_grouped_capacity = pd.Series(
        {ts: min_capacity_in_slot(ts, symmetric_limit['inst_symmetric_limit'])
         for ts in fcr_prices.index},
        name='fcr_capacity_kWh'
    )

    fcr_grouped_capacity = fcr_grouped_capacity.where(
        fcr_grouped_capacity >= 1000, other=0.0
    )

    fcr_result = fcr_grouped_capacity.to_frame().join(fcr_prices, how='inner')
    fcr_result = fcr_result.dropna(subset=['fcr_capacity_kWh', 'fcr_price'])
    fcr_result['fcr_revenue_eur'] = (
        fcr_result['fcr_capacity_kWh'] / 1000.0
    ) * fcr_result['fcr_price']

    common_index = fcr_result.index.intersection(symmetric_limit.index).intersection(fcr_grouped_capacity.index)
    fcr_result = fcr_result.loc[common_index]
    symmetric_limit = symmetric_limit.loc[common_index]
    fcr_grouped_capacity = fcr_grouped_capacity.loc[common_index]

    return symmetric_limit, fcr_grouped_capacity, fcr_result