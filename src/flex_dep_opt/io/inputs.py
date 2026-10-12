from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from flex_dep_opt.config.settings import Settings
from flex_dep_opt.io.flexibility import (
    align_and_validate_flexibility_bounds,
    read_flexibility_bounds_csv,
)
from flex_dep_opt.io.prices import (
    build_forecast_prices_from_settings,
    build_prices_from_settings,
    get_fcr_frequency_data,
    get_fcr_prices,
)
from flex_dep_opt.io.time import LOCAL_TIMEZONE, local_config_timestamp_to_utc, validate_regular_index
from flex_dep_opt.market.forecast_error import draw_ar1_shape, perturb_prices

logger = logging.getLogger(__name__)


@dataclass
class SimulationInputs:
    """
    Time series a simulation runs on, in UTC, sliced to the simulation window including warm-up.

    `prices` are the realized prices used for settlement; `forecast_prices` are the prices
    the MPC decides on. Both are in €/kWh, keyed by market code ("DA", "ID").
    `index` holds the N decision timestamps, `state_index` the N+1 state timestamps
    that `flexibility_bounds` is aligned to. `fcr_prices` is in €/MW per product slot.
    """

    index: pd.DatetimeIndex
    state_index: pd.DatetimeIndex
    prices: dict[str, pd.Series]
    forecast_prices: dict[str, pd.Series]
    flexibility_bounds: pd.DataFrame
    imbalance_prices_pos: pd.Series | None
    imbalance_prices_neg: pd.Series | None
    fcr_prices: pd.Series
    fcr_frequency: pd.DataFrame | None


def simulation_window(settings: Settings) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Return `(start, report_start, end)` in UTC.

    Results are reported from `report_start`; `start` lies `warmup_hours` earlier.
    """
    sim_cfg = settings.simulation
    report_start = local_config_timestamp_to_utc(sim_cfg.start, local_tz=LOCAL_TIMEZONE)
    end = local_config_timestamp_to_utc(sim_cfg.end, local_tz=LOCAL_TIMEZONE)
    start = report_start - pd.Timedelta(hours=float(sim_cfg.warmup_hours))
    return start, report_start, end


def load_inputs(settings: Settings) -> SimulationInputs:
    """Read, slice and validate all input time series named in the settings."""
    sim_cfg = settings.simulation
    opt_cfg = settings.optimization
    step_hours = float(sim_cfg.timestep_hours)
    warmup_hours = float(sim_cfg.warmup_hours)

    flexibility_bounds_full = read_flexibility_bounds_csv(opt_cfg.flexibility.bounds_file)

    fcr_prices_full = (
        get_fcr_prices(opt_cfg.trading.fcr.prices_source)
        if opt_cfg.trading.fcr.enabled
        else pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="Europe/Berlin"))
    )
    fcr_freq_source = opt_cfg.trading.fcr.frequency_source
    fcr_frequency_data_full: pd.DataFrame | None = None
    if opt_cfg.trading.fcr.enabled and fcr_freq_source:
        fcr_frequency_data_full = get_fcr_frequency_data(fcr_freq_source)

    prices_by_market = build_prices_from_settings(settings)
    forecast_prices_by_market = build_forecast_prices_from_settings(settings)

    start, report_start, end = simulation_window(settings)

    for mk in prices_by_market:
        prices_by_market[mk] = prices_by_market[mk].loc[start:end]
        validate_regular_index(
            prices_by_market[mk].index,
            timestep_hours=step_hours,
            name=f"{mk} price index",
        )
        forecast_prices_by_market[mk] = forecast_prices_by_market[mk].loc[start:end]
        validate_regular_index(
            forecast_prices_by_market[mk].index,
            timestep_hours=step_hours,
            name=f"{mk} forecast price index",
        )

    # A forecast that does not cover the realized index would produce NaNs when sliced per window.
    mk_cfg = opt_cfg.markets
    for mk, detail in (("DA", mk_cfg.dayahead), ("ID", mk_cfg.intraday)):
        if not (detail.enabled and detail.forecast_source):
            continue
        realized_idx = prices_by_market[mk].index
        forecast_idx = forecast_prices_by_market[mk].index
        if not forecast_idx.equals(realized_idx):
            raise ValueError(
                f"{mk} forecast prices ({detail.forecast_source}) do not cover the full "
                f"simulation window incl. horizon lookahead: expected {len(realized_idx)} steps "
                f"[{realized_idx[0]} .. {realized_idx[-1]}], got {len(forecast_idx)} steps "
                f"[{forecast_idx[0] if len(forecast_idx) else 'empty'} .. "
                f"{forecast_idx[-1] if len(forecast_idx) else 'empty'}]."
            )

    # Only the decision prices are perturbed; settlement stays on the realized prices.
    # DA and ID need distinct seeds, otherwise both draw the same AR(1) shape.
    fe_by_market = {
        "DA": opt_cfg.markets.dayahead.forecast_error,
        "ID": opt_cfg.markets.intraday.forecast_error,
    }
    for mk, fe_cfg in fe_by_market.items():
        if fe_cfg.enabled and mk in forecast_prices_by_market:
            z_fe = draw_ar1_shape(forecast_prices_by_market[mk].index, rho=fe_cfg.rho, seed=fe_cfg.seed)
            forecast_prices_by_market[mk] = perturb_prices(
                forecast_prices_by_market[mk], z_fe, fe_cfg.sigma_eur_per_mwh
            )
            logger.info(
                f"{mk} forecast error active: sigma={fe_cfg.sigma_eur_per_mwh} EUR/MWh, "
                f"rho={fe_cfg.rho}, seed={fe_cfg.seed}"
            )

    full_index = prices_by_market[next(iter(prices_by_market))].index
    validate_regular_index(full_index, timestep_hours=step_hours, name="simulation decision index")

    if warmup_hours > 0.0 and full_index[0] >= report_start:
        raise ValueError(
            f"warmup_hours={warmup_hours} requested but no input data before the reporting start "
            f"{report_start.tz_convert(LOCAL_TIMEZONE)} — extend the input series back by at least "
            f"the warm-up length, or reduce warmup_hours."
        )

    dt = pd.Timedelta(hours=step_hours)
    full_state_index = full_index.append(pd.DatetimeIndex([full_index[-1] + dt]))

    flexibility_bounds_full = align_and_validate_flexibility_bounds(
        bounds=flexibility_bounds_full,
        time_index=full_state_index,
        expected_len=len(full_state_index),
    )

    imb_pos_full = prices_by_market.pop("IMB_POS", None)
    imb_neg_full = prices_by_market.pop("IMB_NEG", None)
    # Imbalance prices are ex-post, so they have no forecast.
    forecast_prices_by_market.pop("IMB_POS", None)
    forecast_prices_by_market.pop("IMB_NEG", None)

    fcr_prices_full = fcr_prices_full.loc[(fcr_prices_full.index >= start) & (fcr_prices_full.index <= end)]

    return SimulationInputs(
        index=full_index,
        state_index=full_state_index,
        prices=prices_by_market,
        forecast_prices=forecast_prices_by_market,
        flexibility_bounds=flexibility_bounds_full,
        imbalance_prices_pos=imb_pos_full,
        imbalance_prices_neg=imb_neg_full,
        fcr_prices=fcr_prices_full,
        fcr_frequency=fcr_frequency_data_full,
    )
