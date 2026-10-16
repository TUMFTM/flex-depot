"""FCR price-based bid acceptance and the capacity-price forecast error."""

import math
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from flex_dep_opt.config.settings import FCRForecastErrorSettings, FCRSettings, Settings
from flex_dep_opt.io.inputs import load_inputs
from flex_dep_opt.workflows.mpc_workflow import _fcr_bid_decision, simulate

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_TOML = REPO_ROOT / "src/flex_dep_opt/config/settings_quickstart.toml"


def test_bid_is_accepted_when_the_clearing_price_reaches_it():
    assert _fcr_bid_decision(5.0, 0.0, 5.0) == (5.0, True)
    assert _fcr_bid_decision(5.0, 0.0, 24.0) == (5.0, True)
    assert _fcr_bid_decision(5.0, 0.0, 4.99) == (5.0, False)


def test_markup_raises_the_bid_price():
    assert _fcr_bid_decision(5.0, 10.0, 14.0) == (15.0, False)
    assert _fcr_bid_decision(5.0, 10.0, 15.0) == (15.0, True)


def test_missing_or_negative_breakeven_bids_as_price_taker():
    assert _fcr_bid_decision(math.nan, 0.0, 0.0) == (0.0, True)
    assert _fcr_bid_decision(-3.0, 2.0, 1.0) == (2.0, False)


def test_settings_default_is_disabled():
    fcr = FCRSettings(prices_source="x.xlsx")
    assert fcr.bid_acceptance is False
    assert fcr.bid_markup_eur_per_mw == 0.0
    assert fcr.forecast_error.enabled is False


def test_settings_reject_invalid_values():
    with pytest.raises(ValidationError):
        FCRSettings(prices_source="x.xlsx", bid_markup_eur_per_mw=-1.0)
    with pytest.raises(ValidationError):
        FCRForecastErrorSettings(sigma_eur_per_mw=-1.0)
    with pytest.raises(ValidationError):
        FCRForecastErrorSettings(rho=1.0)


@pytest.fixture
def settings(monkeypatch) -> Settings:
    monkeypatch.chdir(REPO_ROOT)
    settings = Settings.from_toml(EXAMPLE_TOML)
    # crosses the 08:00 gate closure and reaches into the next day, so its first FCR slot can be bid
    settings.simulation.start = settings.simulation.start + pd.Timedelta(hours=7, minutes=30)
    settings.simulation.end = settings.simulation.start + pd.Timedelta(hours=16, minutes=45)
    settings.simulation.warmup_hours = 0.0
    settings.optimization.trading.fcr.bid_acceptance = True
    return settings


def test_forecast_error_perturbs_only_the_decision_prices(settings):
    fe = settings.optimization.trading.fcr.forecast_error
    fe.enabled, fe.sigma_eur_per_mw = True, 20.0

    inputs = load_inputs(settings)

    assert inputs.fcr_forecast_prices.index.equals(inputs.fcr_prices.index)
    assert not inputs.fcr_forecast_prices.equals(inputs.fcr_prices)
    assert (inputs.fcr_forecast_prices >= 0.0).all()


def test_without_forecast_error_the_decision_prices_are_the_clearing_prices(settings):
    assert load_inputs(settings).fcr_forecast_prices is None


def test_with_perfect_foresight_every_planned_bid_is_accepted(settings):
    commits = simulate(settings, load_inputs(settings), progress=False).fcr_commits

    bids = commits[commits["bid_kw"] > 0]
    assert not bids.empty
    assert bids["accepted"].all()
    assert (bids["x_fcr_kw"] == bids["bid_kw"]).all()


def test_rejected_bids_are_committed_at_zero(settings):
    settings.optimization.trading.fcr.bid_markup_eur_per_mw = 1e6

    commits = simulate(settings, load_inputs(settings), progress=False).fcr_commits

    bids = commits[commits["bid_kw"] > 0]
    assert not bids.empty
    assert not bids["accepted"].any()
    assert (commits["x_fcr_kw"] == 0.0).all()
    assert (commits["fcr_revenue_eur"] == 0.0).all()
