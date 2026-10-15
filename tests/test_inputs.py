"""Loading the simulation inputs of the quick-start example (no MPC runs)."""

from pathlib import Path

import pandas as pd
import pytest

from flex_dep_opt.config.settings import Settings
from flex_dep_opt.io.inputs import load_inputs, simulation_window

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_TOML = REPO_ROOT / "src/flex_dep_opt/config/settings_quickstart.toml"


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings.from_toml(EXAMPLE_TOML)


@pytest.fixture(scope="module")
def inputs(settings):
    # data paths in the settings are relative to the repository root
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(REPO_ROOT)
        return load_inputs(settings)


def test_simulation_window_prepends_warmup(settings):
    start, report_start, end = simulation_window(settings)

    assert report_start - start == pd.Timedelta(hours=settings.simulation.warmup_hours)
    assert report_start == pd.Timestamp("2026-02-06 00:00", tz="Europe/Berlin")
    assert end == pd.Timestamp("2026-02-10 00:00", tz="Europe/Berlin")


def test_index_covers_the_window_including_warmup(settings, inputs):
    start, _, end = simulation_window(settings)
    step = pd.Timedelta(hours=settings.simulation.timestep_hours)

    assert inputs.index[0] == start
    assert inputs.index[-1] == end
    assert (inputs.index[1:] - inputs.index[:-1] == step).all()
    assert inputs.state_index[:-1].equals(inputs.index)
    assert inputs.state_index[-1] == end + step


def test_all_series_share_the_simulation_index(inputs):
    assert set(inputs.prices) == {"DA", "ID"}
    for market, prices in inputs.prices.items():
        assert prices.index.equals(inputs.index), market
        assert inputs.forecast_prices[market].index.equals(inputs.index), market
    assert inputs.flexibility_bounds.index.equals(inputs.state_index)
    assert inputs.imbalance_prices_pos.index.equals(inputs.index)
    assert inputs.imbalance_prices_neg.index.equals(inputs.index)


def test_without_forecast_the_decision_prices_are_the_realized_prices(inputs):
    for market, prices in inputs.prices.items():
        pd.testing.assert_series_equal(inputs.forecast_prices[market], prices)


def test_fcr_prices_lie_inside_the_window(settings, inputs):
    start, _, end = simulation_window(settings)

    assert not inputs.fcr_prices.empty
    assert inputs.fcr_prices.index.min() >= start
    assert inputs.fcr_prices.index.max() <= end


def test_missing_reference_energy_column_fails_before_the_simulation(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    settings = Settings.from_toml(EXAMPLE_TOML)
    settings.postprocessing.reference_driving_energy_costs.energy_column = "no_such_column"

    with pytest.raises(ValueError, match="no_such_column"):
        load_inputs(settings)
