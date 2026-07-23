"""
Fast configuration tests for the illustrative-example scenarios S1-S4 (no MPC runs).

Asserts that the four TOMLs parse against the Settings model, enable exactly
the intended markets, differ ONLY in markets / forecast_source / name, share
identical band/penalty/terminal/fee parameters and simulation windows, and
that the S4 forecast CSVs exist and cover the simulation window plus the DA
horizon lookahead.
"""

from pathlib import Path

import pandas as pd
import pytest

from flex_dep_opt.config.settings import Settings
from flex_dep_opt.io.prices import read_prices_csv
from flex_dep_opt.io.time import LOCAL_TIMEZONE, local_config_timestamp_to_utc

REPO_ROOT = Path(__file__).parent.parent
SCENARIO_DIR = REPO_ROOT / "examples/illustrative_example"
SCENARIO_IDS = ("s1", "s2", "s3", "s4")

# scenario -> (dayahead, intraday, fcr) enabled flags
_EXPECTED_MARKETS = {
    "s1": (True, False, False),
    "s2": (True, True, False),
    "s3": (True, True, True),
    "s4": (True, True, True),
}


@pytest.fixture(scope="module")
def scenarios() -> dict[str, Settings]:
    return {sid: Settings.load(SCENARIO_DIR / f"settings_{sid}.toml") for sid in SCENARIO_IDS}


def test_market_flags_per_scenario(scenarios):
    for sid, (da, intraday, fcr) in _EXPECTED_MARKETS.items():
        s = scenarios[sid]
        assert s.optimization.markets.dayahead.enabled is da, sid
        assert s.optimization.markets.intraday.enabled is intraday, sid
        assert s.optimization.trading.fcr.enabled is fcr, sid
        assert s.optimization.trading.mode == "realistic", sid


def test_forecast_source_only_in_s4(scenarios):
    for sid in ("s1", "s2", "s3"):
        s = scenarios[sid]
        assert s.optimization.markets.dayahead.forecast_source is None, sid
        assert s.optimization.markets.intraday.forecast_source is None, sid

    s4 = scenarios["s4"]
    assert s4.optimization.markets.dayahead.forecast_source
    assert s4.optimization.markets.intraday.forecast_source
    assert "forecast" in s4.optimization.markets.dayahead.forecast_source
    assert "forecast" in s4.optimization.markets.intraday.forecast_source


def test_shared_parameters_identical(scenarios):
    """Everything except markets, forecast_source and simulation.name is identical."""

    def comparable(s: Settings) -> dict:
        d = s.model_dump(mode="json")
        d["simulation"].pop("name")
        d["optimization"]["markets"]["dayahead"].pop("forecast_source")
        d["optimization"]["markets"]["intraday"].pop("enabled")
        d["optimization"]["markets"]["intraday"].pop("forecast_source")
        d["optimization"]["trading"]["fcr"].pop("enabled")
        return d

    reference = comparable(scenarios["s1"])
    for sid in ("s2", "s3", "s4"):
        assert comparable(scenarios[sid]) == reference, f"{sid} differs from s1 beyond scenario knobs"


def test_identical_simulation_window(scenarios):
    windows = {(s.simulation.start, s.simulation.end) for s in scenarios.values()}
    assert len(windows) == 1


def test_forecast_csvs_exist_and_cover_window(scenarios, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    s4 = scenarios["s4"]

    start = local_config_timestamp_to_utc(s4.simulation.start, local_tz=LOCAL_TIMEZONE)
    end = local_config_timestamp_to_utc(s4.simulation.end, local_tz=LOCAL_TIMEZONE)
    lookahead_end = end + pd.Timedelta(hours=float(s4.optimization.mpc.da_horizon_hours))

    for src in (
        s4.optimization.markets.dayahead.forecast_source,
        s4.optimization.markets.intraday.forecast_source,
    ):
        assert Path(src).is_file(), src
        idx = read_prices_csv(src).index
        assert idx[0] <= start, f"{src} starts after the simulation window ({idx[0]} > {start})"
        assert idx[-1] >= lookahead_end, (
            f"{src} ends before window + DA lookahead ({idx[-1]} < {lookahead_end})"
        )
