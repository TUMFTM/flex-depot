"""
Unit tests for the optional price-forecast interface (decision vs. settlement prices).

The MPC optimizes on DECISION prices (forecast where configured, realized
otherwise); settlement always uses the realized series. Without any
`forecast_source`, `build_forecast_prices_from_settings` must be identical to
`build_prices_from_settings` (perfect foresight — backward compatible).
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from flex_dep_opt.config.settings import Settings
from flex_dep_opt.io.prices import build_forecast_prices_from_settings, build_prices_from_settings
from flex_dep_opt.io.time import LOCAL_TIMEZONE, local_config_timestamp_to_utc
from flex_dep_opt.workflows.mpc_workflow import run_mpc

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_TOML = REPO_ROOT / "src/flex_dep_opt/config/settings_quickstart.toml"


def _write_prices_csv(path: Path, index: pd.DatetimeIndex, values) -> None:
    pd.DataFrame({"time": index, "price": values}).to_csv(path, index=False)


def _minimal_settings(da_source: Path, da_forecast_source: Path | None) -> SimpleNamespace:
    """Duck-typed stand-in for Settings covering everything the price builders read."""
    return SimpleNamespace(
        optimization=SimpleNamespace(
            markets=SimpleNamespace(
                dayahead=SimpleNamespace(
                    enabled=True,
                    source=str(da_source),
                    forecast_source=str(da_forecast_source) if da_forecast_source else None,
                ),
                intraday=SimpleNamespace(enabled=False, source="", forecast_source=None),
            ),
            imbalance=SimpleNamespace(enabled=False, source_pos="", source_neg=""),
        )
    )


def test_forecast_falls_back_to_realized(monkeypatch):
    """Without any forecast_source, decision prices equal realized prices for every market."""
    monkeypatch.chdir(REPO_ROOT)
    settings = Settings.load(EXAMPLE_TOML)

    realized = build_prices_from_settings(settings)
    decision = build_forecast_prices_from_settings(settings)

    assert set(decision.keys()) == set(realized.keys())
    for mk in realized:
        pd.testing.assert_series_equal(decision[mk], realized[mk])


def test_forecast_overrides_decision_prices(tmp_path):
    """A configured DA forecast replaces the decision prices; realized prices stay untouched."""
    idx = pd.date_range("2026-01-01 00:00", periods=8, freq="15min", tz="UTC")
    realized_csv = tmp_path / "prices_da_realized.csv"
    forecast_csv = tmp_path / "prices_da_forecast.csv"
    _write_prices_csv(realized_csv, idx, [10.0] * len(idx))
    _write_prices_csv(forecast_csv, idx, [20.0] * len(idx))

    settings = _minimal_settings(realized_csv, forecast_csv)

    decision = build_forecast_prices_from_settings(settings)
    realized = build_prices_from_settings(settings)

    assert (decision["DA"] == 20.0).all()
    assert (realized["DA"] == 10.0).all()
    assert decision["DA"].index.equals(realized["DA"].index)


def test_forecast_coverage_validation(tmp_path, monkeypatch):
    """A forecast CSV missing part of the simulation window fails fast with ValueError."""
    monkeypatch.chdir(REPO_ROOT)
    settings = Settings.load(EXAMPLE_TOML)

    realized = build_prices_from_settings(settings)["DA"]
    start = local_config_timestamp_to_utc(settings.simulation.start, local_tz=LOCAL_TIMEZONE)
    end = local_config_timestamp_to_utc(settings.simulation.end, local_tz=LOCAL_TIMEZONE)
    window = realized.loc[start:end]
    truncated = window.iloc[: len(window) // 2]
    forecast_csv = tmp_path / "prices_da_forecast_truncated.csv"
    _write_prices_csv(forecast_csv, truncated.index, truncated.to_numpy())

    settings.optimization.markets.dayahead.forecast_source = str(forecast_csv)

    with pytest.raises(ValueError, match="DA forecast prices .* do not cover"):
        run_mpc(settings, run_dir=tmp_path / "run")
