"""Tests for grid usage fee accounting (energy + capacity price) and MiSpeL-A11 netting."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from flex_dep_opt.config.settings import NetTariffSettings, Settings
from flex_dep_opt.domain.depot import Depot
from flex_dep_opt.grid_fees import (
    monthly_net_billable,
    reference_grid_fees,
    settle_grid_fees,
    split_import_export,
)
from flex_dep_opt.opt.model import flexibility_commercialization
from flex_dep_opt.opt.solve import solve_model

_TZ = "Europe/Berlin"
_ORIGIN = pd.Timestamp("2026-02-01 00:00", tz=_TZ)


def _grid_model(
    prices: dict[str, list[float]],
    *,
    grid_fee: dict | None = None,
    p_lower=-1000.0,
    p_upper=1000.0,
    e_lower=0.0,
    e_upper=10_000.0,
    e_lower_last: float | None = None,
    e0=5000.0,
    dt_h=1.0,
    virtual_arbitrage=False,
) -> pyo.ConcreteModel:
    n = len(next(iter(prices.values())))
    freq = pd.tseries.frequencies.to_offset(pd.Timedelta(hours=dt_h))
    idx = pd.date_range(_ORIGIN, periods=n, freq=freq, tz=_TZ)
    state_idx = pd.date_range(_ORIGIN, periods=n + 1, freq=freq, tz=_TZ)

    p_lo = list(p_lower) if isinstance(p_lower, (list, np.ndarray)) else [float(p_lower)] * (n + 1)
    p_hi = list(p_upper) if isinstance(p_upper, (list, np.ndarray)) else [float(p_upper)] * (n + 1)
    e_lo = [float(e_lower)] * (n + 1)
    if e_lower_last is not None:
        e_lo[-1] = float(e_lower_last)

    bounds = pd.DataFrame(
        {
            "Power_lower_kW": p_lo,
            "Power_upper_kW": p_hi,
            "Capacity_lower_kWh": e_lo,
            "Capacity_upper_kWh": e_upper,
        },
        index=state_idx,
    )
    depot = Depot(eta_grid2depot=1.0, eta_depot2grid=1.0, grid_connection_limit=5000.0)
    prices_s = {mk: pd.Series(v, index=idx) for mk, v in prices.items()}
    masks = {mk: pd.Series(True, index=idx) for mk in prices}
    committed = {mk: pd.Series(0.0, index=idx) for mk in prices}

    m = flexibility_commercialization(
        depot=depot,
        prices_by_market=prices_s,
        timestep_hours=dt_h,
        virtual_arbitrage=virtual_arbitrage,
        cycling_cost_eur_per_kwh=0.0,
        market_activity_mask=masks,
        committed_positions=committed,
        flexibility_bounds=bounds,
        **(grid_fee or {}),
    )
    m.E0.set_value(e0)
    return m


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def test_config_disabled_by_default():
    nt = NetTariffSettings()
    assert nt.enabled is False
    assert nt.mispel_applicable is False


def test_config_requires_prices_and_reference_peak_when_enabled():
    with pytest.raises(ValueError):
        NetTariffSettings(enabled=True)
    with pytest.raises(ValueError):
        NetTariffSettings(enabled=True, energy_price_eur_per_kwh=0.05, capacity_price_eur_per_kw_year=100.0)


def test_config_rejects_negative_values():
    with pytest.raises(ValueError):
        NetTariffSettings(
            enabled=True,
            energy_price_eur_per_kwh=-0.01,
            capacity_price_eur_per_kw_year=100.0,
            reference_peak_kw=1000.0,
        )


def test_config_mispel_requires_both_flags():
    base = dict(
        enabled=True,
        energy_price_eur_per_kwh=0.05,
        capacity_price_eur_per_kw_year=100.0,
        reference_peak_kw=1000.0,
    )
    assert NetTariffSettings(**base, mispel_a11_enabled=True).mispel_applicable is False
    assert NetTariffSettings(**base, v2g_grid_fee_exemption_eligible=True).mispel_applicable is False
    assert (
        NetTariffSettings(
            **base, mispel_a11_enabled=True, v2g_grid_fee_exemption_eligible=True
        ).mispel_applicable
        is True
    )


# ---------------------------------------------------------------------------
# Pure billing math
# ---------------------------------------------------------------------------
def test_energy_price_on_gross_import_without_mispel():
    idx = pd.date_range(_ORIGIN, periods=4, freq="h", tz=_TZ)
    p_net = pd.Series([100.0, -40.0, 60.0, 0.0], index=idx)  # kW
    imp, exp = split_import_export(p_net, timestep_hours=1.0)
    monthly = monthly_net_billable(imp, exp, mispel_applicable=False)
    # Gross import billable, export ignored.
    assert monthly["billable_import_kwh"].sum() == pytest.approx(160.0)


def test_mispel_nets_import_and_export_per_month():
    idx = pd.date_range(_ORIGIN, periods=4, freq="h", tz=_TZ)
    p_net = pd.Series([100.0, -40.0, 60.0, 0.0], index=idx)
    imp, exp = split_import_export(p_net, timestep_hours=1.0)
    monthly = monthly_net_billable(imp, exp, mispel_applicable=True)
    # 160 import - 40 export = 120 billable.
    assert monthly["billable_import_kwh"].sum() == pytest.approx(120.0)


def test_mispel_export_above_import_never_negative():
    idx = pd.date_range(_ORIGIN, periods=3, freq="h", tz=_TZ)
    p_net = pd.Series([50.0, -200.0, 10.0], index=idx)  # export dominates
    imp, exp = split_import_export(p_net, timestep_hours=1.0)
    monthly = monthly_net_billable(imp, exp, mispel_applicable=True)
    assert monthly["billable_import_kwh"].sum() == pytest.approx(0.0)
    assert (monthly["billable_import_kwh"] >= 0.0).all()


def test_mispel_surplus_not_carried_across_months():
    # Feb: export-heavy (surplus). Mar: import-only. Surplus must not reduce March.
    feb = pd.date_range("2026-02-27 00:00", periods=2, freq="h", tz=_TZ)
    mar = pd.date_range("2026-03-01 00:00", periods=2, freq="h", tz=_TZ)
    idx = feb.append(mar)
    p_net = pd.Series([-100.0, -100.0, 50.0, 30.0], index=idx)
    imp, exp = split_import_export(p_net, timestep_hours=1.0)
    monthly = monthly_net_billable(imp, exp, mispel_applicable=True)
    billable = dict(monthly["billable_import_kwh"])
    assert billable[(2026, 2)] == pytest.approx(0.0)
    assert billable[(2026, 3)] == pytest.approx(80.0)


def test_reference_capacity_equals_lp_times_reference_peak():
    ref = reference_grid_fees(
        reference_energy_kwh=1000.0,
        reference_peak_kw=1500.0,
        energy_price_eur_per_kwh=0.05,
        capacity_price_eur_per_kw_year=100.0,
    )
    assert ref["ref_grid_fee_capacity_eur"] == pytest.approx(100.0 * 1500.0)
    # No efficiency correction: energy charge is AP * grid-side energy directly.
    assert ref["ref_grid_fee_energy_eur"] == pytest.approx(0.05 * 1000.0)
    assert ref["ref_grid_fee_total_eur"] == pytest.approx(0.05 * 1000.0 + 100.0 * 1500.0)


def test_settle_per_year_peak_with_initial():
    idx = pd.date_range(_ORIGIN, periods=4, freq="h", tz=_TZ)
    p_net = pd.Series([300.0, 700.0, 200.0, 0.0], index=idx)
    # Realized peak 700 < initial 1200 -> billed on initial.
    summary, _ = settle_grid_fees(
        p_net,
        timestep_hours=1.0,
        energy_price_eur_per_kwh=0.0,
        capacity_price_eur_per_kw_year=10.0,
        mispel_applicable=False,
        initial_annual_peak_kw=1200.0,
        initial_peak_year=2026,
    )
    assert summary["grid_peak_kw"] == pytest.approx(1200.0)
    assert summary["grid_fee_capacity_eur"] == pytest.approx(10.0 * 1200.0)


# ---------------------------------------------------------------------------
# Optimization model integration
# ---------------------------------------------------------------------------
def test_disabled_grid_fee_is_inert():
    m = _grid_model({"DA": [0.0, 0.0]})
    assert not hasattr(m, "p_grid_imp")
    assert float(pyo.value(m.obj_grid_fee_cost)) == 0.0


def test_grid_import_column_is_not_a_market_position():
    # The diagnostic billable-import column must not be mistaken for a market in
    # cashflow/energy aggregates or the dispatch plot.
    from flex_dep_opt.post.metrics import infer_market_position_columns

    df = pd.DataFrame(columns=["p_net_kw", "p_ch_kw", "p_dis_kw", "p_da_kw", "p_id_kw", "p_grid_imp_kw"])
    assert infer_market_position_columns(df) == ["p_da_kw", "p_id_kw"]


def test_grid_import_equals_positive_part_of_pnet():
    # Force a fixed import of 500 kW each step via the power band.
    m = _grid_model(
        {"DA": [0.0, 0.0, 0.0]},
        p_lower=500.0,
        p_upper=500.0,
        e_upper=1e9,
        grid_fee={
            "grid_fee_enabled": True,
            "grid_fee_energy_eur_per_kwh": 0.05,
            "grid_fee_capacity_eur_per_kw_year": 10.0,
        },
    )
    solve_model(m, solver_name="highs")
    imp = [float(pyo.value(m.p_grid_imp[t])) for t in m.T]
    net = [float(pyo.value(m.p_net[t])) for t in m.T]
    assert imp == pytest.approx([max(0.0, p) for p in net])
    assert float(pyo.value(m.p_peak_annual)) == pytest.approx(500.0)


def test_new_annual_peak_is_charged_but_below_previous_peak_is_free():
    gf = {
        "grid_fee_enabled": True,
        "grid_fee_energy_eur_per_kwh": 0.0,
        "grid_fee_capacity_eur_per_kw_year": 10.0,
    }
    # Import 500 kW each step. With previous peak 0 the 500 kW peak is charged.
    m_new = _grid_model({"DA": [0.0, 0.0]}, p_lower=500.0, p_upper=500.0, e_upper=1e9, grid_fee=gf)
    solve_model(m_new, solver_name="highs")
    assert float(pyo.value(m_new.obj_grid_fee_cost)) == pytest.approx(10.0 * 500.0)

    # Same import, but previous annual peak already 800 -> no incremental capacity cost.
    gf_prev = {**gf, "grid_fee_previous_annual_peak_kw": 800.0}
    m_below = _grid_model({"DA": [0.0, 0.0]}, p_lower=500.0, p_upper=500.0, e_upper=1e9, grid_fee=gf_prev)
    solve_model(m_below, solver_name="highs")
    assert float(pyo.value(m_below.obj_grid_fee_cost)) == pytest.approx(0.0)


def test_optimizer_performs_peak_shaving():
    # Need +4000 kWh over 8 flexible steps; a high capacity price should flatten import.
    prices = {"DA": [0.0] * 8}
    m = _grid_model(
        prices,
        p_lower=0.0,
        p_upper=2000.0,
        e_lower=5000.0,
        e_upper=9000.0,
        e_lower_last=9000.0,
        e0=5000.0,
        grid_fee={
            "grid_fee_enabled": True,
            "grid_fee_energy_eur_per_kwh": 0.0,
            "grid_fee_capacity_eur_per_kw_year": 1000.0,
        },
    )
    solve_model(m, solver_name="highs")
    peak = max(float(pyo.value(m.p_grid_imp[t])) for t in m.T)
    assert peak == pytest.approx(500.0, abs=1.0)  # 4000 kWh / 8 h


def test_peak_shaving_disabled_drops_capacity_term():
    gf = {
        "grid_fee_enabled": True,
        "grid_fee_energy_eur_per_kwh": 0.05,
        "grid_fee_capacity_eur_per_kw_year": 1000.0,
        "grid_fee_peak_shaving": False,
    }
    # Import 500 kW each step. With peak shaving off, the capacity price is not an incentive:
    # no annual-peak variable and the objective term is energy only.
    m = _grid_model({"DA": [0.0, 0.0]}, p_lower=500.0, p_upper=500.0, e_upper=1e9, grid_fee=gf)
    assert not hasattr(m, "p_peak_annual")
    solve_model(m, solver_name="highs")
    # Energy only: 0.05 * 500 kW * 1 h * 2 steps = 50; no capacity charge in the objective.
    assert float(pyo.value(m.obj_grid_fee_cost)) == pytest.approx(0.05 * 500.0 * 2)


def test_mispel_builds_monthly_billable_var():
    gf = {
        "grid_fee_enabled": True,
        "grid_fee_energy_eur_per_kwh": 0.05,
        "grid_fee_capacity_eur_per_kw_year": 0.0,
        "grid_fee_mispel_applicable": True,
    }
    m = _grid_model({"DA": [0.0, 0.0]}, grid_fee=gf)
    assert hasattr(m, "grid_billable_month")
    m_plain = _grid_model(
        {"DA": [0.0, 0.0]},
        grid_fee={**gf, "grid_fee_mispel_applicable": False},
    )
    assert not hasattr(m_plain, "grid_billable_month")


# ---------------------------------------------------------------------------
# End-to-end MPC + postprocessing
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
QUICKSTART_TOML = REPO_ROOT / "src" / "flex_dep_opt" / "config" / "settings_quickstart.toml"


def _short_net_tariff_settings(monkeypatch, *, initial_peak_kw: float) -> Settings:
    monkeypatch.chdir(REPO_ROOT)
    s = Settings.from_toml(QUICKSTART_TOML)
    s.simulation.end = s.simulation.start + pd.Timedelta(hours=6)
    s.simulation.warmup_hours = 12.0
    s.optimization.trading.fcr.enabled = False
    nt = s.optimization.net_tariff
    nt.enabled = True
    nt.energy_price_eur_per_kwh = 0.05
    nt.capacity_price_eur_per_kw_year = 100.0
    nt.reference_peak_kw = 1500.0
    nt.initial_annual_peak_kw = initial_peak_kw
    nt.peak_shaving = True
    return s


@pytest.mark.slow
def test_mpc_postprocessing_grid_fee_consistency(tmp_path, monkeypatch):
    from flex_dep_opt.io.results import SimulationResult
    from flex_dep_opt.workflows.mpc_workflow import run_mpc
    from flex_dep_opt.workflows.postprocessing_workflow import evaluate

    s = _short_net_tariff_settings(monkeypatch, initial_peak_kw=0.0)
    run_dir = run_mpc(s, run_dir=tmp_path / "run")
    result = SimulationResult.load(run_dir, settings=s)
    ev = evaluate(result)
    k = ev.kpis

    realized_peak = float(result.dispatch["p_net_kw"].clip(lower=0.0).max())
    assert k["grid_peak_kw"] == pytest.approx(realized_peak, rel=1e-6)
    assert k["ref_grid_fee_capacity_eur"] == pytest.approx(100.0 * 1500.0)
    assert k["grid_fee_total_eur"] == pytest.approx(k["grid_fee_energy_eur"] + k["grid_fee_capacity_eur"])
    # Grid fees are reported alongside, not inside, the market gross profit.
    assert k["profit_after_grid_fees_eur"] == pytest.approx(k["gross_profit_eur"] - k["grid_fee_total_eur"])
    assert k["grid_fee_annual_basis"] == "sub_annual_indicative"


@pytest.mark.slow
def test_mpc_high_initial_peak_dominates_billed_peak(tmp_path, monkeypatch):
    from flex_dep_opt.io.results import SimulationResult
    from flex_dep_opt.workflows.mpc_workflow import run_mpc
    from flex_dep_opt.workflows.postprocessing_workflow import evaluate

    s = _short_net_tariff_settings(monkeypatch, initial_peak_kw=9000.0)
    run_dir = run_mpc(s, run_dir=tmp_path / "run")
    result = SimulationResult.load(run_dir, settings=s)
    k = evaluate(result).kpis
    # A historical peak above every realized import is the billed peak (MPC carry-forward).
    assert k["grid_peak_kw"] == pytest.approx(9000.0)
    assert k["grid_fee_capacity_eur"] == pytest.approx(100.0 * 9000.0)
