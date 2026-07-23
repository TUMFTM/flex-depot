import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from flex_dep_opt.domain.depot import Depot
from flex_dep_opt.io.prices import FCR_DROOP_COL
from flex_dep_opt.market.fcr import droop_signal, fcr_gate_closure_timestamp
from flex_dep_opt.opt.model import flexibility_commercialization
from flex_dep_opt.opt.solve import extract_objective_terms, solve_model
from flex_dep_opt.post.metrics import compute_cashflows_per_step, compute_kpis

_KW = dict(nominal_hz=50.0, deadband_hz=0.010, full_activation_hz=0.200)


def test_droop_sign_and_deadband():
    # delta_f:        0     +0.005   -0.1    +0.1    -0.02
    freq = pd.Series([50.0, 50.005, 49.9, 50.1, 49.98])
    d = droop_signal(freq, **_KW)

    # Ramp is linear from the deadband edge: magnitude = (|df| - 0.010) / (0.200 - 0.010).
    span = 0.200 - 0.010
    assert d.iloc[0] == 0.0  # at nominal -> 0
    assert d.iloc[1] == 0.0  # inside deadband -> 0
    assert abs(d.iloc[2] - (0.090 / span)) < 1e-9  # low freq  -> upward FCR -> droop > 0
    assert abs(d.iloc[3] + (0.090 / span)) < 1e-9  # high freq -> downward FCR -> droop < 0
    assert abs(d.iloc[4] - (0.010 / span)) < 1e-9  # just past deadband -> small activation

    # Continuous at the deadband edge: |df| = 0.010 gives ~0, not a jump to 0.05.
    edge = droop_signal(pd.Series([50.0 - 0.010]), **_KW)
    assert abs(edge.iloc[0]) < 1e-9


def test_droop_clipped_to_unit_interval():
    freq = pd.Series([49.0, 51.0])  # delta -1.0 / +1.0, far past full activation
    d = droop_signal(freq, **_KW)
    assert d.iloc[0] == 1.0
    assert d.iloc[1] == -1.0


def test_gate_closure_d_minus_1_default():
    slot = pd.Timestamp("2025-10-02 00:00", tz="Europe/Berlin")
    gc = fcr_gate_closure_timestamp(slot)  # default 08:00, closes previous day
    assert gc == pd.Timestamp("2025-10-01 08:00", tz="Europe/Berlin")


def test_gate_closure_same_day_and_custom_hour():
    slot = pd.Timestamp("2025-10-02 00:00", tz="Europe/Berlin")
    gc = fcr_gate_closure_timestamp(slot, hour="06:30", closes_previous_day=False)
    assert gc == pd.Timestamp("2025-10-02 06:30", tz="Europe/Berlin")


# =============================================================================
# Integration tests: FCR activation cashflow (Option A)
# =============================================================================

def _make_fcr_model(
    droop_vals: list[float],
    rebap_pos_price: float,
    rebap_neg_price: float,
    allow_imbalance: bool = False,
) -> pyo.ConcreteModel:
    """
    Minimal 4-step FCR model with one slot covering all steps.

    Sign convention (import-positive):
      droop_vals[t] > 0  → upward FCR   → p_droop[t] = -droop * x_fcr < 0 (export)
                                           BKV Überdeckung → earn rebap_neg_price
      droop_vals[t] < 0  → downward FCR → p_droop[t] = -droop * x_fcr > 0 (import)
                                           BKV Unterdeckung → pay rebap_pos_price

    FCR capacity is fixed at one bid-block (1000 kW) via gate_open=0 / committed=1000.
    """
    tz = "Europe/Berlin"
    slot_start = pd.Timestamp("2026-01-01 00:00", tz=tz)
    n = len(droop_vals)
    dt_h = 1.0
    idx = pd.date_range(slot_start, periods=n, freq="h", tz=tz)

    # Flat DA price; ID market active but zero price so spot is neutral
    prices = {
        "DA": pd.Series(0.05, index=idx),
        "ID": pd.Series(0.05, index=idx),
    }
    masks = {mk: pd.Series(True, index=idx) for mk in prices}
    committed = {mk: pd.Series(0.0, index=idx) for mk in prices}

    # Generous flex band: 10 MWh energy, ±1 MW power
    state_idx = pd.date_range(slot_start, periods=n + 1, freq="h", tz=tz)
    bounds = pd.DataFrame(
        {
            "Power_lower_kW": -1000.0,
            "Power_upper_kW": 1000.0,
            "Capacity_lower_kWh": 0.0,
            "Capacity_upper_kWh": 10000.0,
        },
        index=state_idx,
    )

    # FCR price series (one slot): 100 EUR/MW
    fcr_prices = pd.Series([100.0], index=[slot_start])

    # Frequency data with the configured droop values
    freq_df = pd.DataFrame({FCR_DROOP_COL: droop_vals}, index=idx)

    # reBAP price series
    rebap_pos = pd.Series(rebap_pos_price, index=idx)
    rebap_neg = pd.Series(rebap_neg_price, index=idx)

    imb_pos = pd.Series(rebap_pos_price, index=idx) if allow_imbalance else None
    imb_neg = pd.Series(rebap_neg_price, index=idx) if allow_imbalance else None

    depot = Depot(eta_grid2depot=1.0, eta_depot2grid=1.0, grid_connection_limit=5000.0)

    m = flexibility_commercialization(
        depot=depot,
        prices_by_market=prices,
        timestep_hours=dt_h,
        virtual_arbitrage=True,
        cycling_cost_eur_per_kwh=0.0,
        market_activity_mask=masks,
        committed_positions=committed,
        flexibility_bounds=bounds,
        allow_imbalance=allow_imbalance,
        imbalance_prices_pos=imb_pos,
        imbalance_prices_neg=imb_neg,
        imbalance_volume_penalty_eur_per_kwh=0.0,
        fcr_prices=fcr_prices,
        fcr_frequency_data=freq_df,
        fcr_product_hours=4.0,
        fcr_bid_block_kw=1000.0,
        fcr_rebap_prices_pos=rebap_pos,
        fcr_rebap_prices_neg=rebap_neg,
    )

    # Pin FCR capacity to exactly one block (1000 kW) so we can compute expected values analytically
    m.fcr_gate_open[0].set_value(0)
    m.x_fcr_committed[0].set_value(1000.0)

    m.E0.set_value(5000.0)  # start at mid-band
    return m


@pytest.mark.parametrize("solver_name", ["highs"])
def test_fcr_activation_cashflow_sign_and_magnitude(solver_name):
    """
    With x_fcr pinned at 1000 kW:
      step 0: droop=+0.5 → upward FCR → p_droop=-500 kW → revenue = rebap_neg * 500 kWh = 0.08*500=40 EUR
      step 2: droop=-0.3 → downward FCR → p_droop=+300 kW → cost = rebap_pos * 300 kWh = 0.10*300=30 EUR
      total obj_fcr_activation_cashflow = +40 - 30 = +10 EUR
    """
    droop_vals = [0.5, 0.0, -0.3, 0.0]
    rebap_pos = 0.10  # EUR/kWh: price for under-supply (downward FCR)
    rebap_neg = 0.08  # EUR/kWh: price for over-supply  (upward FCR)
    dt_h = 1.0

    m = _make_fcr_model(droop_vals, rebap_pos_price=rebap_pos, rebap_neg_price=rebap_neg)
    solve_model(m, solver_name=solver_name)
    terms = extract_objective_terms(m)

    expected = rebap_neg * 0.5 * 1000.0 * dt_h - rebap_pos * 0.3 * 1000.0 * dt_h  # 40 - 30 = 10
    assert terms["obj_fcr_activation_cashflow"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("solver_name", ["highs"])
def test_fcr_activation_cashflow_upward_only(solver_name):
    """All steps have upward FCR → activation cashflow is purely positive revenue."""
    droop_vals = [0.3, 0.5, 0.2, 0.0]
    rebap_neg = 0.09
    expected_rev = rebap_neg * (0.3 + 0.5 + 0.2) * 1000.0 * 1.0  # = 90 EUR

    m = _make_fcr_model(droop_vals, rebap_pos_price=0.10, rebap_neg_price=rebap_neg)
    solve_model(m, solver_name=solver_name)
    terms = extract_objective_terms(m)

    assert terms["obj_fcr_activation_cashflow"] == pytest.approx(expected_rev, abs=1e-6)
    assert terms["obj_fcr_activation_cashflow"] > 0.0


@pytest.mark.parametrize("solver_name", ["highs"])
def test_fcr_activation_cashflow_zero_without_droop(solver_name):
    """With zero droop signal, the activation cashflow term must be exactly zero."""
    droop_vals = [0.0, 0.0, 0.0, 0.0]

    m = _make_fcr_model(droop_vals, rebap_pos_price=0.10, rebap_neg_price=0.08)
    solve_model(m, solver_name=solver_name)
    terms = extract_objective_terms(m)

    assert terms["obj_fcr_activation_cashflow"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("solver_name", ["highs"])
def test_fcr_activation_cashflow_without_rebap_prices(solver_name):
    """When no reBAP prices are passed, obj_fcr_activation_cashflow must be 0."""
    tz = "Europe/Berlin"
    slot_start = pd.Timestamp("2026-01-01 00:00", tz=tz)
    idx = pd.date_range(slot_start, periods=4, freq="h", tz=tz)
    state_idx = pd.date_range(slot_start, periods=5, freq="h", tz=tz)
    prices = {"DA": pd.Series(0.05, index=idx)}
    bounds = pd.DataFrame(
        {"Power_lower_kW": -1000.0, "Power_upper_kW": 1000.0,
         "Capacity_lower_kWh": 0.0, "Capacity_upper_kWh": 10000.0},
        index=state_idx,
    )
    freq_df = pd.DataFrame({FCR_DROOP_COL: [0.5, 0.0, -0.3, 0.0]}, index=idx)
    depot = Depot(eta_grid2depot=1.0, eta_depot2grid=1.0, grid_connection_limit=5000.0)

    m = flexibility_commercialization(
        depot=depot,
        prices_by_market=prices,
        timestep_hours=1.0,
        virtual_arbitrage=True,
        cycling_cost_eur_per_kwh=0.0,
        market_activity_mask={"DA": pd.Series(True, index=idx)},
        committed_positions={"DA": pd.Series(0.0, index=idx)},
        flexibility_bounds=bounds,
        fcr_prices=pd.Series([100.0], index=[slot_start]),
        fcr_frequency_data=freq_df,
        fcr_bid_block_kw=1000.0,
        # fcr_rebap_prices_pos/neg deliberately omitted
    )
    m.fcr_gate_open[0].set_value(0)
    m.x_fcr_committed[0].set_value(1000.0)
    m.E0.set_value(5000.0)

    solve_model(m, solver_name=solver_name)
    terms = extract_objective_terms(m)
    assert terms["obj_fcr_activation_cashflow"] == pytest.approx(0.0, abs=1e-9)


def test_fcr_activation_cashflow_in_metrics():
    """
    compute_cashflows_per_step produces 'FCR Activation Cashflow [€/step]' and
    compute_kpis includes 'fcr_activation_cf_eur' when p_droop and IMB prices are present.
    """
    tz = "Europe/Berlin"
    idx = pd.date_range("2026-01-01 00:00", periods=4, freq="h", tz=tz)
    dt_h = 1.0
    x_fcr = 1000.0
    droop_vals = np.array([0.5, 0.0, -0.3, 0.0])

    # p_droop = -droop * x_fcr (import-positive)
    p_droop = -droop_vals * x_fcr

    dispatch = pd.DataFrame(
        {
            "p_net_kw": p_droop,
            "p_ch_kw": np.clip(p_droop, 0, None),
            "p_dis_kw": np.clip(-p_droop, 0, None),
            "p_da_kw": np.zeros(4),
            "p_droop_kw": p_droop,
            "x_fcr_kw": np.full(4, x_fcr),
            "E_kWh": np.ones(4) * 5000.0,
            "E_next_kWh": np.ones(4) * 5000.0,
        },
        index=idx,
    )

    rebap_pos = 0.10  # EUR/kWh (Unterdeckung price)
    rebap_neg = 0.08  # EUR/kWh (Überdeckung price)

    prices = {
        "DA": pd.Series(0.05, index=idx),
        "IMB_POS": pd.Series(rebap_pos, index=idx),
        "IMB_NEG": pd.Series(rebap_neg, index=idx),
    }

    cf_df = compute_cashflows_per_step(dispatch, prices, timestep_hours=dt_h)
    assert "FCR Activation Cashflow [€/step]" in cf_df.columns

    # step 0: upward FCR, p_droop=-500 → earn rebap_neg * 500 * 1h = 40 EUR
    # step 2: downward FCR, p_droop=+300 → pay rebap_pos * 300 * 1h = -30 EUR
    assert cf_df["FCR Activation Cashflow [€/step]"].iloc[0] == pytest.approx(+40.0, abs=1e-9)
    assert cf_df["FCR Activation Cashflow [€/step]"].iloc[1] == pytest.approx(0.0, abs=1e-9)
    assert cf_df["FCR Activation Cashflow [€/step]"].iloc[2] == pytest.approx(-30.0, abs=1e-9)

    # KPI extraction
    from flex_dep_opt.post.metrics import compute_market_aggregates

    energy_by_mk, _, _ = compute_market_aggregates(dispatch, prices, timestep_hours=dt_h)
    kpis = compute_kpis(cf_df, energy_by_mk, {})
    assert "fcr_activation_cf_eur" in kpis
    assert kpis["fcr_activation_cf_eur"] == pytest.approx(+10.0, abs=1e-9)  # 40 - 30


if __name__ == "__main__":
    test_droop_sign_and_deadband()
    test_droop_clipped_to_unit_interval()
    test_gate_closure_d_minus_1_default()
    test_gate_closure_same_day_and_custom_hour()
    print("ok")
