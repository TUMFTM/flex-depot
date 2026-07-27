"""
Unit tests for fundamental LP / MILP optimisation model constraints.
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from flex_dep_opt.domain.depot import Depot
from flex_dep_opt.opt.model import flexibility_commercialization
from flex_dep_opt.opt.solve import solve_model

_TZ = "Europe/Berlin"
_ORIGIN = pd.Timestamp("2026-01-01 00:00", tz=_TZ)


def _build_model(
    prices_by_market: dict[str, list[float]],
    *,
    e0: float = 5000.0,
    p_lower: float | list[float] = -1000.0,
    p_upper: float | list[float] = 1000.0,
    e_lower: float = 0.0,
    e_upper: float = 10_000.0,
    eta_c: float = 1.0,
    eta_d: float = 1.0,
    dt_h: float = 1.0,
    virtual_arbitrage: bool = False,
) -> pyo.ConcreteModel:
    n = len(next(iter(prices_by_market.values())))
    freq = pd.tseries.frequencies.to_offset(pd.Timedelta(hours=dt_h))
    idx = pd.date_range(_ORIGIN, periods=n, freq=freq, tz=_TZ)
    state_idx = pd.date_range(_ORIGIN, periods=n + 1, freq=freq, tz=_TZ)

    p_lo = list(p_lower) if isinstance(p_lower, (list, np.ndarray)) else [float(p_lower)] * (n + 1)
    p_hi = list(p_upper) if isinstance(p_upper, (list, np.ndarray)) else [float(p_upper)] * (n + 1)

    bounds = pd.DataFrame(
        {
            "Power_lower_kW": p_lo,
            "Power_upper_kW": p_hi,
            "Capacity_lower_kWh": e_lower,
            "Capacity_upper_kWh": e_upper,
        },
        index=state_idx,
    )
    depot = Depot(eta_grid2depot=eta_c, eta_depot2grid=eta_d, grid_connection_limit=5000.0)
    prices_s = {mk: pd.Series(v, index=idx) for mk, v in prices_by_market.items()}
    masks = {mk: pd.Series(True, index=idx) for mk in prices_by_market}
    committed = {mk: pd.Series(0.0, index=idx) for mk in prices_by_market}

    m = flexibility_commercialization(
        depot=depot,
        prices_by_market=prices_s,
        timestep_hours=dt_h,
        virtual_arbitrage=virtual_arbitrage,
        cycling_cost_eur_per_kwh=0.0,
        market_activity_mask=masks,
        committed_positions=committed,
        flexibility_bounds=bounds,
    )
    m.E0.set_value(e0)
    return m


# =============================================================================
# 1. Big-M switching: no simultaneous import / export
# =============================================================================

@pytest.mark.parametrize("solver_name", ["highs"])
def test_milp_no_simultaneous_import_export(solver_name):
    """
    With virtual_arbitrage=False the Big-M binary u_state[t] enforces that the
    depot is in either import mode or export mode each step — never both.

    At step 1 the DA price is strongly negative (import profitable) while the ID
    price is strongly positive (export profitable).  Without the integer
    constraint the LP relaxation would net 1200 EUR at that step by importing
    1000 kW on DA and exporting 1000 kW on ID simultaneously, leaving p_net = 0.
    The MILP must choose one direction and earns at most 600 EUR.

    Two invariants are checked:
      (a) Physical: p_ch[t] · p_dis[t] ≈ 0 for all t.
      (b) Market:   total imports · total exports ≈ 0 for all t, i.e. no
                    cross-market roundtrip survives the integer constraint.
    """
    prices = {
        "DA": [0.05, -0.60, 0.05, 0.05],
        "ID": [0.05,  0.60, 0.05, 0.05],
    }
    m = _build_model(prices, virtual_arbitrage=False)
    solve_model(m, solver_name=solver_name)

    T = list(m.T)
    markets = list(m.MARKETS)

    p_ch  = np.array([pyo.value(m.p_ch[t])  for t in T])
    p_dis = np.array([pyo.value(m.p_dis[t]) for t in T])

    assert np.all(p_ch  >= -1e-6), "p_ch must be non-negative at all steps"
    assert np.all(p_dis >= -1e-6), "p_dis must be non-negative at all steps"

    prod_physical = p_ch * p_dis
    assert np.all(prod_physical < 1e-6), (
        f"Simultaneous physical charge and discharge: max p_ch·p_dis = {prod_physical.max():.3e}"
    )

    total_pos = np.array([
        sum(pyo.value(m.p_market_pos[mk, t]) for mk in markets) for t in T
    ])
    total_neg = np.array([
        sum(pyo.value(m.p_market_neg[mk, t]) for mk in markets) for t in T
    ])

    prod_market = total_pos * total_neg
    assert np.all(prod_market < 1e-6), (
        "Cross-market simultaneous import/export detected at step(s) "
        + str(np.where(prod_market >= 1e-6)[0].tolist())
        + f"; max total_pos·total_neg = {prod_market.max():.3e}"
    )


# =============================================================================
# 2. Energy state transition: η_c / η_d balance equation
# =============================================================================

@pytest.mark.parametrize("solver_name", ["highs"])
def test_energy_transition_satisfies_efficiency_balance(solver_name):
    """
    For every solved step the energy state must satisfy the exact recurrence

        E[t+1] = E[t] + η_c · p_ch[t] · dt  −  p_dis[t] / η_d · dt

    Tested with non-unit efficiencies (η_c = 0.90, η_d = 0.95) and a
    15-minute timestep (dt = 0.25 h) to cover the full arithmetic path.
    """
    eta_c, eta_d, dt_h = 0.90, 0.95, 0.25
    prices = {"DA": [0.05, 0.30, 0.05, 0.10]}

    m = _build_model(prices, eta_c=eta_c, eta_d=eta_d, dt_h=dt_h, e0=2000.0)
    solve_model(m, solver_name=solver_name)

    T = list(m.T)
    E     = np.array([pyo.value(m.E[t])     for t in T])
    E_nxt = np.array([pyo.value(m.E[t + 1]) for t in T])
    p_ch  = np.array([pyo.value(m.p_ch[t])  for t in T])
    p_dis = np.array([pyo.value(m.p_dis[t]) for t in T])

    expected = E + eta_c * p_ch * dt_h - p_dis / eta_d * dt_h
    np.testing.assert_allclose(
        E_nxt,
        expected,
        atol=1e-6,
        err_msg="E[t+1] = E[t] + η_c·p_ch·dt − p_dis/η_d·dt violated",
    )


# =============================================================================
# 3. Flex-band compliance: power and energy bounds always respected
# =============================================================================

@pytest.mark.parametrize("solver_name", ["highs"])
def test_solved_dispatch_respects_flex_band(solver_name):
    """
    After solving, every p_net[t] must lie within [P_lower[t], P_upper[t]] and
    every E[s] within [E_lower, E_upper], regardless of what the objective
    incentivises.  Tested with step-varying power limits so the constraint is
    binding at different magnitudes across the horizon.
    """
    # Six decision steps → seven state entries (last row needed for E[N]).
    p_lo = [-800.0, -400.0, -600.0, -300.0, -700.0, -800.0, -800.0]
    p_hi = [ 800.0,  400.0,  600.0,  300.0,  700.0,  800.0,  800.0]

    prices = {"DA": [0.05, 0.20, 0.08, 0.35, 0.05, 0.10]}
    m = _build_model(
        prices,
        p_lower=p_lo,
        p_upper=p_hi,
        e0=3000.0,
        e_lower=500.0,
        e_upper=9500.0,
    )
    solve_model(m, solver_name=solver_name)

    T = list(m.T)
    S = list(m.S)

    p_net  = np.array([pyo.value(m.p_net[t])    for t in T])
    p_lo_v = np.array([pyo.value(m.P_lower[t])  for t in T])
    p_hi_v = np.array([pyo.value(m.P_upper[t])  for t in T])
    E      = np.array([pyo.value(m.E[s])         for s in S])
    e_lo_v = np.array([pyo.value(m.E_lower[s])  for s in S])
    e_hi_v = np.array([pyo.value(m.E_upper[s])  for s in S])

    violations_p_lo = np.where(p_net < p_lo_v - 1e-6)[0]
    violations_p_hi = np.where(p_net > p_hi_v + 1e-6)[0]
    violations_e_lo = np.where(E < e_lo_v - 1e-6)[0]
    violations_e_hi = np.where(E > e_hi_v + 1e-6)[0]

    assert len(violations_p_lo) == 0, (
        f"p_net below P_lower at steps {violations_p_lo.tolist()}: "
        f"p_net={p_net[violations_p_lo]}, P_lower={p_lo_v[violations_p_lo]}"
    )
    assert len(violations_p_hi) == 0, (
        f"p_net above P_upper at steps {violations_p_hi.tolist()}: "
        f"p_net={p_net[violations_p_hi]}, P_upper={p_hi_v[violations_p_hi]}"
    )
    assert len(violations_e_lo) == 0, (
        f"E below E_lower at states {violations_e_lo.tolist()}: "
        f"E={E[violations_e_lo]}, E_lower={e_lo_v[violations_e_lo]}"
    )
    assert len(violations_e_hi) == 0, (
        f"E above E_upper at states {violations_e_hi.tolist()}: "
        f"E={E[violations_e_hi]}, E_upper={e_hi_v[violations_e_hi]}"
    )
