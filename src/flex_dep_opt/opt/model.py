from __future__ import annotations
from typing import Dict, Iterable
import pyomo.environ as pyo
import pandas as pd
from ..domain.vehicle import Vehicle


def fleet_commercialization(
    vehicle: Vehicle,
    prices_by_market: Dict[str, pd.Series],
    *,
    timestep_hours: float | None = None,
    virtual_arbitrage: bool = True,
    degradation_cost_eur_per_kwh: float = 0.0,
    market_activity_mask: Dict[str, pd.Series],
    committed_positions: Dict[str, pd.Series],
    mobility_bounds: pd.DataFrame,
) -> pyo.ConcreteModel:
    """
    Flex-band based fleet commercialization model.

    Core idea:
    - Physical feasibility is ensured by aggregated power and energy bands
    - No explicit vehicle SOC is modeled
    - Any trajectory inside the bands is disaggregation-feasible by construction
    """

    # ------------------------------------------------------------------
    # 1) Time axis and markets
    # ------------------------------------------------------------------
    markets: Iterable[str] = list(prices_by_market.keys())
    if not markets:
        raise ValueError("prices_by_market must contain at least one market")

    ref = prices_by_market[next(iter(markets))].sort_index()
    time_index = ref.index

    for mk, s in prices_by_market.items():
        if not s.sort_index().index.equals(time_index):
            raise ValueError(f"Price index mismatch for market {mk}")
        prices_by_market[mk] = s.sort_index()

    # ------------------------------------------------------------------
    # 2) Timestep
    # ------------------------------------------------------------------
    if timestep_hours is None:
        if len(time_index) < 2:
            raise ValueError("Cannot infer timestep from single timestamp")
        timestep_hours = (time_index[1] - time_index[0]).total_seconds() / 3600.0

    # ------------------------------------------------------------------
    # 3) Mobility flex bands (assumed CLEAN & ALIGNED)
    # ------------------------------------------------------------------
    P_lower = mobility_bounds["Power_lower_kW"]
    P_upper = mobility_bounds["Power_upper_kW"]
    E_lower = mobility_bounds["Capacity_lower_kWh"]
    E_upper = mobility_bounds["Capacity_upper_kWh"]

    # Time-dependent maxima used for MILP big-M constraints
    P_ch_max = P_upper.clip(lower=0.0)          # max import
    P_dis_max = (-P_lower).clip(lower=0.0)      # max export

    # Global market power limit
    p_market_max_value = float(
        max(P_upper.max(), (-P_lower).max())
    )

    # ------------------------------------------------------------------
    # 4) Pyomo model
    # ------------------------------------------------------------------
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, len(time_index) - 1)
    m.MARKETS = pyo.Set(initialize=list(markets))
    m.dt = pyo.Param(initialize=float(timestep_hours))

    # ------------------------------------------------------------------
    # 5) Parameters
    # ------------------------------------------------------------------
    m.price = pyo.Param(
        m.MARKETS, m.T,
        initialize=lambda mdl, mk, t: float(prices_by_market[mk].iloc[int(t)])
    )

    m.P_lower = pyo.Param(m.T, initialize=lambda mdl, t: float(P_lower.iloc[int(t)]))
    m.P_upper = pyo.Param(m.T, initialize=lambda mdl, t: float(P_upper.iloc[int(t)]))
    m.E_lower = pyo.Param(m.T, initialize=lambda mdl, t: float(E_lower.iloc[int(t)]))
    m.E_upper = pyo.Param(m.T, initialize=lambda mdl, t: float(E_upper.iloc[int(t)]))
    m.P_ch_max_t = pyo.Param(m.T, initialize=lambda mdl, t: float(P_ch_max.iloc[int(t)]))
    m.P_dis_max_t = pyo.Param(m.T, initialize=lambda mdl, t: float(P_dis_max.iloc[int(t)]))

    m.eta_c = pyo.Param(initialize=float(vehicle.eta_charge))
    m.eta_d = pyo.Param(initialize=float(vehicle.eta_discharge))
    m.c_deg = pyo.Param(initialize=float(degradation_cost_eur_per_kwh))
    m.p_market_max = pyo.Param(initialize=p_market_max_value)

    # Initial energy (set externally by MPC)
    m.E0 = pyo.Param(initialize=0.0, mutable=True)

    # ------------------------------------------------------------------
    # 6) Decision variables
    # ------------------------------------------------------------------
    m.p_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)  # charging power (import)
    m.p_dis = pyo.Var(m.T, within=pyo.NonNegativeReals)  # discharging power (export)
    m.p_net = pyo.Var(m.T, within=pyo.Reals)            # net power to grid
    m.E = pyo.Var(m.T, within=pyo.Reals)                # aggregated energy state

    m.p_market = pyo.Var(
        m.MARKETS, m.T,
        bounds=lambda mdl, mk, t: (-mdl.p_market_max, mdl.p_market_max),
    )

    m.p_market_committed = pyo.Param(
        m.MARKETS, m.T,
        initialize=lambda mdl, mk, t: float(committed_positions[mk].iloc[int(t)])
    )

    # ------------------------------------------------------------------
    # 7) Physical constraints
    # ------------------------------------------------------------------

    # Net power definition: export minus import
    m.p_net_def = pyo.Constraint(
        m.T, rule=lambda mdl, t: mdl.p_net[t] == mdl.p_dis[t] - mdl.p_ch[t]
    )

    # Power band constraints
    m.p_net_lb = pyo.Constraint(
        m.T, rule=lambda mdl, t: mdl.p_net[t] >= mdl.P_lower[t]
    )
    m.p_net_ub = pyo.Constraint(
        m.T, rule=lambda mdl, t: mdl.p_net[t] <= mdl.P_upper[t]
    )

    # Charge / discharge power limits derived from band
    m.ch_lim = pyo.Constraint(
        m.T, rule=lambda mdl, t: mdl.p_ch[t] <= mdl.P_ch_max_t[t]
    )
    m.dis_lim = pyo.Constraint(
        m.T, rule=lambda mdl, t: mdl.p_dis[t] <= mdl.P_dis_max_t[t]
    )

    # Energy state transition
    def energy_state_rule(mdl, t):
        if t == 0:
            return mdl.E[t] == mdl.E0 \
                   + mdl.eta_c * mdl.p_ch[t] * mdl.dt \
                   - (1.0 / mdl.eta_d) * mdl.p_dis[t] * mdl.dt
        return mdl.E[t] == mdl.E[t - 1] \
               + mdl.eta_c * mdl.p_ch[t] * mdl.dt \
               - (1.0 / mdl.eta_d) * mdl.p_dis[t] * mdl.dt

    m.energy_state = pyo.Constraint(m.T, rule=energy_state_rule)

    # Energy band constraints
    m.E_lb = pyo.Constraint(
        m.T, rule=lambda mdl, t: mdl.E[t] >= mdl.E_lower[t]
    )
    m.E_ub = pyo.Constraint(
        m.T, rule=lambda mdl, t: mdl.E[t] <= mdl.E_upper[t]
    )

    # ------------------------------------------------------------------
    # 8) Market balance
    # ------------------------------------------------------------------
    if virtual_arbitrage:
        # LP case: markets can offset internally
        m.market_balance = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: sum(mdl.p_market[mk, t] for mk in mdl.MARKETS) == mdl.p_net[t]
        )

        def market_activity_rule(mdl, mk, t):
            if market_activity_mask[mk].iloc[int(t)]:
                return pyo.Constraint.Skip
            return mdl.p_market[mk, t] == mdl.p_market_committed[mk, t]

        m.market_activity = pyo.Constraint(m.MARKETS, m.T, rule=market_activity_rule)

    else:
        # MILP case: no simultaneous import/export
        m.net_pos = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.net_neg = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.u_state = pyo.Var(m.T, within=pyo.Binary)

        m.net_balance = pyo.Constraint(
            m.T, rule=lambda mdl, t: mdl.p_net[t] == mdl.net_pos[t] - mdl.net_neg[t]
        )

        m.net_pos_limit = pyo.Constraint(
            m.T, rule=lambda mdl, t: mdl.net_pos[t] <= mdl.P_dis_max_t[t] * mdl.u_state[t]
        )
        m.net_neg_limit = pyo.Constraint(
            m.T, rule=lambda mdl, t: mdl.net_neg[t] <= mdl.P_ch_max_t[t] * (1.0 - mdl.u_state[t])
        )

        m.p_market_pos = pyo.Var(m.MARKETS, m.T, within=pyo.NonNegativeReals)
        m.p_market_neg = pyo.Var(m.MARKETS, m.T, within=pyo.NonNegativeReals)

        m.p_market_def = pyo.Constraint(
            m.MARKETS, m.T,
            rule=lambda mdl, mk, t: mdl.p_market[mk, t]
                                 == mdl.p_market_pos[mk, t] - mdl.p_market_neg[mk, t]
        )

        m.export_balance = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: sum(mdl.p_market_pos[mk, t] for mk in mdl.MARKETS) == mdl.net_pos[t]
        )
        m.import_balance = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: sum(mdl.p_market_neg[mk, t] for mk in mdl.MARKETS) == mdl.net_neg[t]
        )

        def market_activity_rule_fix(mdl, mk, t):
            if market_activity_mask[mk].iloc[int(t)]:
                return pyo.Constraint.Skip
            return mdl.p_market[mk, t] == mdl.p_market_committed[mk, t]

        m.market_activity_fix = pyo.Constraint(m.MARKETS, m.T, rule=market_activity_rule_fix)

    # ------------------------------------------------------------------
    # 9) Objective
    # ------------------------------------------------------------------
    def obj_expr(mdl):
        revenue = sum(
            mdl.price[mk, t] * mdl.p_market[mk, t] * mdl.dt
            for mk in mdl.MARKETS
            for t in mdl.T
        )
        degradation = mdl.c_deg * sum(
            (mdl.p_ch[t] + mdl.p_dis[t]) * mdl.dt
            for t in mdl.T
        )
        return revenue - degradation

    m.obj = pyo.Objective(rule=obj_expr, sense=pyo.maximize)

    return m
