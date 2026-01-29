from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd
import pyomo.environ as pyo

from ..domain.depot import Depot


def flexibility_commercialization(
    depot: Depot,
    prices_by_market: Dict[str, pd.Series],
    fee_eur_per_kwh_by_market: Optional[Dict[str, float]] = None,
    *,
    timestep_hours: float | None = None,
    virtual_arbitrage: bool = True,
    cycling_cost_eur_per_kwh: float = 0.0,
    market_activity_mask: Dict[str, pd.Series],
    committed_positions: Dict[str, pd.Series],
    flexibility_bounds: pd.DataFrame,
    allow_imbalance: bool = False,
    imbalance_prices_pos: pd.Series | None = None,
    imbalance_prices_neg: pd.Series | None = None,
    imbalance_volume_penalty_eur_per_kwh: float = 0.0,
) -> pyo.ConcreteModel:
    """
    Flex-band based multi-market commercialization model (Pyomo).

    This function intentionally assumes that all time series (prices, masks,
    committed positions, flexibility bounds) are already validated and aligned
    by the IO / workflow layer. The model therefore focuses purely on the
    mathematical formulation, with minimal guardrails.

    Modeling conventions
    --------------------
    - Decisions live on T = 0..N-1 (power + market positions), aligned to a
      decision time index of length N.
    - Energy states live on S = 0..N (N+1 points), aligned to flexibility bounds.
    - Net power is defined as:
        p_net[t] = p_ch[t] - p_dis[t]
      with p_ch, p_dis >= 0 to represent efficiencies and cycling costs.

    Market coupling
    ---------------
    - Market positions are optimized per market and summed to match p_net.
    - Gate closures are enforced by fixing closed positions to committed values.
    - If virtual_arbitrage=False, a MILP prevents simultaneous import/export
      within a timestep using a binary mode variable and tight Big-M bounds.

    Units
    -----
    - Power: kW
    - Energy: kWh
    - Prices/fees: EUR/kWh
    - Objective: EUR (over the horizon)
    """

    # ============================================================
    # 1) Time axis and derived constants (assumed aligned upstream)
    # ============================================================
    markets: Iterable[str] = list(prices_by_market.keys())
    ref = prices_by_market[next(iter(markets))].sort_index()
    time_index = ref.index
    N = len(time_index)

    if timestep_hours is None:
        timestep_hours = (time_index[1] - time_index[0]).total_seconds() / 3600.0

    fee_eur_per_kwh_by_market = fee_eur_per_kwh_by_market or {}

    # Flexibility bands (states: N+1 rows)
    P_lower_ser = flexibility_bounds["Power_lower_kW"]
    P_upper_ser = flexibility_bounds["Power_upper_kW"]
    E_lower_ser = flexibility_bounds["Capacity_lower_kWh"]
    E_upper_ser = flexibility_bounds["Capacity_upper_kWh"]

    # Tight per-timestep import/export maxima (Big-M values in MILP constraints)
    P_ch_max_ser = P_upper_ser.clip(lower=0.0)         # max import (>=0)
    P_dis_max_ser = (-P_lower_ser).clip(lower=0.0)     # max export (>=0)

    # Global symmetric bound for market positions derived from fleet power bands
    p_market_max_value = float(max(P_upper_ser.max(), (-P_lower_ser).max()))

    # ============================================================
    # 2) Pyomo model structure (Sets / Params)
    # ============================================================
    m = pyo.ConcreteModel()

    m.T = pyo.RangeSet(0, N - 1)  # decisions
    m.S = pyo.RangeSet(0, N)      # states (N+1)

    m.MARKETS = pyo.Set(initialize=list(markets))
    m.dt = pyo.Param(initialize=float(timestep_hours))

    # --- Market prices & fees (EUR/kWh) ---
    m.price = pyo.Param(
        m.MARKETS,
        m.T,
        initialize=lambda mdl, mk, t: float(prices_by_market[mk].iloc[int(t)]),
    )
    m.fee = pyo.Param(
        m.MARKETS,
        initialize=lambda mdl, mk: float(fee_eur_per_kwh_by_market.get(mk, 0.0)),
        within=pyo.NonNegativeReals,
    )

    # --- Flexibility bands ---
    m.P_lower = pyo.Param(m.T, initialize=lambda mdl, t: float(P_lower_ser.iloc[int(t)]))
    m.P_upper = pyo.Param(m.T, initialize=lambda mdl, t: float(P_upper_ser.iloc[int(t)]))
    m.E_lower = pyo.Param(m.S, initialize=lambda mdl, s: float(E_lower_ser.iloc[int(s)]))
    m.E_upper = pyo.Param(m.S, initialize=lambda mdl, s: float(E_upper_ser.iloc[int(s)]))

    # Tight Big-M bounds per timestep (for MILP mode)
    m.P_ch_max_t = pyo.Param(m.T, initialize=lambda mdl, t: float(P_ch_max_ser.iloc[int(t)]))
    m.P_dis_max_t = pyo.Param(m.T, initialize=lambda mdl, t: float(P_dis_max_ser.iloc[int(t)]))
    # Big-M bounds:
    # Tight per-timestep upper bounds used in MILP mode to link continuous power/market
    # variables with a binary import/export mode. The Big-M technique "switches"
    # constraints on or off via a binary variable, preventing simultaneous import
    # and export within the same timestep. Here, M is chosen as the physically
    # feasible maximum power derived from the flexibility bands (not an arbitrary
    # large constant), which keeps the LP relaxation tight and numerically stable.

    # --- Depot parameters ---
    m.eta_c = pyo.Param(initialize=float(depot.eta_grid2depot))
    m.eta_d = pyo.Param(initialize=float(depot.eta_depot2grid))
    m.grid_limit = pyo.Param(initialize=float(depot.grid_connection_limit))

    # --- Cost coefficients ---
    m.c_deg = pyo.Param(initialize=float(cycling_cost_eur_per_kwh))
    m.c_imb_vol = pyo.Param(initialize=float(imbalance_volume_penalty_eur_per_kwh))

    # Global symmetric bound for market power
    m.p_market_max = pyo.Param(initialize=float(p_market_max_value))

    # Initial energy state (set by MPC workflow)
    m.E0 = pyo.Param(initialize=0.0, mutable=True)

    # Terminal target and weight (set by MPC workflow)
    m.Eterm = pyo.Param(initialize=0.0, mutable=True)
    m.w_term = pyo.Param(initialize=0.0, mutable=True, within=pyo.NonNegativeReals)

    # Committed market positions (fixed schedule after gate closure)
    m.p_market_committed = pyo.Param(
        m.MARKETS,
        m.T,
        initialize=lambda mdl, mk, t: float(committed_positions[mk].iloc[int(t)]),
    )

    # ============================================================
    # 3) Decision variables
    # ============================================================
    m.p_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)   # import magnitude [kW]
    m.p_dis = pyo.Var(m.T, within=pyo.NonNegativeReals)  # export magnitude [kW]
    m.p_net = pyo.Var(m.T, within=pyo.Reals)             # net power (+import / -export) [kW]
    m.E = pyo.Var(m.S, within=pyo.Reals)                 # energy state [kWh] (band model)

    # Market positions (signed): +import (buy), -export (sell)
    m.p_market = pyo.Var(
        m.MARKETS,
        m.T,
        bounds=lambda mdl, mk, t: (-mdl.p_market_max, mdl.p_market_max),
    )

    # Optional imbalance variables
    if allow_imbalance:
        m.p_imb_pos = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.p_imb_neg = pyo.Var(m.T, within=pyo.NonNegativeReals)

        m.price_imb_pos = pyo.Param(m.T, initialize=lambda mdl, t: float(imbalance_prices_pos.iloc[int(t)]))
        m.price_imb_neg = pyo.Param(m.T, initialize=lambda mdl, t: float(imbalance_prices_neg.iloc[int(t)]))

    # Terminal deviation (absolute) for soft terminal objective
    m.e_term_dev = pyo.Var(within=pyo.NonNegativeReals)

    # ============================================================
    # 4) Physical constraints (bands + efficiencies)
    # ============================================================
    # Initial condition
    m.energy_init = pyo.Constraint(expr=m.E[0] == m.E0)

    # Net power definition: p_net = p_ch - p_dis
    m.p_net_def = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_net[t] == mdl.p_ch[t] - mdl.p_dis[t])

    # Fleet power band constraints
    m.p_net_lb = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_net[t] >= mdl.P_lower[t])
    m.p_net_ub = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_net[t] <= mdl.P_upper[t])

    # Symmetric grid connection limit (additional depot constraint)
    m.grid_ub = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_net[t] <= mdl.grid_limit)
    m.grid_lb = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_net[t] >= -mdl.grid_limit)

    # Energy state transition with efficiencies:
    # E[t+1] = E[t] + eta_c * p_ch[t] * dt - (1/eta_d) * p_dis[t] * dt
    def energy_state_rule(mdl, t):
        return (
            mdl.E[t + 1]
            == mdl.E[t]
            + mdl.eta_c * mdl.p_ch[t] * mdl.dt
            - (1.0 / mdl.eta_d) * mdl.p_dis[t] * mdl.dt
        )

    m.energy_state = pyo.Constraint(m.T, rule=energy_state_rule)

    # Fleet energy band constraints (state index S)
    m.E_lb = pyo.Constraint(m.S, rule=lambda mdl, s: mdl.E[s] >= mdl.E_lower[s])
    m.E_ub = pyo.Constraint(m.S, rule=lambda mdl, s: mdl.E[s] <= mdl.E_upper[s])

    # Terminal deviation |E[N] - Eterm| <= e_term_dev
    last_s = N
    m.term_dev_pos = pyo.Constraint(expr=m.E[last_s] - m.Eterm <= m.e_term_dev)
    m.term_dev_neg = pyo.Constraint(expr=m.Eterm - m.E[last_s] <= m.e_term_dev)

    # Optional hard terminal constraint (disabled by default; MPC can activate)
    m.energy_term_hard = pyo.Constraint(expr=m.E[last_s] == m.Eterm)
    m.energy_term_hard.deactivate()

    # ============================================================
    # 5) Market coupling + gate-closure commitments
    # ============================================================
    # Gate closure: if a slot is closed, fix p_market to committed position
    def market_activity_rule(mdl, mk, t):
        allowed = bool(market_activity_mask[mk].iloc[int(t)])
        if allowed:
            return pyo.Constraint.Skip
        return mdl.p_market[mk, t] == mdl.p_market_committed[mk, t]

    m.market_activity = pyo.Constraint(m.MARKETS, m.T, rule=market_activity_rule)

    # Balance markets to physical net power
    def balance_rule(mdl, t):
        base = sum(mdl.p_market[mk, t] for mk in mdl.MARKETS)
        if allow_imbalance:
            return base + mdl.p_imb_pos[t] - mdl.p_imb_neg[t] == mdl.p_net[t]
        return base == mdl.p_net[t]

    m.market_balance = pyo.Constraint(m.T, rule=balance_rule)

    # ------------------------------------------------------------
    # Virtual arbitrage handling:
    # - LP mode: allow offsetting between markets, track absolute volume for fees
    # - MILP mode: prevent simultaneous import/export using binary mode and Big-M
    # ------------------------------------------------------------
    if virtual_arbitrage:
        # Absolute market volume for fee calculation (LP-safe linearization)
        m.p_market_abs = pyo.Var(m.MARKETS, m.T, within=pyo.NonNegativeReals)
        m.p_market_abs_pos = pyo.Constraint(
            m.MARKETS, m.T, rule=lambda mdl, mk, t: mdl.p_market_abs[mk, t] >= mdl.p_market[mk, t]
        )
        m.p_market_abs_neg = pyo.Constraint(
            m.MARKETS, m.T, rule=lambda mdl, mk, t: mdl.p_market_abs[mk, t] >= -mdl.p_market[mk, t]
        )
        m.p_market_vol = pyo.Expression(m.MARKETS, m.T, rule=lambda mdl, mk, t: mdl.p_market_abs[mk, t])

    else:
        # MILP: prevent simultaneous import/export within a timestep
        m.u_state = pyo.Var(m.T, within=pyo.Binary)  # 1=export mode, 0=import mode

        # Split signed market position into positive (buy/import) and negative (sell/export) parts
        m.p_market_pos = pyo.Var(m.MARKETS, m.T, within=pyo.NonNegativeReals)  # buy/import [kW]
        m.p_market_neg = pyo.Var(m.MARKETS, m.T, within=pyo.NonNegativeReals)  # sell/export [kW]
        m.p_market_vol = pyo.Expression(
            m.MARKETS, m.T, rule=lambda mdl, mk, t: mdl.p_market_pos[mk, t] + mdl.p_market_neg[mk, t]
        )

        # Define signed market position: p_market = buy - sell
        m.p_market_def = pyo.Constraint(
            m.MARKETS, m.T,
            rule=lambda mdl, mk, t: mdl.p_market[mk, t] == mdl.p_market_pos[mk, t] - mdl.p_market_neg[mk, t]
        )

        def total_export(mdl, t):
            return sum(mdl.p_market_neg[mk, t] for mk in mdl.MARKETS)

        def total_import(mdl, t):
            return sum(mdl.p_market_pos[mk, t] for mk in mdl.MARKETS)

        # Enforce either export or import using tight Big-M from bands
        m.export_mode_limit = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: total_export(mdl, t) <= mdl.P_dis_max_t[t] * mdl.u_state[t]
        )
        m.import_mode_limit = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: total_import(mdl, t) <= mdl.P_ch_max_t[t] * (1.0 - mdl.u_state[t])
        )

        # Also prevent simultaneous physical charging/discharging (efficiency loopholes)
        m.p_dis_mode_limit = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: mdl.p_dis[t] <= mdl.P_dis_max_t[t] * mdl.u_state[t]
        )
        m.p_ch_mode_limit = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: mdl.p_ch[t] <= mdl.P_ch_max_t[t] * (1.0 - mdl.u_state[t])
        )

    # ============================================================
    # 6) Objective: revenue - fees - degradation - imbalance penalty - terminal dev
    # ============================================================
    def obj_expr(mdl):
        # Market cashflow (EUR)
        energy_cashflow = sum(
            -mdl.price[mk, t] * mdl.p_market[mk, t] * mdl.dt
            for mk in mdl.MARKETS for t in mdl.T
        )

        # Transaction fees on absolute volume (EUR)
        fee_cost = sum(
            mdl.fee[mk] * mdl.p_market_vol[mk, t] * mdl.dt
            for mk in mdl.MARKETS for t in mdl.T
        )

        # Cycling / degradation cost on throughput (EUR)
        deg_cost = mdl.c_deg * sum(
            (mdl.p_ch[t] + mdl.p_dis[t]) * mdl.dt
            for t in mdl.T
        )

        # Optional imbalance cashflow and volume penalty
        imb_cash = 0.0
        imb_vol_pen = 0.0
        if allow_imbalance:
            imb_cash = sum(
                -mdl.price_imb_pos[t] * mdl.p_imb_pos[t] * mdl.dt
                + mdl.price_imb_neg[t] * mdl.p_imb_neg[t] * mdl.dt
                for t in mdl.T
            )
            imb_vol_pen = mdl.c_imb_vol * sum(
                (mdl.p_imb_pos[t] + mdl.p_imb_neg[t]) * mdl.dt
                for t in mdl.T
            )

        # Soft terminal objective (weight set by MPC)
        term_penalty = mdl.w_term * mdl.e_term_dev

        return energy_cashflow + imb_cash - fee_cost - deg_cost - imb_vol_pen - term_penalty

    m.obj = pyo.Objective(rule=obj_expr, sense=pyo.maximize)

    return m


