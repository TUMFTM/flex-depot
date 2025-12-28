from __future__ import annotations

from typing import Dict, Iterable
import pandas as pd
import pyomo.environ as pyo

from ..domain.vehicle import Vehicle
from ..domain.site import Site


def fleet_commercialization(
    vehicle: Vehicle,
    site: Site,
    prices_by_market: Dict[str, pd.Series],
    fee_eur_per_kwh_by_market: Dict[str, float] | None = None,
    *,
    timestep_hours: float | None = None,
    virtual_arbitrage: bool = True,
    degradation_cost_eur_per_kwh: float = 0.0,
    market_activity_mask: Dict[str, pd.Series],
    committed_positions: Dict[str, pd.Series],
    mobility_bounds: pd.DataFrame,
    allow_imbalance: bool = False,
    imbalance_prices_pos: pd.Series | None = None,
    imbalance_prices_neg: pd.Series | None = None,
    imbalance_volume_penalty_eur_per_kwh: float = 0.0,
) -> pyo.ConcreteModel:
    """
    Flex-band based fleet commercialization model.

    Key modeling choices:
    - Aggregated energy state E[t] (kWh) can be negative (band model).
    - Aggregated net grid power p_net[t] is constrained by fleet bands (and optionally grid limit).
    - Physical charging/discharging variables (p_ch, p_dis) remain to model efficiencies and degradation.
    - Market positions are optimized per market and linked to physical net power.
    - MILP mode (virtual_arbitrage=False) prevents simultaneous import/export within a timestep.
    """

    # ----------------------------
    # 1) Basic time axis alignment
    # ----------------------------
    markets: Iterable[str] = list(prices_by_market.keys())
    if not markets:
        raise ValueError("prices_by_market must contain at least one market")

    ref = prices_by_market[next(iter(markets))].sort_index()
    time_index = ref.index
    if not isinstance(time_index, pd.DatetimeIndex):
        raise ValueError("Price series must have a DatetimeIndex")

    for mk, s in prices_by_market.items():
        s_sorted = s.sort_index()
        if not s_sorted.index.equals(time_index):
            raise ValueError(f"Price index mismatch for market {mk}")
        prices_by_market[mk] = s_sorted

    # ----------------------------
    # 2) Timestep
    # ----------------------------
    if timestep_hours is None:
        if len(time_index) < 2:
            raise ValueError("Need at least two timestamps to infer timestep")
        timestep_hours = (time_index[1] - time_index[0]).total_seconds() / 3600.0

    # ----------------------------
    # 3) Mobility flex bands (assumed clean + aligned by workflow/io)
    # ----------------------------
    required_cols = [
        "Power_lower_kW", "Power_upper_kW",
        "Capacity_lower_kWh", "Capacity_upper_kWh",
    ]
    missing_cols = [c for c in required_cols if c not in mobility_bounds.columns]
    if missing_cols:
        raise ValueError(f"mobility_bounds missing columns: {missing_cols}")
    if len(mobility_bounds) != len(time_index) + 1:
        raise ValueError(
            "mobility_bounds must have exactly N+1 rows (states) "
            "for N decision steps"
        )

    P_lower_ser = mobility_bounds["Power_lower_kW"]
    P_upper_ser = mobility_bounds["Power_upper_kW"]
    E_lower_ser = mobility_bounds["Capacity_lower_kWh"]
    E_upper_ser = mobility_bounds["Capacity_upper_kWh"]

    # Time-dependent maxima (used as tight Big-M values in MILP constraints)
    P_ch_max_ser = P_upper_ser.clip(lower=0.0)         # max import (>=0)
    P_dis_max_ser = (-P_lower_ser).clip(lower=0.0)     # max export (>=0)

    # Global market position bound derived from fleet power bands (safe and tight)
    p_market_max_value = float(max(P_upper_ser.max(), (-P_lower_ser).max()))

    # ----------------------------
    # 4) Pyomo model structure
    # ----------------------------
    m = pyo.ConcreteModel()

    N = len(time_index)  # number of decision steps
    m.T = pyo.RangeSet(0, N - 1)  # decisions (power, markets)
    m.S = pyo.RangeSet(0, N)  # states (energy) incl. terminal

    m.MARKETS = pyo.Set(initialize=list(markets))
    m.dt = pyo.Param(initialize=float(timestep_hours))

    # ----------------------------
    # 5) Parameters
    # ----------------------------
    # Marketprices & fees
    m.price = pyo.Param(m.MARKETS, m.T,initialize=lambda mdl, mk, t: float(prices_by_market[mk].iloc[int(t)]))
    m.fee = pyo.Param(m.MARKETS,initialize=lambda mdl, mk: float(fee_eur_per_kwh_by_market.get(mk, 0.0)),within=pyo.NonNegativeReals,)

    # Flex bands (power + energy)
    m.P_lower = pyo.Param(m.T, initialize=lambda mdl, t: float(P_lower_ser.iloc[int(t)]))
    m.P_upper = pyo.Param(m.T, initialize=lambda mdl, t: float(P_upper_ser.iloc[int(t)]))
    m.E_lower = pyo.Param(m.S, initialize=lambda mdl, s: float(E_lower_ser.iloc[int(s)]))
    m.E_upper = pyo.Param(m.S, initialize=lambda mdl, s: float(E_upper_ser.iloc[int(s)]))

    # Tight per-timestep import/export maxima (Big-M values)
    m.P_ch_max_t = pyo.Param(m.T, initialize=lambda mdl, t: float(P_ch_max_ser.iloc[int(t)]))
    m.P_dis_max_t = pyo.Param(m.T, initialize=lambda mdl, t: float(P_dis_max_ser.iloc[int(t)]))

    # Efficiency + degradation + imbalance
    m.eta_c = pyo.Param(initialize=float(vehicle.eta_charge))
    m.eta_d = pyo.Param(initialize=float(vehicle.eta_discharge))
    m.c_deg = pyo.Param(initialize=float(degradation_cost_eur_per_kwh))
    m.c_imb_vol = pyo.Param(initialize=float(imbalance_volume_penalty_eur_per_kwh))

    # Global market max (symmetric)
    m.p_market_max = pyo.Param(initialize=float(p_market_max_value))

    # Optional symmetric grid connection limit (symmetric import/export)
    m.grid_limit = pyo.Param(initialize=float(site.grid_connection_limit))

    # Initial energy state (set by MPC workflow)
    m.E0 = pyo.Param(initialize=0.0, mutable=True)

    # --- Terminal target (set by MPC workflow, optional) ---
    m.Eterm = pyo.Param(initialize=0.0, mutable=True)
    # Weight for soft terminal objective (0 => disabled)
    m.w_term = pyo.Param(initialize=0.0, mutable=True, within=pyo.NonNegativeReals)


    # Committed positions (set by MPC workflow; must be aligned)
    for mk in markets:
        if mk not in market_activity_mask:
            raise ValueError(f"market_activity_mask missing market: {mk}")
        if mk not in committed_positions:
            raise ValueError(f"committed_positions missing market: {mk}")
        if not committed_positions[mk].index.equals(time_index):
            raise ValueError(f"committed_positions[{mk}] must be aligned to model time index")
        if not market_activity_mask[mk].index.equals(time_index):
            raise ValueError(f"market_activity_mask[{mk}] must be aligned to model time index")

    m.p_market_committed = pyo.Param(
        m.MARKETS, m.T,
        initialize=lambda mdl, mk, t: float(committed_positions[mk].iloc[int(t)])
    )

    # ----------------------------
    # 6) Decision variables
    # ----------------------------
    m.p_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)   # Charging power (import magnitude) [kW]
    m.p_dis = pyo.Var(m.T, within=pyo.NonNegativeReals)  # Discharging power (export magnitude) [kW]
    m.p_net = pyo.Var(m.T, within=pyo.Reals)             # Net grid power (+export / -import) [kW]
    m.E = pyo.Var(m.S, within=pyo.Reals)                 # Aggregated energy state [kWh] (can be negative)

    # Initial condition
    m.energy_init = pyo.Constraint(expr=m.E[0] == m.E0)

    m.p_market = pyo.Var(m.MARKETS, m.T,bounds=lambda mdl, mk, t: (-mdl.p_market_max, mdl.p_market_max),)

    if allow_imbalance:
        m.p_imb_pos = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.p_imb_neg = pyo.Var(m.T, within=pyo.NonNegativeReals)

        # prices (EUR/kWh) aligned to time_index (decisions)
        if imbalance_prices_pos is None or imbalance_prices_neg is None:
            raise ValueError("allow_imbalance=True requires imbalance price series")

        if not imbalance_prices_pos.index.equals(time_index) or not imbalance_prices_neg.index.equals(time_index):
            raise ValueError("imbalance price series must align to model time index")

        m.price_imb_pos = pyo.Param(m.T, initialize=lambda mdl, t: float(imbalance_prices_pos.iloc[int(t)]))
        m.price_imb_neg = pyo.Param(m.T, initialize=lambda mdl, t: float(imbalance_prices_neg.iloc[int(t)]))

    # Absolute deviation from terminal target at end of horizon
    m.e_term_dev = pyo.Var(within=pyo.NonNegativeReals)

    # ----------------------------
    # 7) Physical constraints (bands + efficiencies)
    # ----------------------------

    # Define net power from charge/discharge decisions
    m.p_net_def = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.p_net[t] == mdl.p_dis[t] - mdl.p_ch[t]
    )  # Enforces p_net = export - import

    # Fleet power band (lower bound)
    m.p_net_lb = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.p_net[t] >= mdl.P_lower[t]
    )  # Enforces p_net not below fleet lower power band

    # Fleet power band (upper bound)
    m.p_net_ub = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.p_net[t] <= mdl.P_upper[t]
    )  # Enforces p_net not above fleet upper power band

    # Optional symmetric grid connection limit (additional site constraint)
    m.grid_ub = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.p_net[t] <= mdl.grid_limit
    )  # Enforces net export limited by site grid connection

    m.grid_lb = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.p_net[t] >= -mdl.grid_limit
    )  # Enforces net import limited by site grid connection

    # Energy state dynamics with efficiencies
    def energy_state_rule(mdl, t):
        # t in decision set 0..N-1 updates E[t+1]
        return mdl.E[t + 1] == mdl.E[t] \
            + mdl.eta_c * mdl.p_ch[t] * mdl.dt \
            - (1.0 / mdl.eta_d) * mdl.p_dis[t] * mdl.dt

    m.energy_state = pyo.Constraint(m.T, rule=energy_state_rule)

    # Terminal condition
    last_s = N  # last state index (since m.S = 0..N)
    m.term_dev_pos = pyo.Constraint(expr=m.E[last_s] - m.Eterm <= m.e_term_dev)
    m.term_dev_neg = pyo.Constraint(expr=m.Eterm - m.E[last_s] <= m.e_term_dev)
    # Optional HARD terminal constraint (disabled by default; MPC can activate it)
    m.energy_term_hard = pyo.Constraint(expr=m.E[last_s] == m.Eterm)
    m.energy_term_hard.deactivate()

    # Fleet energy band (lower bound)
    m.E_lb = pyo.Constraint(
        m.S,
        rule=lambda mdl, s: mdl.E[s] >= mdl.E_lower[s]
    )  # Enforces energy not below fleet lower energy band

    # Fleet energy band (upper bound)
    m.E_ub = pyo.Constraint(
        m.S,
        rule=lambda mdl, s: mdl.E[s] <= mdl.E_upper[s]
    )  # Enforces energy not above fleet upper energy band

    # ----------------------------
    # 8) Market coupling + gate-closure commitments
    # ----------------------------

    # Gate closure: if a slot is closed, fix p_market to committed position
    def market_activity_rule(mdl, mk, t):
        allowed = bool(market_activity_mask[mk].iloc[int(t)])
        if allowed:
            return pyo.Constraint.Skip
        return mdl.p_market[mk, t] == mdl.p_market_committed[mk, t]
    m.market_activity = pyo.Constraint(m.MARKETS, m.T, rule=market_activity_rule)  # Free when open, fixed when closed

    if virtual_arbitrage:
        # LP: allow offsetting between markets; only total must match physical net power
        def balance_rule(mdl, t):
            base = sum(mdl.p_market[mk, t] for mk in mdl.MARKETS)
            if allow_imbalance:
                return base + mdl.p_imb_pos[t] - mdl.p_imb_neg[t] == mdl.p_net[t]
            return base == mdl.p_net[t]
        m.market_balance = pyo.Constraint(m.T, rule=balance_rule)

        # Absolute market volume for fee calculation (LP-safe)
        m.p_market_abs = pyo.Var(m.MARKETS, m.T, within=pyo.NonNegativeReals)
        m.p_market_abs_pos = pyo.Constraint(m.MARKETS, m.T,rule=lambda mdl, mk, t: mdl.p_market_abs[mk, t] >= mdl.p_market[mk, t])
        m.p_market_abs_neg = pyo.Constraint(m.MARKETS, m.T,rule=lambda mdl, mk, t: mdl.p_market_abs[mk, t] >= -mdl.p_market[mk, t])
        m.p_market_vol = pyo.Expression(m.MARKETS, m.T, rule=lambda mdl, mk, t: mdl.p_market_abs[mk, t])

    else:
        # MILP: prevent simultaneous import and export within a timestep (no virtual arbitrage)
        m.u_state = pyo.Var(m.T, within=pyo.Binary)  # 1=export mode, 0=import mode

        # Split market power into positive (sell/export) and negative (buy/import) parts per market
        m.p_market_pos = pyo.Var(m.MARKETS, m.T, within=pyo.NonNegativeReals)  # Sell component [kW]
        m.p_market_neg = pyo.Var(m.MARKETS, m.T, within=pyo.NonNegativeReals)  # Buy component [kW]
        m.p_market_vol = pyo.Expression(m.MARKETS, m.T,rule=lambda mdl, mk, t: mdl.p_market_pos[mk, t] + mdl.p_market_neg[mk, t])

        # Define signed market position from pos/neg parts
        m.p_market_def = pyo.Constraint(
            m.MARKETS, m.T,
            rule=lambda mdl, mk, t: mdl.p_market[mk, t] == mdl.p_market_pos[mk, t] - mdl.p_market_neg[mk, t]
        )  # Enforces p_market = sell - buy

        # Total market export/import
        def total_export(mdl, t):
            return sum(mdl.p_market_pos[mk, t] for mk in mdl.MARKETS)

        def total_import(mdl, t):
            return sum(mdl.p_market_neg[mk, t] for mk in mdl.MARKETS)

        # Couple markets to physical net power
        def balance_rule(mdl, t):
            base = sum(mdl.p_market[mk, t] for mk in mdl.MARKETS)
            if allow_imbalance:
                return base + mdl.p_imb_pos[t] - mdl.p_imb_neg[t] == mdl.p_net[t]
            return base == mdl.p_net[t]
        m.market_balance = pyo.Constraint(m.T, rule=balance_rule)

        # Enforce "either export or import" using tight Big-M from bands
        m.export_mode_limit = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: total_export(mdl, t) <= mdl.P_dis_max_t[t] * mdl.u_state[t]
        )  # If u_state=0 => total export must be 0; if u_state=1 => export <= max export band

        m.import_mode_limit = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: total_import(mdl, t) <= mdl.P_ch_max_t[t] * (1.0 - mdl.u_state[t])
        )  # If u_state=1 => total import must be 0; if u_state=0 => import <= max import band

        # Also prevent simultaneous physical charging/discharging (avoids efficiency loopholes)
        m.p_dis_mode_limit = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: mdl.p_dis[t] <= mdl.P_dis_max_t[t] * mdl.u_state[t]
        )  # If import mode => discharging must be 0

        m.p_ch_mode_limit = pyo.Constraint(
            m.T,
            rule=lambda mdl, t: mdl.p_ch[t] <= mdl.P_ch_max_t[t] * (1.0 - mdl.u_state[t])
        )  # If export mode => charging must be 0

    # ----------------------------
    # 9) Objective: market revenue - degradation
    # ----------------------------
    def obj_expr(mdl):
        # Revenue term
        revenue = sum(
            mdl.price[mk, t] * mdl.p_market[mk, t] * mdl.dt
            for mk in mdl.MARKETS for t in mdl.T
        )

        # Market fee cost term
        fee_cost = sum(
            mdl.fee[mk] * mdl.p_market_vol[mk, t] * mdl.dt
            for mk in mdl.MARKETS for t in mdl.T
        )

        # Degradation cost term
        deg_cost = mdl.c_deg * sum(
            (mdl.p_ch[t] + mdl.p_dis[t]) * mdl.dt
            for t in mdl.T
        )

        # Imbalance cash & penalty term
        imb_cash = 0.0
        imb_vol_pen = 0.0
        if allow_imbalance:
            imb_cash = sum(
                mdl.price_imb_pos[t] * mdl.p_imb_pos[t] * mdl.dt
                - mdl.price_imb_neg[t] * mdl.p_imb_neg[t] * mdl.dt
                for t in mdl.T
            )
            # Penalize absolute imbalance volume (kWh) to prevent schedule-vs-physical arbitrage
            imb_vol_pen = mdl.c_imb_vol * sum(
                (mdl.p_imb_pos[t] + mdl.p_imb_neg[t]) * mdl.dt
                for t in mdl.T
            )

        return revenue + imb_cash - fee_cost - deg_cost - imb_vol_pen - mdl.w_term * mdl.e_term_dev

    m.obj = pyo.Objective(rule=obj_expr, sense=pyo.maximize)

    return m

