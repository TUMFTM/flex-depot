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
    market_activity_mask: Dict[str, pd.Series] | None = None,
    committed_positions: Dict[str, pd.Series] | None = None,
    enforce_terminal_soc: bool = True,  # aktuell nicht genutzt (war physischer SOC)
    mobility_bounds: pd.DataFrame | None = None,
) -> pyo.ConcreteModel:
    """
    Flexband-basiertes Aggregationsmodell (Paper-Logik):
    - State E[t] (kWh) darf negativ sein
    - Power-Band: Power_lower_kW <= p_net[t] <= Power_upper_kW
    - Energy-Band: Capacity_lower_kWh <= E[t] <= Capacity_upper_kWh
    - p_net[t] = p_dis[t] - p_ch[t]
    - Märkte (DA/ID) liefern p_market[mk,t], deren Summe = p_net[t]
    - MILP-Modus verhindert gleichzeitigen Import+Export (keine virtuelle Arbitrage)
    """

    # ----------------------------
    # 1) Validierung & Alignment
    # ----------------------------
    markets: Iterable[str] = list(prices_by_market.keys())
    if not markets:
        raise ValueError("prices_by_market must contain at least one market")

    first_market = next(iter(markets))
    ref = prices_by_market[first_market].sort_index()
    if not isinstance(ref.index, pd.DatetimeIndex):
        raise ValueError("price series must have a DatetimeIndex")

    # Preise auf gemeinsame Zeitachse prüfen
    for mk, s in prices_by_market.items():
        s_sorted = s.sort_index()
        if not s_sorted.index.equals(ref.index):
            raise ValueError(
                f"Timestamps for market {mk} do not match reference market {first_market}"
            )
        prices_by_market[mk] = s_sorted

    time_index = ref.index

    # ----------------------------
    # 2) Masken & committed prüfen
    # ----------------------------
    if market_activity_mask is None:
        market_activity_mask = {mk: pd.Series(True, index=time_index) for mk in markets}
    else:
        for mk in markets:
            mask = market_activity_mask.get(mk)
            if mask is None:
                market_activity_mask[mk] = pd.Series(True, index=time_index)
            else:
                market_activity_mask[mk] = mask.reindex(time_index, fill_value=True)

    if committed_positions is None:
        committed_positions = {mk: pd.Series(0.0, index=time_index) for mk in markets}
    else:
        for mk in markets:
            pos = committed_positions.get(mk)
            if pos is None:
                committed_positions[mk] = pd.Series(0.0, index=time_index)
            else:
                committed_positions[mk] = pos.reindex(time_index, fill_value=0.0)

    # ----------------------------
    # 3) Mobility-Bounds verpflichtend + align
    # ----------------------------
    if mobility_bounds is None:
        raise ValueError("fleet_commercialization requires mobility_bounds (flex bands).")

    mob = mobility_bounds.sort_index()
    if not isinstance(mob.index, pd.DatetimeIndex):
        raise ValueError("mobility_bounds must have a DatetimeIndex")

    if not mob.index.equals(time_index):
        mob = mob.reindex(time_index)
        if mob.isnull().any().any():
            missing = mob[mob.isnull().any(axis=1)].index[:5]
            raise ValueError(f"mobility_bounds missing data for timestamps: {list(missing)}")

    mobility_bounds = mob

    required_cols = [
        "Power_lower_kW", "Power_upper_kW",
        "Capacity_lower_kWh", "Capacity_upper_kWh",
    ]
    missing_cols = [c for c in required_cols if c not in mobility_bounds.columns]
    if missing_cols:
        raise ValueError(f"mobility_bounds missing columns: {missing_cols}")

    # ----------------------------
    # 4) Δt bestimmen
    # ----------------------------
    if timestep_hours is None:
        if len(ref.index) < 2:
            raise ValueError("Need at least two timestamps to infer timestep.")
        dt_seconds = (ref.index[1] - ref.index[0]).total_seconds()
        timestep_hours = dt_seconds / 3600.0

    # p_market_max einmalig bestimmen (Fix #1)
    p_market_max_value = float(
        max(
            mobility_bounds["Power_upper_kW"].max(),
            (-mobility_bounds["Power_lower_kW"]).max(),
        )
    )

    # ----------------------------
    # 5) Pyomo Model
    # ----------------------------
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, len(ref) - 1)
    m.MARKETS = pyo.Set(initialize=list(markets))
    m.dt = pyo.Param(initialize=float(timestep_hours))

    # Preise Param(mk,t)
    def price_init(mdl, mk, t):
        return float(prices_by_market[mk].iloc[int(t)])
    m.price = pyo.Param(m.MARKETS, m.T, initialize=price_init)

    # Effizienzen + Degradationskosten
    m.eta_c = pyo.Param(initialize=float(vehicle.eta_charge))
    m.eta_d = pyo.Param(initialize=float(vehicle.eta_discharge))
    m.c_deg = pyo.Param(initialize=float(degradation_cost_eur_per_kwh))

    # Marktgrenze aus Band (Fix #1)
    m.p_market_max = pyo.Param(initialize=float(p_market_max_value))

    # ----------------------------
    # 6) Flexband-Parameter (aus CSV)
    # ----------------------------
    P_lower_ser = mobility_bounds["Power_lower_kW"]
    P_upper_ser = mobility_bounds["Power_upper_kW"]
    C_lower_ser = mobility_bounds["Capacity_lower_kWh"]
    C_upper_ser = mobility_bounds["Capacity_upper_kWh"]

    # abgeleitete (zeitabhängige) Maxima für MILP Big-M / p_ch/p_dis-Limits
    P_ch_max_ser = P_upper_ser.clip(lower=0.0)      # max Import (>=0)
    P_dis_max_ser = (-P_lower_ser).clip(lower=0.0)  # max Export (>=0)

    def C_lower_init(mdl, t): return float(C_lower_ser.iloc[int(t)])
    def C_upper_init(mdl, t): return float(C_upper_ser.iloc[int(t)])
    def P_lower_init(mdl, t): return float(P_lower_ser.iloc[int(t)])
    def P_upper_init(mdl, t): return float(P_upper_ser.iloc[int(t)])
    def P_ch_max_init(mdl, t): return float(P_ch_max_ser.iloc[int(t)])
    def P_dis_max_init(mdl, t): return float(P_dis_max_ser.iloc[int(t)])

    m.E_lower = pyo.Param(m.T, initialize=C_lower_init)
    m.E_upper = pyo.Param(m.T, initialize=C_upper_init)
    m.P_lower = pyo.Param(m.T, initialize=P_lower_init)
    m.P_upper = pyo.Param(m.T, initialize=P_upper_init)
    m.P_ch_max_t = pyo.Param(m.T, initialize=P_ch_max_init)
    m.P_dis_max_t = pyo.Param(m.T, initialize=P_dis_max_init)

    # Startzustand (wird von außen im MPC gesetzt)
    m.E0 = pyo.Param(initialize=0.0, mutable=True)

    # ----------------------------
    # 7) Physik / State
    # ----------------------------
    m.p_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)  # Import-Magnitude [kW]
    m.p_dis = pyo.Var(m.T, within=pyo.NonNegativeReals) # Export-Magnitude [kW]
    m.p_net = pyo.Var(m.T, within=pyo.Reals)            # Nettoleistung [kW]
    m.E = pyo.Var(m.T, within=pyo.Reals)                # Energiezustand [kWh], darf negativ sein

    # Kopplung p_net = p_dis - p_ch
    m.p_net_def = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_net[t] == mdl.p_dis[t] - mdl.p_ch[t])

    # Band-Grenzen (Power)
    m.p_net_lb = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_net[t] >= mdl.P_lower[t])
    m.p_net_ub = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_net[t] <= mdl.P_upper[t])

    # optionale Stabilitätslimits (abgeleitet aus Band)
    m.ch_lim = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_ch[t] <= mdl.P_ch_max_t[t])
    m.dis_lim = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_dis[t] <= mdl.P_dis_max_t[t])

    # Energie-State (E)
    def energy_state_rule(mdl, t):
        if t == 0:
            return mdl.E[t] == mdl.E0 + mdl.eta_c * mdl.p_ch[t] * mdl.dt - (1.0 / mdl.eta_d) * mdl.p_dis[t] * mdl.dt
        return mdl.E[t] == mdl.E[t - 1] + mdl.eta_c * mdl.p_ch[t] * mdl.dt - (1.0 / mdl.eta_d) * mdl.p_dis[t] * mdl.dt
    m.energy_state = pyo.Constraint(m.T, rule=energy_state_rule)

    # Band-Grenzen (Energy)
    m.E_lb = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.E[t] >= mdl.E_lower[t])
    m.E_ub = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.E[t] <= mdl.E_upper[t])

    # Terminal-SOC war physisch; band-basiert später optional ergänzen
    # if enforce_terminal_soc: ...

    # ----------------------------
    # 8) Marktvariablen + committed
    # ----------------------------
    m.p_market = pyo.Var(
        m.MARKETS, m.T,
        bounds=lambda mdl, mk, t: (-mdl.p_market_max, mdl.p_market_max),
    )

    def committed_init(mdl, mk, t):
        return float(committed_positions[mk].iloc[int(t)])
    m.p_market_committed = pyo.Param(m.MARKETS, m.T, initialize=committed_init)

    # ----------------------------
    # 9) Balance + "no virtual arbitrage" (MILP)
    # ----------------------------
    if virtual_arbitrage:
        # LP: Märkte können intern gegeneinander laufen, aber Gesamt = p_net
        def market_balance_rule(mdl, t):
            return sum(mdl.p_market[mk, t] for mk in mdl.MARKETS) == mdl.p_net[t]
        m.market_balance = pyo.Constraint(m.T, rule=market_balance_rule)

        # Handelsmasken: geschlossene Slots = committed
        def market_activity_rule(mdl, mk, t):
            allowed = bool(market_activity_mask[mk].iloc[int(t)])
            if allowed:
                return pyo.Constraint.Skip
            return mdl.p_market[mk, t] == mdl.p_market_committed[mk, t]
        m.market_activity = pyo.Constraint(m.MARKETS, m.T, rule=market_activity_rule)

    else:
        # MILP: kein gleichzeitiger Import/Export (keine virtuelle Arbitrage)
        m.net_pos = pyo.Var(m.T, within=pyo.NonNegativeReals)  # Export >= 0
        m.net_neg = pyo.Var(m.T, within=pyo.NonNegativeReals)  # Import >= 0
        m.u_state = pyo.Var(m.T, within=pyo.Binary)            # 1=Export, 0=Import

        # p_net = net_pos - net_neg
        m.net_balance = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_net[t] == mdl.net_pos[t] - mdl.net_neg[t])

        # Big-M mit zeitabhängigen Band-Maxima (Fix #2)
        m.net_pos_limit = pyo.Constraint(
            m.T, rule=lambda mdl, t: mdl.net_pos[t] <= mdl.P_dis_max_t[t] * mdl.u_state[t]
        )
        m.net_neg_limit = pyo.Constraint(
            m.T, rule=lambda mdl, t: mdl.net_neg[t] <= mdl.P_ch_max_t[t] * (1.0 - mdl.u_state[t])
        )

        # Optional: auch p_ch/p_dis an u_state binden (konsistent + stärker)
        m.p_dis_state_lim = pyo.Constraint(
            m.T, rule=lambda mdl, t: mdl.p_dis[t] <= mdl.P_dis_max_t[t] * mdl.u_state[t]
        )
        m.p_ch_state_lim = pyo.Constraint(
            m.T, rule=lambda mdl, t: mdl.p_ch[t] <= mdl.P_ch_max_t[t] * (1.0 - mdl.u_state[t])
        )

        # Marktweise Aufteilung in pos/neg
        m.p_market_pos = pyo.Var(m.MARKETS, m.T, within=pyo.NonNegativeReals)
        m.p_market_neg = pyo.Var(m.MARKETS, m.T, within=pyo.NonNegativeReals)

        # Definition: p_market = pos - neg
        m.p_market_def = pyo.Constraint(
            m.MARKETS, m.T,
            rule=lambda mdl, mk, t: mdl.p_market[mk, t] == mdl.p_market_pos[mk, t] - mdl.p_market_neg[mk, t]
        )

        # Sum pos = net_pos, Sum neg = net_neg  (=> Sum p_market = p_net)
        m.export_balance = pyo.Constraint(
            m.T, rule=lambda mdl, t: sum(mdl.p_market_pos[mk, t] for mk in mdl.MARKETS) == mdl.net_pos[t]
        )
        m.import_balance = pyo.Constraint(
            m.T, rule=lambda mdl, t: sum(mdl.p_market_neg[mk, t] for mk in mdl.MARKETS) == mdl.net_neg[t]
        )

        # Handelsmasken: geschlossene Slots = committed
        def market_activity_rule_fix(mdl, mk, t):
            allowed = bool(market_activity_mask[mk].iloc[int(t)])
            if allowed:
                return pyo.Constraint.Skip
            return mdl.p_market[mk, t] == mdl.p_market_committed[mk, t]
        m.market_activity_fix = pyo.Constraint(m.MARKETS, m.T, rule=market_activity_rule_fix)

    # ----------------------------
    # 10) Objective
    # ----------------------------
    def obj_expr(mdl):
        # Marktwert
        revenue = sum(
            mdl.price[mk, t] * mdl.p_market[mk, t] * mdl.dt
            for mk in mdl.MARKETS
            for t in mdl.T
        )

        # Degradation: proportional zu throughput
        deg_cost = mdl.c_deg * sum(
            (mdl.p_ch[t] + mdl.p_dis[t]) * mdl.dt
            for t in mdl.T
        )

        return revenue - deg_cost

    m.obj = pyo.Objective(rule=obj_expr, sense=pyo.maximize)

    return m
