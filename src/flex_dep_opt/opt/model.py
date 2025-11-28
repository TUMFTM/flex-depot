from __future__ import annotations
from typing import Dict, Iterable
import pyomo.environ as pyo
import pandas as pd
from ..domain.vehicle import Vehicle



def vehicle_commercialization(
    vehicle: Vehicle,
    prices_by_market: Dict[str, pd.Series],  # z.B. {"DA": da_series, "ID": id_series}
    *,
    timestep_hours: float | None = None,
    virtual_arbitrage: bool = True,
    degradation_cost_eur_per_kwh: float = 0.0,
    market_activity_mask: Dict[str, pd.Series] | None = None,
    committed_positions: Dict[str, pd.Series] | None = None,
) -> pyo.ConcreteModel:
    """
    Generisches Modell für die Kommerzialisierung eines Fahrzeugs/Speichers
    in mehreren Strommärkten (DA, ID, ...).

    Parameter
    ---------
    vehicle : Vehicle
        Fahrzeug-/Speicher-Parameter (Kapazität, Leistungen, Wirkungsgrade, ...).
    prices_by_market : dict[str, pd.Series]
        Preiszeitreihen pro Markt, in EUR/kWh, alle mit identischer Zeitachse.
        Beispiel: {"DA": da_series, "ID": id_series}
    timestep_hours : float, optional
        Länge eines Zeitschritts [h]. Wenn None, wird aus der Zeitachse abgeleitet.
    virtual_arbitrage : bool, default True
        - True: Virtuelle Arbitrage erlaubt (rein finanzielle DA↔ID-Geschäfte möglich),
                solange |p_market| durch p_market_max begrenzt ist.
        - False: Keine virtuelle Arbitrage. Markt-Käufe/Verkäufe müssen durch
                 physische Batterieflüsse gedeckt sein, und pro Zeitschritt ist
                 nur netter Import oder netter Export möglich (kein gleichzeitiger
                 Import+Export).
    """

    # 1) Validierung & Alignment
    markets: Iterable[str] = list(prices_by_market.keys())
    if not markets:
        raise ValueError("prices_by_market must contain at least one market")

    first_market = next(iter(markets))
    ref = prices_by_market[first_market].sort_index()
    if not isinstance(ref.index, pd.DatetimeIndex):
        raise ValueError("price series must have a DatetimeIndex")

    for mkt, s in prices_by_market.items():
        s_sorted = s.sort_index()
        if not s_sorted.index.equals(ref.index):
            raise ValueError(
                f"Timestamps for market {mkt} do not match reference market {first_market}"
            )
        prices_by_market[mkt] = s_sorted

    time_index = ref.index

    # Activity-Mask validieren
    if market_activity_mask is None:
        # Fallback: überall True
        market_activity_mask = {mkt: pd.Series(True, index=time_index) for mkt in markets}
    else:
        # Sicherstellen, dass Index passt
        for mkt in markets:
            mask = market_activity_mask.get(mkt)
            if mask is None:
                # Wenn für diesen Markt keine Maske gegeben: alles True
                market_activity_mask[mkt] = pd.Series(True, index=time_index)
            else:
                mask = mask.reindex(time_index, fill_value=True)
                market_activity_mask[mkt] = mask

    # --- NEU: committed_positions nur validieren, noch nicht benutzen ---
    if committed_positions is None:
        # Fallback: überall 0
        committed_positions = {
            mkt: pd.Series(0.0, index=time_index) for mkt in markets
        }
    else:
        # Sicherstellen, dass Index passt
        for mkt in markets:
            pos = committed_positions.get(mkt)
            if pos is None:
                committed_positions[mkt] = pd.Series(0.0, index=time_index)
            else:
                committed_positions[mkt] = pos.reindex(time_index, fill_value=0.0)

    # Δt bestimmen
    if timestep_hours is None:
        if len(ref.index) < 2:
            raise ValueError("Need at least two timestamps to infer timestep.")
        dt_seconds = (ref.index[1] - ref.index[0]).total_seconds()
        timestep_hours = dt_seconds / 3600.0

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, len(ref) - 1)
    m.MARKETS = pyo.Set(initialize=list(markets))

    # Preise als Param(market, t)
    def price_init(model, mk, t):
        s = prices_by_market[mk]
        return float(s.iloc[int(t)])
    m.price = pyo.Param(m.MARKETS, m.T, initialize=price_init)

    m.dt = pyo.Param(initialize=float(timestep_hours))

    # Fahrzeugparameter
    m.cap = pyo.Param(initialize=float(vehicle.capacity_kwh))
    m.soc_min = pyo.Param(
        initialize=float(vehicle.soc_min) * float(vehicle.capacity_kwh)
    )
    m.soc_max = pyo.Param(
        initialize=float(vehicle.soc_max) * float(vehicle.capacity_kwh)
    )
    m.soc0 = pyo.Param(
        initialize=float(vehicle.soc0) * float(vehicle.capacity_kwh), mutable=True
    )
    m.p_ch_max = pyo.Param(initialize=float(vehicle.p_charge_max_kw))
    m.p_dis_max = pyo.Param(initialize=float(vehicle.p_discharge_max_kw))
    m.eta_c = pyo.Param(initialize=float(vehicle.eta_charge))
    m.eta_d = pyo.Param(initialize=float(vehicle.eta_discharge))

    # Degradationskosten [€/kWh Durchsatz]
    m.c_deg = pyo.Param(initialize=float(degradation_cost_eur_per_kwh))

    # Max. Marktleistung (symmetrisch), hier an physische Leistung gekoppelt
    P_market_max = max(
        float(vehicle.p_charge_max_kw), float(vehicle.p_discharge_max_kw)
    )
    m.p_market_max = pyo.Param(initialize=P_market_max)

    # ---------- Physik ----------
    m.p_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)    # [kW]
    m.p_dis = pyo.Var(m.T, within=pyo.NonNegativeReals)   # [kW]
    m.soc = pyo.Var(m.T, within=pyo.NonNegativeReals)     # [kWh]

    # SOC-Dynamik
    def soc_rule(mdl, t):
        if t == 0:
            return mdl.soc[t] == mdl.soc0 + mdl.eta_c * mdl.p_ch[t] * mdl.dt - (
                1.0 / mdl.eta_d
            ) * mdl.p_dis[t] * mdl.dt
        return mdl.soc[t] == mdl.soc[t - 1] + mdl.eta_c * mdl.p_ch[t] * mdl.dt - (
            1.0 / mdl.eta_d
        ) * mdl.p_dis[t] * mdl.dt
    m.soc_dyn = pyo.Constraint(m.T, rule=soc_rule)

    # Bounds
    m.soc_lb = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.soc[t] >= mdl.soc_min)
    m.soc_ub = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.soc[t] <= mdl.soc_max)
    m.ch_lim = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_ch[t] <= mdl.p_ch_max)
    m.dis_lim = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_dis[t] <= mdl.p_dis_max)

    # ---------- Markt-Variablen (für beide Modi) ----------
    m.p_market = pyo.Var(
        m.MARKETS,
        m.T,
        bounds=lambda mdl, mk, t: (-mdl.p_market_max, mdl.p_market_max),
    )


    # Virtual arbitrage active / inactive
    if virtual_arbitrage:
        # ===== LP-Fall: Virtuelle Arbitrage erlaubt =====
        def market_balance_rule(mdl, t):
            return sum(mdl.p_market[mk, t] for mk in mdl.MARKETS) == \
                   mdl.p_dis[t] - mdl.p_ch[t]
        m.market_balance = pyo.Constraint(m.T, rule=market_balance_rule)
        # --- Handelsmasken anwenden: LP-Variante ---
        def market_activity_rule(mdl, mk, t):
            idx = int(t)
            allowed = bool(market_activity_mask[mk].iloc[idx])
            if allowed:
                return pyo.Constraint.Skip
            return mdl.p_market[mk, t] == 0.0
        m.market_activity = pyo.Constraint(m.MARKETS, m.T, rule=market_activity_rule)

    else:
        # ===== MILP-Fall: Keine virtuelle Arbitrage =====
        #
        # Idee:
        # - net_pos[t]  >= 0: physischer Export
        # - net_neg[t]  >= 0: physischer Import
        # - p_dis[t] - p_ch[t] = net_pos[t] - net_neg[t]
        # - Binärvariable u[t]:
        #       u[t] = 1  → nur Export erlaubt (net_pos > 0, net_neg = 0)
        #       u[t] = 0  → nur Import erlaubt (net_neg > 0, net_pos = 0)
        #   → verhindert gleichzeitigen Import+Export → keine rein virtuellen Trades.
        # - Marktweise:
        #       p_market_pos[m,t] ≥ 0 (Verkäufe)
        #       p_market_neg[m,t] ≥ 0 (Käufe)
        #       p_market[m,t] = p_market_pos - p_market_neg
        #       Sum_m p_market_pos[m,t] = net_pos[t]
        #       Sum_m p_market_neg[m,t] = net_neg[t]

        m.net_pos = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.net_neg = pyo.Var(m.T, within=pyo.NonNegativeReals)

        # Binärvariable pro Zeitschritt: 1 = Export/Entladen, 0 = Import/Laden
        m.u_state = pyo.Var(m.T, within=pyo.Binary)

        # Energiegleichgewicht zwischen Batterie und Netz
        def net_balance_rule(mdl, t):
            return mdl.p_dis[t] - mdl.p_ch[t] == mdl.net_pos[t] - mdl.net_neg[t]
        m.net_balance = pyo.Constraint(m.T, rule=net_balance_rule)

        # Big-M-Beschränkungen: pro t entweder Export oder Import (oder beides 0)
        def net_pos_limit_rule(mdl, t):
            return mdl.net_pos[t] <= mdl.p_dis_max * mdl.u_state[t]
        m.net_pos_limit = pyo.Constraint(m.T, rule=net_pos_limit_rule)

        def net_neg_limit_rule(mdl, t):
            return mdl.net_neg[t] <= mdl.p_ch_max * (1.0 - mdl.u_state[t])
        m.net_neg_limit = pyo.Constraint(m.T, rule=net_neg_limit_rule)

        # Optional: nicht gleichzeitig laden und entladen
        def p_dis_limit_state_rule(mdl, t):
            return mdl.p_dis[t] <= mdl.p_dis_max * mdl.u_state[t]
        m.p_dis_state_lim = pyo.Constraint(m.T, rule=p_dis_limit_state_rule)

        def p_ch_limit_state_rule(mdl, t):
            return mdl.p_ch[t] <= mdl.p_ch_max * (1.0 - mdl.u_state[t])
        m.p_ch_state_lim = pyo.Constraint(m.T, rule=p_ch_limit_state_rule)

        # Marktweise Aufteilung
        m.p_market_pos = pyo.Var(m.MARKETS, m.T, within=pyo.NonNegativeReals)
        m.p_market_neg = pyo.Var(m.MARKETS, m.T, within=pyo.NonNegativeReals)

        # p_market = pos - neg
        def p_market_def_rule(mdl, mk, t):
            return mdl.p_market[mk, t] == mdl.p_market_pos[mk, t] - mdl.p_market_neg[mk, t]
        m.p_market_def = pyo.Constraint(m.MARKETS, m.T, rule=p_market_def_rule)

        # Summe der Verkäufe über Märkte = physischer Export
        def export_balance_rule(mdl, t):
            return sum(mdl.p_market_pos[mk, t] for mk in mdl.MARKETS) == mdl.net_pos[t]
        m.export_balance = pyo.Constraint(m.T, rule=export_balance_rule)

        # Summe der Käufe über Märkte = physischer Import
        def import_balance_rule(mdl, t):
            return sum(mdl.p_market_neg[mk, t] for mk in mdl.MARKETS) == mdl.net_neg[t]
        m.import_balance = pyo.Constraint(m.T, rule=import_balance_rule)

        # Handelsmasken anwenden
        def market_activity_rule_pos(mdl, mk, t):
            idx = int(t)
            allowed = bool(market_activity_mask[mk].iloc[idx])
            if allowed:
                return pyo.Constraint.Skip
            return mdl.p_market_pos[mk, t] == 0.0

        def market_activity_rule_neg(mdl, mk, t):
            idx = int(t)
            allowed = bool(market_activity_mask[mk].iloc[idx])
            if allowed:
                return pyo.Constraint.Skip
            return mdl.p_market_neg[mk, t] == 0.0
        m.market_activity_pos = pyo.Constraint(m.MARKETS, m.T, rule=market_activity_rule_pos)
        m.market_activity_neg = pyo.Constraint(m.MARKETS, m.T, rule=market_activity_rule_neg)



    # ---------- Zielfunktion ----------
    def obj_expr(mdl):
        # Erlöse / Kosten aus Märkten
        revenue = sum(
            mdl.price[mk, t] * mdl.p_market[mk, t] * mdl.dt
            for mk in mdl.MARKETS
            for t in mdl.T
        )

        # Degradationskosten: c_deg * (p_ch + p_dis) * dt
        deg_cost = mdl.c_deg * sum(
            (mdl.p_ch[t] + mdl.p_dis[t]) * mdl.dt
            for t in mdl.T
        )

        return revenue - deg_cost

    m.obj = pyo.Objective(expr=obj_expr(m), sense=pyo.maximize)

    if len(markets) > 0:
        mk0 = next(iter(markets))
        print("[DEBUG] committed_positions example for", mk0)
        print(committed_positions[mk0].head())

    return m