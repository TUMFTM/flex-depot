from __future__ import annotations
import pyomo.environ as pyo
import pandas as pd
from ..domain.vehicle import Vehicle


def build_single_vehicle_model(vehicle: Vehicle, prices_eur_per_kwh: pd.Series,*,timestep_hours: float | None = None) -> pyo.ConcreteModel:
    """
    Build a simple hourly storage arbitrage model for a single vehicle/storage unit.

    Variables per hour t:
      - p_ch[t]  >= 0  (kW)
      - p_dis[t] >= 0  (kW)
      - soc[t]   >= 0  (kWh)

    SOC dynamics (Δt=1h):
      soc[t] = soc[t-1] + eta_c*p_ch[t] - (1/eta_d)*p_dis[t]
      with soc[0] = soc0 + eta_c*p_ch[0] - (1/eta_d)*p_dis[0]

    Bounds:
      soc_min*capacity <= soc[t] <= soc_max*capacity
      0 <= p_ch[t]  <= p_charge_max
      0 <= p_dis[t] <= p_discharge_max

    Objective:
      Maximize profit = sum_t price[t] * (p_dis[t] - p_ch[t])
      (price in EUR/kWh, power in kW, Δt=1h → EUR)
    """
    if not isinstance(prices_eur_per_kwh.index, pd.DatetimeIndex):
        raise ValueError("prices must be a pandas Series with a DatetimeIndex")

    # Ensure chronological order
    prices = prices_eur_per_kwh.sort_index()
    T = range(len(prices))

    # Infer Δt (in hours) from first two timestamps if not provided
    if timestep_hours is None:
        if len(prices.index) < 2:
            raise ValueError("Need at least two timestamps to infer timestep.")
        dt_seconds = (prices.index[1] - prices.index[0]).total_seconds()
        timestep_hours = dt_seconds / 3600.0

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, len(prices) - 1)

    # Parameters
    m.price = pyo.Param(m.T, initialize={t: float(prices.iloc[t]) for t in T}, mutable=False)
    m.dt = pyo.Param(initialize=float(timestep_hours))  # hours per step

    m.cap = pyo.Param(initialize=float(vehicle.capacity_kwh))
    m.soc_min = pyo.Param(initialize=float(vehicle.soc_min) * float(vehicle.capacity_kwh))
    m.soc_max = pyo.Param(initialize=float(vehicle.soc_max) * float(vehicle.capacity_kwh))
    m.soc0 = pyo.Param(initialize=float(vehicle.soc0) * float(vehicle.capacity_kwh))
    m.p_ch_max = pyo.Param(initialize=float(vehicle.p_charge_max_kw))
    m.p_dis_max = pyo.Param(initialize=float(vehicle.p_discharge_max_kw))
    m.eta_c = pyo.Param(initialize=float(vehicle.eta_charge))
    m.eta_d = pyo.Param(initialize=float(vehicle.eta_discharge))


    # Variables
    m.p_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.p_dis = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.soc = pyo.Var(m.T, within=pyo.NonNegativeReals)

    # SOC dynamics with Δt [h]: energy change = power * time
    def soc_rule(m, t):
        if t == 0:
            return m.soc[t] == m.soc0 + m.eta_c * m.p_ch[t] * m.dt - (1.0 / m.eta_d) * m.p_dis[t] * m.dt
        return m.soc[t] == m.soc[t - 1] + m.eta_c * m.p_ch[t] * m.dt - (1.0 / m.eta_d) * m.p_dis[t] * m.dt

    m.soc_dyn = pyo.Constraint(m.T, rule=soc_rule)

    # Bounds
    m.soc_lb = pyo.Constraint(m.T, rule=lambda m, t: m.soc[t] >= m.soc_min)
    m.soc_ub = pyo.Constraint(m.T, rule=lambda m, t: m.soc[t] <= m.soc_max)
    m.ch_lim = pyo.Constraint(m.T, rule=lambda m, t: m.p_ch[t] <= m.p_ch_max)
    m.dis_lim = pyo.Constraint(m.T, rule=lambda m, t: m.p_dis[t] <= m.p_dis_max)

    # Objective: profit = sum price [€/kWh] * (net energy [kWh]) = sum price * (kW * h)
    m.obj = pyo.Objective(
        expr=sum(m.price[t] * (m.p_dis[t] - m.p_ch[t]) * m.dt for t in m.T),
        sense=pyo.maximize
    )

    return m


def vehicle_commercialization(
    vehicle: Vehicle,
    prices_by_market: Dict[str, pd.Series],  # z.B. {"DA": da_series, "ID": id_series}
    *,
    timestep_hours: float | None = None,
) -> pyo.ConcreteModel:
    """Generisches Modell mit beliebigen Märkten (DA, ID, etc.)."""

    # 1) Validierung & Alignment
    markets: Iterable[str] = list(prices_by_market.keys())
    if not markets:
        raise ValueError("prices_by_market must contain at least one market")

    # Nimm erste Serie als Referenz
    first_market = next(iter(markets))
    ref = prices_by_market[first_market].sort_index()
    if not isinstance(ref.index, pd.DatetimeIndex):
        raise ValueError("price series must have DatetimeIndex")

    # Prüfe, dass alle Märkte dieselben Timestamps haben
    for mkt, s in prices_by_market.items():
        s_sorted = s.sort_index()
        if not s_sorted.index.equals(ref.index):
            raise ValueError(f"Timestamps for market {mkt} do not match reference market {first_market}")
        prices_by_market[mkt] = s_sorted  # überschreibe mit sortierter Serie

    T = range(len(ref))

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
        return float(s.iloc[t])
    m.price = pyo.Param(m.MARKETS, m.T, initialize=price_init)

    m.dt = pyo.Param(initialize=float(timestep_hours))

    # Fahrzeugparameter wie gehabt
    m.cap = pyo.Param(initialize=float(vehicle.capacity_kwh))
    m.soc_min = pyo.Param(initialize=float(vehicle.soc_min) * float(vehicle.capacity_kwh))
    m.soc_max = pyo.Param(initialize=float(vehicle.soc_max) * float(vehicle.capacity_kwh))
    m.soc0 = pyo.Param(initialize=float(vehicle.soc0) * float(vehicle.capacity_kwh))
    m.p_ch_max = pyo.Param(initialize=float(vehicle.p_charge_max_kw))
    m.p_dis_max = pyo.Param(initialize=float(vehicle.p_discharge_max_kw))
    m.eta_c = pyo.Param(initialize=float(vehicle.eta_charge))
    m.eta_d = pyo.Param(initialize=float(vehicle.eta_discharge))

    # Max. Marktleistung (symmetrisch), hier einfach an physische Leistung gekoppelt
    P_market_max = max(float(vehicle.p_charge_max_kw), float(vehicle.p_discharge_max_kw))
    m.p_market_max = pyo.Param(initialize=P_market_max)

    # Physik
    m.p_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.p_dis = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.soc = pyo.Var(m.T, within=pyo.NonNegativeReals)

    # Markt-Variablen: dürfen positiv (Verkauf) oder negativ (Kauf) sein
    m.p_market = pyo.Var(m.MARKETS,m.T,bounds=lambda mdl, mk, t: (-mdl.p_market_max, mdl.p_market_max))

    # SOC-Dynamik
    def soc_rule(mdl, t):
        if t == 0:
            return mdl.soc[t] == mdl.soc0 + mdl.eta_c * mdl.p_ch[t] * mdl.dt - (1.0 / mdl.eta_d) * mdl.p_dis[t] * mdl.dt
        return mdl.soc[t] == mdl.soc[t-1] + mdl.eta_c * mdl.p_ch[t] * mdl.dt - (1.0 / mdl.eta_d) * mdl.p_dis[t] * mdl.dt
    m.soc_dyn = pyo.Constraint(m.T, rule=soc_rule)

    # Bounds
    m.soc_lb = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.soc[t] >= mdl.soc_min)
    m.soc_ub = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.soc[t] <= mdl.soc_max)
    m.ch_lim = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_ch[t] <= mdl.p_ch_max)
    m.dis_lim = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_dis[t] <= mdl.p_dis_max)

    # Kopplung: Summe der Marktpositionen = physische Einspeisung
    def market_balance_rule(mdl, t):
        return sum(mdl.p_market[mk, t] for mk in mdl.MARKETS) == mdl.p_dis[t] - mdl.p_ch[t]
    m.market_balance = pyo.Constraint(m.T, rule=market_balance_rule)

    # Zielfunktion: Summe über alle Märkte und Zeiten
    def obj_expr(mdl):
        return sum(
            mdl.price[mk, t] * mdl.p_market[mk, t] * mdl.dt
            for mk in mdl.MARKETS
            for t in mdl.T
        )
    m.obj = pyo.Objective(expr=obj_expr(m), sense=pyo.maximize)

    return m