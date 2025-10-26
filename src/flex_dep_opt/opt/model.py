from __future__ import annotations
import pyomo.environ as pyo
import pandas as pd
from ..domain.vehicle import Vehicle


def build_single_vehicle_model(vehicle: Vehicle, prices_eur_per_kwh: pd.Series) -> pyo.ConcreteModel:
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

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, len(prices) - 1)

    # Parameters
    m.price = pyo.Param(m.T, initialize={t: float(prices.iloc[t]) for t in T}, mutable=False)
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

    # SOC dynamics
    def soc_rule(m, t):
        if t == 0:
            return m.soc[t] == m.soc0 + m.eta_c * m.p_ch[t] - (1.0 / m.eta_d) * m.p_dis[t]
        return m.soc[t] == m.soc[t - 1] + m.eta_c * m.p_ch[t] - (1.0 / m.eta_d) * m.p_dis[t]

    m.soc_dyn = pyo.Constraint(m.T, rule=soc_rule)

    # Bounds
    m.soc_lb = pyo.Constraint(m.T, rule=lambda m, t: m.soc[t] >= m.soc_min)
    m.soc_ub = pyo.Constraint(m.T, rule=lambda m, t: m.soc[t] <= m.soc_max)
    m.ch_lim = pyo.Constraint(m.T, rule=lambda m, t: m.p_ch[t] <= m.p_ch_max)
    m.dis_lim = pyo.Constraint(m.T, rule=lambda m, t: m.p_dis[t] <= m.p_dis_max)

    # Objective (maximize profit)
    m.obj = pyo.Objective(expr=sum(m.price[t] * (m.p_dis[t] - m.p_ch[t]) for t in m.T), sense=pyo.maximize)

    return m