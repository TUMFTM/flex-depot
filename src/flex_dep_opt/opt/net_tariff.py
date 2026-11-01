"""Grid usage fee terms for the Pyomo commercialization model.

Adds a billable grid-import quantity, a running annual peak and the matching objective cost
(energy price on billable import plus capacity price on the incremental yearly peak). With the
MiSpeL-A11 privilege the billable import is netted per calendar month against export, seeded by
the month-to-date balance carried through the MPC.

The term is attached as ``m.obj_grid_fee_cost`` and is always created (``0`` when disabled) so
the objective sum stays structurally identical and disabled runs are unaffected.
"""

from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo

from ..market.grid_fees import month_key


def add_grid_fee_terms(
    m: pyo.ConcreteModel,
    time_index: pd.DatetimeIndex,
    *,
    enabled: bool,
    energy_price_eur_per_kwh: float = 0.0,
    capacity_price_eur_per_kw_year: float = 0.0,
    previous_annual_peak_kw: float = 0.0,
    mispel_applicable: bool = False,
    monthly_realized_net_kwh: dict[tuple[int, int], float] | None = None,
    peak_shaving: bool = True,
) -> None:
    """Attach grid-fee variables, constraints and the ``obj_grid_fee_cost`` expression to ``m``.

    With ``peak_shaving=False`` the capacity price is left out of the objective, so the optimizer
    only reacts to the energy charge; the demand charge is then settled downstream on the realized
    peak (the grid connection is capped at the reference peak by the caller).
    """
    if not enabled:
        m.obj_grid_fee_cost = pyo.Expression(expr=0.0)
        return

    ap = float(energy_price_eur_per_kwh)
    lp = float(capacity_price_eur_per_kw_year)
    prev_peak = float(previous_annual_peak_kw)
    realized_net = dict(monthly_realized_net_kwh or {})

    # Billable grid import P_import = max(0, p_net). The lower bound plus the cost pressure of the
    # energy and capacity terms drive it onto max(0, p_net); where the directional binary exists it
    # is pinned exactly to zero in export mode, so no artificial simultaneous import/export arises.
    m.p_grid_imp = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.grid_imp_ge_net = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_grid_imp[t] >= mdl.p_net[t])
    if hasattr(m, "u_state"):
        m.grid_imp_mode_limit = pyo.Constraint(
            m.T, rule=lambda mdl, t: mdl.p_grid_imp[t] <= mdl.P_ch_max_t[t] * (1.0 - mdl.u_state[t])
        )

    # Running annual peak import: at least every import step and at least the peak already reached
    # earlier this calendar year (seeded via the MPC carry-over). Only built when the optimizer is
    # allowed to shave; otherwise the demand charge is settled on the realized peak downstream.
    if peak_shaving:
        m.peak_prev = pyo.Param(initialize=prev_peak, mutable=True)
        m.p_peak_annual = pyo.Var(within=pyo.NonNegativeReals)
        m.peak_ge_import = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_peak_annual >= mdl.p_grid_imp[t])
        m.peak_ge_prev = pyo.Constraint(expr=m.p_peak_annual >= m.peak_prev)

    # Energy charge on billable import.
    if mispel_applicable:
        # Net billable import per calendar month: max(0, month-to-date net + planned net in window).
        # Export offsets import only within the month and never becomes a credit (billable >= 0).
        months: dict[tuple[int, int], list[int]] = {}
        for t, ts in enumerate(time_index):
            months.setdefault(month_key(ts), []).append(t)
        month_list = list(months.keys())

        m.GRID_MONTHS = pyo.Set(initialize=range(len(month_list)))
        m.grid_month_realized_net = pyo.Param(
            m.GRID_MONTHS,
            initialize={i: float(realized_net.get(key, 0.0)) for i, key in enumerate(month_list)},
            mutable=True,
        )
        m.grid_billable_month = pyo.Var(m.GRID_MONTHS, within=pyo.NonNegativeReals)

        def _billable_rule(mdl, i):
            planned_net = pyo.quicksum(mdl.p_net[t] * mdl.dt for t in months[month_list[i]])
            return mdl.grid_billable_month[i] >= mdl.grid_month_realized_net[i] + planned_net

        m.grid_billable_def = pyo.Constraint(m.GRID_MONTHS, rule=_billable_rule)
        energy_cost = ap * pyo.quicksum(m.grid_billable_month[i] for i in m.GRID_MONTHS)
    else:
        energy_cost = ap * pyo.quicksum(m.p_grid_imp[t] * m.dt for t in m.T)

    # Only additional peaks above the previously realized annual peak are penalized. With peak
    # shaving off the capacity price is not an optimization incentive (settled downstream instead).
    capacity_cost = lp * (m.p_peak_annual - m.peak_prev) if peak_shaving else 0.0

    m.obj_grid_fee_cost = pyo.Expression(expr=energy_cost + capacity_cost)
