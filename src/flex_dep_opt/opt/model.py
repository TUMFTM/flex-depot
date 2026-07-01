from __future__ import annotations

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from flex_dep_opt.market.fcr import FCR_DROOP_ABS_COL, FCR_DROOP_COL

from ..domain.depot import Depot


def flexibility_commercialization(
    depot: Depot,
    prices_by_market: dict[str, pd.Series],
    fee_eur_per_kwh_by_market: dict[str, float] | None = None,
    *,
    timestep_hours: float | None = None,
    virtual_arbitrage: bool = True,
    cycling_cost_eur_per_kwh: float = 0.0,
    market_activity_mask: dict[str, pd.Series],
    committed_positions: dict[str, pd.Series],
    flexibility_bounds: pd.DataFrame,
    allow_imbalance: bool = False,
    imbalance_prices_pos: pd.Series | None = None,
    imbalance_prices_neg: pd.Series | None = None,
    imbalance_volume_penalty_eur_per_kwh: float = 0.0,
    fcr_prices: pd.Series | None = None,
    fcr_frequency_data: pd.DataFrame | None = None,
    fcr_product_hours: float = 4.0,
    fcr_bid_block_kw: float = 1000.0,
    fcr_cap_max_by_slot: dict | None = None,
    fcr_slot_hours_by_slot: dict | None = None,
    fcr_energy_reserve_kwh_per_kw: float = 0.0,
    fcr_reserve_penalty_eur_per_kwh: float = 0.0,
    fcr_balance_penalty_eur_per_kwh: float = 0.0,
) -> pyo.ConcreteModel:
    """
    Flex-band based multi-market commercialization model (Pyomo).

    Sign conventions
    ----------------
    - p_net positive = import (charge), negative = export (discharge).
      p_net is the PHYSICAL grid flow and equals p_ch - p_dis.
    - Market positions follow the same convention (+ = depot imports / buys).
    - FCR activation is folded into the balance via p_droop (import-positive):
      p_droop[t] = -droop[t] * x_fcr[j(t)]
      droop > 0 (low freq) -> upward FCR (export) -> p_droop < 0
      droop < 0 (high freq) -> downward FCR (import) -> p_droop > 0

    Time axes
    ---------
    - Decisions live on T = 0..N-1.
    - Energy states live on S = 0..N (N+1 points).
    - Net power: p_net[t] = p_ch[t] - p_dis[t], both >= 0.

    Units
    -----
    Power kW, energy kWh, prices/fees EUR/kWh, objective EUR.
    """

    # ============================================================
    # 1) Time axis and derived constants
    # ============================================================
    markets = list(prices_by_market.keys())
    ref = prices_by_market[markets[0]].sort_index()
    time_index = ref.index
    N = len(time_index)

    if timestep_hours is None:
        timestep_hours = (time_index[1] - time_index[0]).total_seconds() / 3600.0
    timestep_hours = float(timestep_hours)

    fee_eur_per_kwh_by_market = fee_eur_per_kwh_by_market or {}
    fee_value = {mk: float(fee_eur_per_kwh_by_market.get(mk, 0.0)) for mk in markets}
    any_fee = any(v > 0.0 for v in fee_value.values())

    price_arr = {mk: prices_by_market[mk].to_numpy(dtype=float) for mk in markets}
    mask_arr = {mk: np.asarray(market_activity_mask[mk].to_numpy(), dtype=bool) for mk in markets}
    committed_arr = {mk: committed_positions[mk].to_numpy(dtype=float) for mk in markets}

    P_lower_arr = flexibility_bounds["Power_lower_kW"].to_numpy(dtype=float)
    P_upper_arr = flexibility_bounds["Power_upper_kW"].to_numpy(dtype=float)
    E_lower_arr = flexibility_bounds["Capacity_lower_kWh"].to_numpy(dtype=float)
    E_upper_arr = flexibility_bounds["Capacity_upper_kWh"].to_numpy(dtype=float)

    if allow_imbalance:
        imb_pos_arr = imbalance_prices_pos.to_numpy(dtype=float)
        imb_neg_arr = imbalance_prices_neg.to_numpy(dtype=float)

    p_market_max_value = float(max(P_upper_arr.max(), (-P_lower_arr).max()))

    eta_c_val = float(depot.eta_grid2depot)
    eta_d_val = float(depot.eta_depot2grid)

    # ============================================================
    # 2) FCR pre-processing (slots, droop signal, hidden-cycling signal)
    # ============================================================
    use_fcr = False
    slot_steps: dict[int, list[int]] = {}
    fcr_slot_starts: list = []
    fcr_slot_price_vals: list[float] = []
    fcr_cap_max_vals: dict[int, float] = {}
    fcr_slot_hours_vals: dict[int, float] = {}

    if fcr_prices is not None and not fcr_prices.empty:
        j = 0
        for slot_start in fcr_prices.index:
            slot_end = slot_start + pd.Timedelta(hours=fcr_product_hours)
            mask = np.asarray((time_index >= slot_start) & (time_index < slot_end))
            steps = np.flatnonzero(mask).tolist()
            if not steps:
                continue
            slot_steps[j] = steps
            fcr_slot_starts.append(slot_start)
            fcr_slot_price_vals.append(float(fcr_prices.loc[slot_start]))

            if fcr_cap_max_by_slot is not None and slot_start in fcr_cap_max_by_slot:
                fcr_cap_max_vals[j] = max(float(fcr_cap_max_by_slot[slot_start]), 0.0)
            else:
                idx = np.asarray(steps)
                cap = float(np.minimum(P_upper_arr[idx], -P_lower_arr[idx]).min())
                fcr_cap_max_vals[j] = max(cap, 0.0)

            if fcr_slot_hours_by_slot is not None and slot_start in fcr_slot_hours_by_slot:
                fcr_slot_hours_vals[j] = float(fcr_slot_hours_by_slot[slot_start])
            else:
                fcr_slot_hours_vals[j] = len(steps) * timestep_hours

            j += 1

        use_fcr = len(fcr_slot_starts) > 0

    t_to_fcr_slot: dict[int, int] = {}
    for j, steps in slot_steps.items():
        for t in steps:
            t_to_fcr_slot[t] = j

    # FCR activation is folded into the power balance (sum(p_market) + p_droop
    # = p_net), so a single market position may have to carry both the net flow
    # and the FCR offset: |p_market| can reach |p_net| + |p_droop| up to
    # p_market_max_value + max(x_fcr). Widen the per-market variable bound by the
    # largest biddable FCR capacity so this envelope never binds artificially.
    fcr_headroom = max(fcr_cap_max_vals.values()) if (use_fcr and fcr_cap_max_vals) else 0.0
    p_market_bound = p_market_max_value + fcr_headroom

    # Per-step signed droop d[t] in [-1, +1] (sign convention: see p_droop below).
    droop_by_t = np.zeros(N, dtype=float)
    hidden_droop_by_t = np.zeros(N, dtype=float)
    if use_fcr and fcr_frequency_data is not None and not fcr_frequency_data.empty:
        col_vals = fcr_frequency_data[FCR_DROOP_COL].to_numpy(dtype=float)
        abs_vals = (
            fcr_frequency_data[FCR_DROOP_ABS_COL].to_numpy(dtype=float)
            if FCR_DROOP_ABS_COL in fcr_frequency_data.columns
            else None
        )
        locs = fcr_frequency_data.index.get_indexer(time_index, method="nearest")
        for t, loc in enumerate(locs):
            if loc >= 0 and t in t_to_fcr_slot:
                droop_by_t[t] = float(np.clip(col_vals[loc], -1.0, 1.0))
                if abs_vals is not None:
                    hidden_droop_by_t[t] = max(float(abs_vals[loc]) - abs(droop_by_t[t]), 0.0)

    # ============================================================
    # 3) Pyomo model (Sets / Params)
    # ============================================================
    m = pyo.ConcreteModel()

    m.T = pyo.RangeSet(0, N - 1)
    m.S = pyo.RangeSet(0, N)

    m.MARKETS = pyo.Set(initialize=markets)
    m.dt = pyo.Param(initialize=timestep_hours)

    m.price = pyo.Param(m.MARKETS, m.T, initialize=lambda mdl, mk, t: float(price_arr[mk][t]))
    m.fee = pyo.Param(m.MARKETS, initialize=lambda mdl, mk: fee_value[mk], within=pyo.NonNegativeReals)

    m.P_lower = pyo.Param(m.T, initialize=lambda mdl, t: float(P_lower_arr[t]))
    m.P_upper = pyo.Param(m.T, initialize=lambda mdl, t: float(P_upper_arr[t]))
    m.E_lower = pyo.Param(m.S, initialize=lambda mdl, s: float(E_lower_arr[s]))
    m.E_upper = pyo.Param(m.S, initialize=lambda mdl, s: float(E_upper_arr[s]))

    m.eta_c = pyo.Param(initialize=eta_c_val)
    m.eta_d = pyo.Param(initialize=eta_d_val)
    m.grid_limit = pyo.Param(initialize=float(depot.grid_connection_limit))

    m.c_deg = pyo.Param(initialize=float(cycling_cost_eur_per_kwh))
    m.c_imb_vol = pyo.Param(initialize=float(imbalance_volume_penalty_eur_per_kwh))

    m.E0 = pyo.Param(initialize=0.0, mutable=True)
    m.Eterm = pyo.Param(initialize=0.0, mutable=True)
    m.w_term = pyo.Param(initialize=0.0, mutable=True, within=pyo.NonNegativeReals)

    closed_pairs = [(mk, t) for mk in markets for t in range(N) if not bool(mask_arr[mk][t])]
    m.MARKET_CLOSED = pyo.Set(initialize=closed_pairs, dimen=2)
    m.p_market_committed = pyo.Param(
        m.MARKET_CLOSED,
        initialize={(mk, t): float(committed_arr[mk][t]) for (mk, t) in closed_pairs},
    )

    # ============================================================
    # 4) FCR variables, parameters, and expressions
    # ============================================================
    if use_fcr:
        n_fcr = len(fcr_slot_starts)
        m.S_FCR = pyo.RangeSet(0, n_fcr - 1)

        m.fcr_price_param = pyo.Param(m.S_FCR, initialize=lambda mdl, j: fcr_slot_price_vals[j])
        m.fcr_slot_hours = pyo.Param(m.S_FCR, initialize=lambda mdl, j: fcr_slot_hours_vals[j])
        m.fcr_droop_signal = pyo.Param(m.T, initialize=lambda mdl, t: float(droop_by_t[t]))
        m.fcr_hidden_droop = pyo.Param(m.T, initialize=lambda mdl, t: float(hidden_droop_by_t[t]))

        m.x_fcr_committed = pyo.Param(m.S_FCR, initialize=0.0, mutable=True)
        m.fcr_gate_open = pyo.Param(m.S_FCR, initialize=1, mutable=True, within=pyo.Binary)

        # Bid as integer multiples of the configured block size (default 1 MW =
        # 1000 kW); x_fcr expressed in kW.
        m.z_fcr = pyo.Var(
            m.S_FCR,
            within=pyo.NonNegativeIntegers,
            bounds=lambda mdl, j: (0, int(fcr_cap_max_vals[j] // fcr_bid_block_kw)),
        )
        m.x_fcr = pyo.Expression(m.S_FCR, rule=lambda mdl, j: fcr_bid_block_kw * mdl.z_fcr[j])

        # Per-step FCR activation power as a market position (import-positive):
        # p_droop[t] = -droop[t] * x_fcr[j(t)].
        # droop > 0 -> upward FCR (export) -> p_droop < 0
        # droop < 0 -> downward FCR (import) -> p_droop > 0
        def _p_droop_rule(mdl, t):
            j = t_to_fcr_slot.get(t)
            if j is None:
                return 0.0
            return -mdl.fcr_droop_signal[t] * mdl.x_fcr[j]

        m.p_droop = pyo.Expression(m.T, rule=_p_droop_rule)

        # EUR/MW price -> EUR/kW (/1000); prorate by covered fraction of the 4h product.
        m.fcr_slot_revenue = pyo.Expression(
            m.S_FCR,
            rule=lambda mdl, j: (
                (mdl.fcr_price_param[j] / 1000.0) * mdl.x_fcr[j] * (mdl.fcr_slot_hours[j] / fcr_product_hours)
            ),
        )

        # Exposed for result extraction.
        m._fcr_slot_starts = list(fcr_slot_starts)
        m._fcr_product_hours = float(fcr_product_hours)

    # ============================================================
    # 5) Decision variables
    # ============================================================
    m.p_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.p_dis = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.p_net = pyo.Var(m.T, within=pyo.Reals)
    m.E = pyo.Var(m.S, within=pyo.Reals)

    m.p_market = pyo.Var(m.MARKETS, m.T, bounds=(-p_market_bound, p_market_bound))

    if allow_imbalance:
        m.p_imb_pos = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.p_imb_neg = pyo.Var(m.T, within=pyo.NonNegativeReals)

        m.price_imb_pos = pyo.Param(m.T, initialize=lambda mdl, t: float(imb_pos_arr[t]))
        m.price_imb_neg = pyo.Param(m.T, initialize=lambda mdl, t: float(imb_neg_arr[t]))

    m.e_term_dev = pyo.Var(within=pyo.NonNegativeReals)

    # ============================================================
    # 6) Physical constraints
    # ============================================================
    # p_net IS the physical grid flow (import-positive). FCR activation flows
    # through p_ch/p_dis just like any market position, so the SoC update is
    # the standard battery transition with no separate FCR throughput term.
    m.energy_init = pyo.Constraint(expr=m.E[0] == m.E0)

    m.p_net_def = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_net[t] == mdl.p_ch[t] - mdl.p_dis[t])

    m.p_band_lb = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_net[t] >= mdl.P_lower[t])
    m.p_band_ub = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_net[t] <= mdl.P_upper[t])

    m.grid_ub = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_net[t] <= mdl.grid_limit)
    m.grid_lb = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.p_net[t] >= -mdl.grid_limit)

    m.energy_state = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: (
            mdl.E[t + 1]
            == mdl.E[t] + mdl.eta_c * mdl.p_ch[t] * mdl.dt - (1.0 / mdl.eta_d) * mdl.p_dis[t] * mdl.dt
        ),
    )

    if allow_imbalance:
        m.e_slack_up = pyo.Var(m.S, within=pyo.NonNegativeReals)
        m.e_slack_dn = pyo.Var(m.S, within=pyo.NonNegativeReals)
        m.E_lb = pyo.Constraint(m.S, rule=lambda mdl, s: mdl.E[s] + mdl.e_slack_dn[s] >= mdl.E_lower[s])
        m.E_ub = pyo.Constraint(m.S, rule=lambda mdl, s: mdl.E[s] - mdl.e_slack_up[s] <= mdl.E_upper[s])
    else:
        m.E_lb = pyo.Constraint(m.S, rule=lambda mdl, s: mdl.E[s] >= mdl.E_lower[s])
        m.E_ub = pyo.Constraint(m.S, rule=lambda mdl, s: mdl.E[s] <= mdl.E_upper[s])

    last_s = N
    m.term_dev_pos = pyo.Constraint(expr=m.E[last_s] - m.Eterm <= m.e_term_dev)
    m.term_dev_neg = pyo.Constraint(expr=m.Eterm - m.E[last_s] <= m.e_term_dev)
    m.energy_term_hard = pyo.Constraint(expr=m.E[last_s] == m.Eterm)
    m.energy_term_hard.deactivate()

    # ============================================================
    # 7) Market coupling + gate closure
    # ============================================================
    m.market_activity = pyo.Constraint(
        m.MARKET_CLOSED,
        rule=lambda mdl, mk, t: mdl.p_market[mk, t] == mdl.p_market_committed[mk, t],
    )

    def balance_rule(mdl, t):
        base = pyo.quicksum(mdl.p_market[mk, t] for mk in mdl.MARKETS)
        if use_fcr:
            base = base + mdl.p_droop[t]
        if allow_imbalance:
            return base + mdl.p_imb_pos[t] - mdl.p_imb_neg[t] == mdl.p_net[t]
        return base == mdl.p_net[t]

    m.market_balance = pyo.Constraint(m.T, rule=balance_rule)

    # ============================================================
    # 8) FCR gate closure constraints
    # ============================================================
    if use_fcr:
        # While the gate is open, x_fcr is free up to its biddable capacity
        # (via the z_fcr integer bound). When the gate closes, fcr_gate_open=0
        # and the offer is pinned to x_fcr_committed.
        m.fcr_gate_ub = pyo.Constraint(
            m.S_FCR,
            rule=lambda mdl, j: (
                mdl.x_fcr[j] <= mdl.x_fcr_committed[j] + fcr_cap_max_vals[j] * mdl.fcr_gate_open[j]
            ),
        )
        m.fcr_gate_lb = pyo.Constraint(
            m.S_FCR,
            rule=lambda mdl, j: (
                mdl.x_fcr[j] >= mdl.x_fcr_committed[j] - fcr_cap_max_vals[j] * mdl.fcr_gate_open[j]
            ),
        )

        # --------------------------------------------------------
        # 8b) FCR energy reserve — soft SoC headroom for droop
        # --------------------------------------------------------
        # Steps covered by an FCR slot need SoC headroom so unforeseen droop
        # activation does not push E through its bounds (which would otherwise
        # cascade into imbalance). The reserve is r kWh per kW of committed FCR
        # capacity, carved out of both ends of the band.
        #
        # SOFT constraint: E is allowed into the reserve buffer, but
        # every kWh of intrusion is penalised in the objective.
        #
        # The reserve constrains the planned trajectory E[1..N]; E[0] is the
        # realized starting SoC (fixed by energy_init) and is skipped.
        if fcr_energy_reserve_kwh_per_kw > 0.0:
            r = float(fcr_energy_reserve_kwh_per_kw)
            m.T_FCR = pyo.Set(initialize=sorted(t_to_fcr_slot.keys()))

            reserve_states = sorted({s for t in t_to_fcr_slot for s in (t, t + 1) if s != 0})
            m.S_FCR_RESERVE = pyo.Set(initialize=reserve_states)
            m.fcr_reserve_slack_up = pyo.Var(m.S_FCR_RESERVE, within=pyo.NonNegativeReals)
            m.fcr_reserve_slack_dn = pyo.Var(m.S_FCR_RESERVE, within=pyo.NonNegativeReals)

            def _reserve_up(mdl, t):
                if t == 0:  # E[0] is realized, not planned — skip
                    return pyo.Constraint.Skip
                return (
                    mdl.E[t] <= mdl.E_upper[t] - r * mdl.x_fcr[t_to_fcr_slot[t]] + mdl.fcr_reserve_slack_up[t]
                )

            def _reserve_dn(mdl, t):
                if t == 0:  # E[0] is realized, not planned — skip
                    return pyo.Constraint.Skip
                return (
                    mdl.E[t] >= mdl.E_lower[t] + r * mdl.x_fcr[t_to_fcr_slot[t]] - mdl.fcr_reserve_slack_dn[t]
                )

            m.fcr_E_reserve_up = pyo.Constraint(m.T_FCR, rule=_reserve_up)
            m.fcr_E_reserve_dn = pyo.Constraint(m.T_FCR, rule=_reserve_dn)
            m.fcr_E_reserve_up_next = pyo.Constraint(
                m.T_FCR,
                rule=lambda mdl, t: (
                    mdl.E[t + 1]
                    <= mdl.E_upper[t + 1] - r * mdl.x_fcr[t_to_fcr_slot[t]] + mdl.fcr_reserve_slack_up[t + 1]
                ),
            )
            m.fcr_E_reserve_dn_next = pyo.Constraint(
                m.T_FCR,
                rule=lambda mdl, t: (
                    mdl.E[t + 1]
                    >= mdl.E_lower[t + 1] + r * mdl.x_fcr[t_to_fcr_slot[t]] - mdl.fcr_reserve_slack_dn[t + 1]
                ),
            )

        # Soft mid-band pull on FCR-covered states: discourages E from sitting
        # right at the (tightened) reserve edge. Objective term only, so it can
        # never cause infeasibility — active in both passes.
        if fcr_balance_penalty_eur_per_kwh > 0.0:
            fcr_states = sorted({s for t in t_to_fcr_slot for s in (t, t + 1)})
            if fcr_states:
                m.S_FCR_BAL = pyo.Set(initialize=fcr_states)
                m.E_mid = pyo.Param(
                    m.S_FCR_BAL,
                    initialize=lambda mdl, s: 0.5 * (float(E_lower_arr[s]) + float(E_upper_arr[s])),
                )
                m.e_balance_dev = pyo.Var(m.S_FCR_BAL, within=pyo.NonNegativeReals)
                m.fcr_bal_up = pyo.Constraint(
                    m.S_FCR_BAL,
                    rule=lambda mdl, s: mdl.e_balance_dev[s] >= mdl.E[s] - mdl.E_mid[s],
                )
                m.fcr_bal_dn = pyo.Constraint(
                    m.S_FCR_BAL,
                    rule=lambda mdl, s: mdl.e_balance_dev[s] >= mdl.E_mid[s] - mdl.E[s],
                )

    # ============================================================
    # 9) Virtual arbitrage handling (LP vs MILP)
    # ============================================================
    if virtual_arbitrage:
        if any_fee:
            m.p_market_abs = pyo.Var(m.MARKETS, m.T, within=pyo.NonNegativeReals)
            m.p_market_abs_pos = pyo.Constraint(
                m.MARKETS, m.T, rule=lambda mdl, mk, t: mdl.p_market_abs[mk, t] >= mdl.p_market[mk, t]
            )
            m.p_market_abs_neg = pyo.Constraint(
                m.MARKETS, m.T, rule=lambda mdl, mk, t: mdl.p_market_abs[mk, t] >= -mdl.p_market[mk, t]
            )
            m.p_market_vol = pyo.Expression(m.MARKETS, m.T, rule=lambda mdl, mk, t: mdl.p_market_abs[mk, t])
        else:
            m.p_market_vol = pyo.Expression(m.MARKETS, m.T, rule=lambda mdl, mk, t: 0.0)

    else:
        P_ch_max_arr = np.clip(P_upper_arr, 0.0, None)
        P_dis_max_arr = np.clip(-P_lower_arr, 0.0, None)
        m.P_ch_max_t = pyo.Param(m.T, initialize=lambda mdl, t: float(P_ch_max_arr[t]))
        m.P_dis_max_t = pyo.Param(m.T, initialize=lambda mdl, t: float(P_dis_max_arr[t]))

        m.u_state = pyo.Var(m.T, within=pyo.Binary)

        m.p_market_pos = pyo.Var(m.MARKETS, m.T, within=pyo.NonNegativeReals)
        m.p_market_neg = pyo.Var(m.MARKETS, m.T, within=pyo.NonNegativeReals)
        m.p_market_vol = pyo.Expression(
            m.MARKETS, m.T, rule=lambda mdl, mk, t: mdl.p_market_pos[mk, t] + mdl.p_market_neg[mk, t]
        )

        m.p_market_def = pyo.Constraint(
            m.MARKETS,
            m.T,
            rule=lambda mdl, mk, t: mdl.p_market[mk, t] == mdl.p_market_pos[mk, t] - mdl.p_market_neg[mk, t],
        )

        def total_export(mdl, t):
            return pyo.quicksum(mdl.p_market_neg[mk, t] for mk in mdl.MARKETS)

        def total_import(mdl, t):
            return pyo.quicksum(mdl.p_market_pos[mk, t] for mk in mdl.MARKETS)

        m.export_mode_limit = pyo.Constraint(
            m.T, rule=lambda mdl, t: total_export(mdl, t) <= mdl.P_dis_max_t[t] * mdl.u_state[t]
        )
        m.import_mode_limit = pyo.Constraint(
            m.T, rule=lambda mdl, t: total_import(mdl, t) <= mdl.P_ch_max_t[t] * (1.0 - mdl.u_state[t])
        )

        m.p_dis_mode_limit = pyo.Constraint(
            m.T, rule=lambda mdl, t: mdl.p_dis[t] <= mdl.P_dis_max_t[t] * mdl.u_state[t]
        )
        m.p_ch_mode_limit = pyo.Constraint(
            m.T, rule=lambda mdl, t: mdl.p_ch[t] <= mdl.P_ch_max_t[t] * (1.0 - mdl.u_state[t])
        )

    # ============================================================
    # 10) Objective
    # ============================================================
    # Named per-term Expressions so the objective can be decomposed term-by-term
    # (FCR breakeven batches); the objective below is their exact sum.
    m.obj_energy_cashflow = pyo.Expression(
        expr=pyo.quicksum(-m.price[mk, t] * m.p_market[mk, t] * m.dt for mk in m.MARKETS for t in m.T)
    )

    fee_cost = 0.0
    if any_fee:
        fee_cost = pyo.quicksum(m.fee[mk] * m.p_market_vol[mk, t] * m.dt for mk in m.MARKETS for t in m.T)
    m.obj_fee_cost = pyo.Expression(expr=fee_cost)

    # Cycling cost on total throughput; mean FCR activation is already in
    # p_ch/p_dis via the balance, so this captures cycling on |mean droop|.
    deg_cost = m.c_deg * pyo.quicksum((m.p_ch[t] + m.p_dis[t]) * m.dt for t in m.T)

    # Hidden FCR cycling: the battery follows the per-second signal (mean|droop|)
    # but the balance only sees |mean droop|. Charge the gap per committed kW so
    # the bid reflects true wear. Needs FREQ_DROOP_ABS_MEAN; gated by c_deg.
    if use_fcr:
        deg_cost = deg_cost + m.c_deg * pyo.quicksum(
            m.fcr_hidden_droop[t] * m.x_fcr[t_to_fcr_slot[t]] * m.dt for t in m.T if t in t_to_fcr_slot
        )
    m.obj_cycling_cost = pyo.Expression(expr=deg_cost)

    # Optional imbalance cashflow and volume penalty
    imb_cash = 0.0
    imb_vol_pen = 0.0
    if allow_imbalance:
        imb_cash = pyo.quicksum(
            (-m.price_imb_pos[t] * m.p_imb_pos[t] + m.price_imb_neg[t] * m.p_imb_neg[t]) * m.dt for t in m.T
        )
        imb_vol_pen = m.c_imb_vol * pyo.quicksum((m.p_imb_pos[t] + m.p_imb_neg[t]) * m.dt for t in m.T)
    m.obj_imb_cashflow = pyo.Expression(expr=imb_cash)
    m.obj_imb_vol_penalty = pyo.Expression(expr=imb_vol_pen)

    # Soft terminal objective (weight set by MPC)
    m.obj_term_penalty = pyo.Expression(expr=m.w_term * m.e_term_dev)

    # FCR capacity revenue (sum of the per-slot revenue expressions).
    m.obj_fcr_revenue = pyo.Expression(
        expr=pyo.quicksum(m.fcr_slot_revenue[j] for j in m.S_FCR) if use_fcr else 0.0
    )

    e_slack_pen = 0.0
    if allow_imbalance:
        # Heavy penalty — slack is a last resort to keep PASS2 feasible.
        e_slack_pen = 1e6 * pyo.quicksum(m.e_slack_up[s] + m.e_slack_dn[s] for s in m.S)
    m.obj_e_slack_penalty = pyo.Expression(expr=e_slack_pen)

    # Soft FCR energy-reserve penalty: cost per kWh that planned E intrudes into
    # the droop headroom buffer. Priced above arbitrage, below imbalance.
    reserve_pen = 0.0
    if fcr_reserve_penalty_eur_per_kwh > 0.0 and hasattr(m, "fcr_reserve_slack_up"):
        reserve_pen = fcr_reserve_penalty_eur_per_kwh * pyo.quicksum(
            m.fcr_reserve_slack_up[s] + m.fcr_reserve_slack_dn[s] for s in m.S_FCR_RESERVE
        )
    m.obj_reserve_penalty = pyo.Expression(expr=reserve_pen)

    # Soft FCR mid-band penalty: pulls E toward band centre on FCR steps;
    # dt-scaled for timestep-resolution independence.
    bal_pen = 0.0
    if fcr_balance_penalty_eur_per_kwh > 0.0 and hasattr(m, "e_balance_dev"):
        bal_pen = (
            fcr_balance_penalty_eur_per_kwh * m.dt * pyo.quicksum(m.e_balance_dev[s] for s in m.S_FCR_BAL)
        )
    m.obj_balance_penalty = pyo.Expression(expr=bal_pen)

    m.obj = pyo.Objective(
        expr=m.obj_energy_cashflow
        + m.obj_fcr_revenue
        + m.obj_imb_cashflow
        - m.obj_fee_cost
        - m.obj_cycling_cost
        - m.obj_imb_vol_penalty
        - m.obj_term_penalty
        - m.obj_e_slack_penalty
        - m.obj_reserve_penalty
        - m.obj_balance_penalty,
        sense=pyo.maximize,
    )

    return m
