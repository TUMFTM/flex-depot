from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from ..domain.depot import Depot
from flex_dep_opt.market.fcr import FCR_FREQ_COL, droop_signal


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
    fcr_prices: pd.Series | None = None,
    fcr_energy_req_hours: float | None = 1.5,
    fcr_enforce_power_headroom: bool = True,
    fcr_frequency_data: pd.DataFrame | None = None,
    frequency_nominal_hz: float = 50.0,
    frequency_deadband_hz: float = 0.010,
    frequency_full_activation_hz: float = 0.200,
    fcr_cap_max_by_slot: dict | None = None,
    fcr_slot_hours_by_slot: dict | None = None,
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

    # Tight per-timestep import/export maxima (Big-M values in MILP constraints)
    P_ch_max_arr = np.clip(P_upper_arr, 0.0, None)      # max import (>=0)
    P_dis_max_arr = np.clip(-P_lower_arr, 0.0, None)    # max export (>=0)

    if allow_imbalance:
        imb_pos_arr = imbalance_prices_pos.to_numpy(dtype=float)
        imb_neg_arr = imbalance_prices_neg.to_numpy(dtype=float)

    # Global symmetric bound for market positions derived from fleet power bands
    p_market_max_value = float(max(P_upper_arr.max(), (-P_lower_arr).max()))

    # ============================================================
    # 2) Pyomo model structure (Sets / Params)
    # ============================================================
    m = pyo.ConcreteModel()

    m.T = pyo.RangeSet(0, N - 1)  # decisions
    m.S = pyo.RangeSet(0, N)      # states (N+1)

    m.MARKETS = pyo.Set(initialize=markets)
    m.dt = pyo.Param(initialize=timestep_hours)

    # --- Market prices & fees (EUR/kWh) ---
    m.price = pyo.Param(m.MARKETS, m.T, initialize=lambda mdl, mk, t: float(price_arr[mk][t]))
    m.fee = pyo.Param(m.MARKETS, initialize=lambda mdl, mk: fee_value[mk], within=pyo.NonNegativeReals)

    # --- Flexibility bands ---
    m.P_lower = pyo.Param(m.T, initialize=lambda mdl, t: float(P_lower_arr[t]))
    m.P_upper = pyo.Param(m.T, initialize=lambda mdl, t: float(P_upper_arr[t]))
    m.E_lower = pyo.Param(m.S, initialize=lambda mdl, s: float(E_lower_arr[s]))
    m.E_upper = pyo.Param(m.S, initialize=lambda mdl, s: float(E_upper_arr[s]))

    # Tight Big-M bounds per timestep (for MILP mode):
    # per-timestep upper bounds used to link continuous power/market variables
    # with a binary import/export mode. M is the physically feasible maximum
    # power derived from the flexibility bands (not an arbitrary large constant),
    # which keeps the LP relaxation tight and numerically stable.
    m.P_ch_max_t = pyo.Param(m.T, initialize=lambda mdl, t: float(P_ch_max_arr[t]))
    m.P_dis_max_t = pyo.Param(m.T, initialize=lambda mdl, t: float(P_dis_max_arr[t]))

    # --- Depot parameters ---
    grid_limit_value = float(depot.grid_connection_limit)
    m.eta_c = pyo.Param(initialize=float(depot.eta_grid2depot))
    m.eta_d = pyo.Param(initialize=float(depot.eta_depot2grid))
    m.grid_limit = pyo.Param(initialize=grid_limit_value)

    # --- Cost coefficients ---
    m.c_deg = pyo.Param(initialize=float(cycling_cost_eur_per_kwh))
    m.c_imb_vol = pyo.Param(initialize=float(imbalance_volume_penalty_eur_per_kwh))

    # Global symmetric bound for market power
    m.p_market_max = pyo.Param(initialize=p_market_max_value)

    # Initial energy state (set by MPC workflow)
    m.E0 = pyo.Param(initialize=0.0, mutable=True)

    # Terminal target and weight (set by MPC workflow)
    m.Eterm = pyo.Param(initialize=0.0, mutable=True)
    m.w_term = pyo.Param(initialize=0.0, mutable=True, within=pyo.NonNegativeReals)

    # Committed market positions (fixed schedule after gate closure). Only the
    # closed (market, timestep) pairs are constrained, so we index those only.
    closed_pairs = [
        (mk, t) for mk in markets for t in range(N) if not bool(mask_arr[mk][t])
    ]
    m.MARKET_CLOSED = pyo.Set(initialize=closed_pairs, dimen=2)
    m.p_market_committed = pyo.Param(
        m.MARKET_CLOSED,
        initialize={(mk, t): float(committed_arr[mk][t]) for (mk, t) in closed_pairs},
    )

    # ============================================================
    # FCR slot setup
    # ============================================================
    use_fcr = False
    slot_steps: dict[int, list[int]] = {}
    fcr_slot_starts: list = []
    if fcr_prices is not None and not fcr_prices.empty:
        fcr_slot_price: list = []
        j = 0
        for slot_start in fcr_prices.index:
            slot_end = slot_start + pd.Timedelta(hours=4)
            mask = np.asarray((time_index >= slot_start) & (time_index < slot_end))
            steps = np.flatnonzero(mask).tolist()
            if steps:
                slot_steps[j] = steps
                fcr_slot_starts.append(slot_start)
                fcr_slot_price.append(float(fcr_prices.loc[slot_start]))
                j += 1

        n_fcr = len(fcr_slot_starts)
        if n_fcr > 0:
            use_fcr = True

            # Tightest symmetric power headroom across each slot bounds the offer,
            # and the covered hours scale the (4 h) capacity-product revenue.
            # When the caller supplies slot-local cap / hours (computed against the
            # full simulation horizon, not just this window), prefer those — the
            # window-local view underestimates cap for slots that extend past the
            # horizon and over-prorates revenue for slots that the rolling MPC
            # will fully cover in subsequent windows.
            fcr_cap_max_vals: dict[int, float] = {}
            fcr_slot_hours_vals: dict[int, float] = {}
            for j, steps in slot_steps.items():
                slot_start = fcr_slot_starts[j]
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

            m.S_FCR = pyo.RangeSet(0, n_fcr - 1)
            m.fcr_price_param = pyo.Param(m.S_FCR, initialize=lambda mdl, j: fcr_slot_price[j])
            m.fcr_cap_max_kw = pyo.Param(m.S_FCR, initialize=lambda mdl, j: fcr_cap_max_vals[j])
            m.fcr_slot_hours = pyo.Param(m.S_FCR, initialize=lambda mdl, j: fcr_slot_hours_vals[j])
            m.fcr_energy_req_hours = pyo.Param(
                initialize=float(fcr_energy_req_hours) if fcr_energy_req_hours else 0.0
            )

            m.x_fcr_committed = pyo.Param(m.S_FCR, initialize=0.0, mutable=True)
            m.fcr_gate_open = pyo.Param(m.S_FCR, initialize=1, mutable=True, within=pyo.Binary)

            fcr_jt_pairs = [(j, t) for j, steps in slot_steps.items() for t in steps]
            m.FCR_JT = pyo.Set(initialize=fcr_jt_pairs, dimen=2)

            # Exposed for result extraction (filtered to slots overlapping this window).
            m._fcr_slot_starts = list(fcr_slot_starts)

    # Reverse map: decision step -> FCR slot index (used by the forced droop power).
    t_to_fcr_slot: dict[int, int] = {}
    for j, steps in slot_steps.items():
        for t in steps:
            t_to_fcr_slot[t] = j

    # Forced droop activation fraction per decision step (0 outside FCR slots).
    droop_signal_by_t: dict[int, float] = {t: 0.0 for t in range(N)}
    if use_fcr and fcr_frequency_data is not None and not fcr_frequency_data.empty:
        freq_vals = fcr_frequency_data[FCR_FREQ_COL].to_numpy(dtype=float)
        locs = fcr_frequency_data.index.get_indexer(time_index, method="nearest")
        for t, loc in enumerate(locs):
            if loc >= 0 and t in t_to_fcr_slot:
                droop_signal_by_t[t] = droop_signal(
                    freq_vals[loc],
                    nominal_hz=frequency_nominal_hz,
                    deadband_hz=frequency_deadband_hz,
                    full_activation_hz=frequency_full_activation_hz,
                )

    m.fcr_droop_signal = pyo.Param(m.T, initialize=lambda mdl, t: droop_signal_by_t[t], mutable=True)

    # Per-step coefficients (computed at build time from the fixed droop signal):
    #   E_change[t]    = fcr_throughput_coef[t] * x_fcr[j(t)] * dt
    #   |p_droop[t]|   = fcr_activation_abs_coef[t] * x_fcr[j(t)]
    # signal > 0 -> upward FCR  -> depot exports -> energy leaves at 1/eta_d
    # signal < 0 -> downward FCR -> depot imports -> energy enters at eta_c
    # Mutating fcr_droop_signal post-build would *not* update these coefficients;
    # the MPC rebuilds the model every step, so this is consistent with the rest
    # of the build-time precomputation here.
    eta_c_val = float(depot.eta_grid2depot)
    eta_d_val = float(depot.eta_depot2grid)
    fcr_throughput_coef: dict[int, float] = {t: 0.0 for t in range(N)}
    fcr_activation_abs_coef: dict[int, float] = {t: 0.0 for t in range(N)}
    if use_fcr:
        for t, sig in droop_signal_by_t.items():
            if sig > 0.0:
                fcr_throughput_coef[t] = -(1.0 / eta_d_val) * sig
            elif sig < 0.0:
                fcr_throughput_coef[t] = -eta_c_val * sig
            fcr_activation_abs_coef[t] = abs(sig)

    # ============================================================
    # 3) Decision variables
    # ============================================================
    m.p_ch = pyo.Var(m.T, within=pyo.NonNegativeReals)   # import magnitude [kW]
    m.p_dis = pyo.Var(m.T, within=pyo.NonNegativeReals)  # export magnitude [kW]
    m.p_net = pyo.Var(m.T, within=pyo.Reals)             # net power (+import / -export) [kW]
    m.E = pyo.Var(m.S, within=pyo.Reals)                 # energy state [kWh] (band model)

    # FCR variables. Offering in a 1 MW integer grid (z_fcr) makes x_fcr take
    # values {0, 1000, 2000, ...} kW, which already encodes the 1 MW minimum
    # bid — no separate commitment binary is required.
    if use_fcr:
        m.x_fcr = pyo.Var(m.S_FCR, within=pyo.NonNegativeReals)  # offered kW
        m.z_fcr = pyo.Var(
            m.S_FCR,
            within=pyo.NonNegativeIntegers,
            bounds=lambda mdl, j: (0, int(fcr_cap_max_vals[j] // 1000.0)),
        )

    # Market positions (signed): +import (buy), -export (sell)
    m.p_market = pyo.Var(m.MARKETS, m.T, bounds=(-p_market_max_value, p_market_max_value))

    # Optional imbalance variables
    if allow_imbalance:
        m.p_imb_pos = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.p_imb_neg = pyo.Var(m.T, within=pyo.NonNegativeReals)

        m.price_imb_pos = pyo.Param(m.T, initialize=lambda mdl, t: float(imb_pos_arr[t]))
        m.price_imb_neg = pyo.Param(m.T, initialize=lambda mdl, t: float(imb_neg_arr[t]))

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

    # Forced FCR droop power at the grid connection (signed; positive = into depot).
    # Wired into market_balance below so droop activations must be settled against
    # scheduled trades / imbalance instead of bypassing through E.
    def _p_droop_rule(mdl, t):
        j = t_to_fcr_slot.get(t)
        if j is None:
            return 0.0
        # signal > 0 (low freq)  -> upward FCR   -> depot exports -> p_droop < 0
        # signal < 0 (high freq) -> downward FCR -> depot imports -> p_droop > 0
        return -mdl.fcr_droop_signal[t] * mdl.x_fcr[j]

    m.p_droop = pyo.Expression(m.T, rule=_p_droop_rule)

    def energy_state_rule(mdl, t):
        base = (
            mdl.E[t]
            + mdl.eta_c * mdl.p_ch[t] * mdl.dt
            - (1.0 / mdl.eta_d) * mdl.p_dis[t] * mdl.dt
        )
        j = t_to_fcr_slot.get(t)
        if j is not None and fcr_throughput_coef[t] != 0.0:
            base = base + fcr_throughput_coef[t] * mdl.x_fcr[j] * mdl.dt
        return mdl.E[t + 1] == base

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
    # Gate closure: closed (market, timestep) pairs are fixed to committed values.
    m.market_activity = pyo.Constraint(
        m.MARKET_CLOSED,
        rule=lambda mdl, mk, t: mdl.p_market[mk, t] == mdl.p_market_committed[mk, t],
    )

    # Balance markets to physical net power. The forced FCR droop power adds to
    # the grid flow that must be settled by scheduled trades and imbalance, so
    # droop cannot bypass markets via E.
    def balance_rule(mdl, t):
        base = pyo.quicksum(mdl.p_market[mk, t] for mk in mdl.MARKETS) + mdl.p_droop[t]
        if allow_imbalance:
            return base + mdl.p_imb_pos[t] - mdl.p_imb_neg[t] == mdl.p_net[t]
        return base == mdl.p_net[t]

    m.market_balance = pyo.Constraint(m.T, rule=balance_rule)

    # ============================================================
    # FCR constraints
    # ============================================================
    if use_fcr:
        # 1 MW resolution: x_fcr = 1000 * z_fcr (=> implicit 1 MW minimum bid)
        m.fcr_resolution = pyo.Constraint(m.S_FCR, rule=lambda mdl, j: mdl.x_fcr[j] == 1000.0 * mdl.z_fcr[j])

        # Available power headroom limit (also implied by the z_fcr bounds).
        m.fcr_cap_limit = pyo.Constraint(m.S_FCR, rule=lambda mdl, j: mdl.x_fcr[j] <= mdl.fcr_cap_max_kw[j])

        # Gate closure: once a slot's gate is closed, fix the offer to the committed
        # value. fcr_cap_max_kw is a sufficient bound on |x_fcr - committed| so it
        # serves as a tight Big-M while the gate is open.
        m.fcr_gate_ub = pyo.Constraint(
            m.S_FCR,
            rule=lambda mdl, j: mdl.x_fcr[j] <= mdl.x_fcr_committed[j] + fcr_cap_max_vals[j] * mdl.fcr_gate_open[j],
        )
        m.fcr_gate_lb = pyo.Constraint(
            m.S_FCR,
            rule=lambda mdl, j: mdl.x_fcr[j] >= mdl.x_fcr_committed[j] - fcr_cap_max_vals[j] * mdl.fcr_gate_open[j],
        )

        # Reserve symmetric power headroom for a full activation in every covered step.
        # When disabled, rely on the droop-signal-based p_droop already wired into the
        # market balance instead of a worst-case full-activation reservation.
        if fcr_enforce_power_headroom:
            m.fcr_headroom_up = pyo.Constraint(
                m.FCR_JT, rule=lambda mdl, j, t: mdl.p_net[t] + mdl.x_fcr[j] <= mdl.P_upper[t]
            )
            m.fcr_headroom_dn = pyo.Constraint(
                m.FCR_JT, rule=lambda mdl, j, t: mdl.p_net[t] - mdl.x_fcr[j] >= mdl.P_lower[t]
            )

        # Reserve enough energy / energy-headroom for a sustained activation.
        # When fcr_energy_req_hours is None or 0, skip the sustained-activation
        # reservation entirely and rely on the droop-based energy throughput
        # (already folded into the energy state) plus the regular E band.
        if fcr_energy_req_hours:
            def fcr_energy_headroom_up_rule(mdl, j, t):
                need_kwh = (mdl.x_fcr[j] * mdl.fcr_energy_req_hours) / mdl.eta_d
                return mdl.E[t] - need_kwh >= mdl.E_lower[t]

            def fcr_energy_headroom_dn_rule(mdl, j, t):
                incoming_kwh = mdl.x_fcr[j] * mdl.fcr_energy_req_hours * mdl.eta_c
                return mdl.E[t] + incoming_kwh <= mdl.E_upper[t]

            m.fcr_energy_cap_up = pyo.Constraint(m.FCR_JT, rule=fcr_energy_headroom_up_rule)
            m.fcr_energy_cap_dn = pyo.Constraint(m.FCR_JT, rule=fcr_energy_headroom_dn_rule)

    # ------------------------------------------------------------
    # Virtual arbitrage handling:
    # - LP mode: allow offsetting between markets, track absolute volume for fees
    # - MILP mode: prevent simultaneous import/export using binary mode and Big-M
    # ------------------------------------------------------------
    if virtual_arbitrage:
        if any_fee:
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
            # No transaction fees -> no need to linearize the absolute volume.
            m.p_market_vol = pyo.Expression(m.MARKETS, m.T, rule=lambda mdl, mk, t: 0.0)

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
            return pyo.quicksum(mdl.p_market_neg[mk, t] for mk in mdl.MARKETS)

        def total_import(mdl, t):
            return pyo.quicksum(mdl.p_market_pos[mk, t] for mk in mdl.MARKETS)

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
        dt = mdl.dt

        # Market cashflow (EUR)
        energy_cashflow = pyo.quicksum(
            -mdl.price[mk, t] * mdl.p_market[mk, t] * dt
            for mk in mdl.MARKETS for t in mdl.T
        )

        # Transaction fees on absolute volume (EUR)
        fee_cost = 0.0
        if any_fee:
            fee_cost = pyo.quicksum(
                mdl.fee[mk] * mdl.p_market_vol[mk, t] * dt
                for mk in mdl.MARKETS for t in mdl.T
            )

        # Cycling / degradation cost on throughput (EUR). FCR activations also
        # cycle the battery — debit them at |p_droop[t]| = |signal[t]| * x_fcr[j].
        deg_cost = mdl.c_deg * pyo.quicksum((mdl.p_ch[t] + mdl.p_dis[t]) * dt for t in mdl.T)
        if use_fcr:
            deg_cost = deg_cost + mdl.c_deg * pyo.quicksum(
                fcr_activation_abs_coef[t] * mdl.x_fcr[t_to_fcr_slot[t]] * dt
                for t in t_to_fcr_slot
                if fcr_activation_abs_coef[t] > 0.0
            )

        # Optional imbalance cashflow and volume penalty
        imb_cash = 0.0
        imb_vol_pen = 0.0
        if allow_imbalance:
            imb_cash = pyo.quicksum(
                (-mdl.price_imb_pos[t] * mdl.p_imb_pos[t] + mdl.price_imb_neg[t] * mdl.p_imb_neg[t]) * dt
                for t in mdl.T
            )
            imb_vol_pen = mdl.c_imb_vol * pyo.quicksum(
                (mdl.p_imb_pos[t] + mdl.p_imb_neg[t]) * dt for t in mdl.T
            )

        # Soft terminal objective (weight set by MPC)
        term_penalty = mdl.w_term * mdl.e_term_dev

        fcr_revenue = 0.0
        if use_fcr:
            fcr_revenue = pyo.quicksum(
                (mdl.fcr_price_param[j] / 1000.0) * mdl.x_fcr[j] * (mdl.fcr_slot_hours[j] / 4.0)
                for j in mdl.S_FCR
            )

        return energy_cashflow + fcr_revenue + imb_cash - fee_cost - deg_cost - imb_vol_pen - term_penalty

    m.obj = pyo.Objective(rule=obj_expr, sense=pyo.maximize)

    return m
