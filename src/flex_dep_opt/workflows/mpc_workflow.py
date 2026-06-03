import logging
import random
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from tqdm.auto import tqdm

from flex_dep_opt.config.settings import Settings
from flex_dep_opt.domain.depot import Depot
from flex_dep_opt.io.flexibility_io import (
    align_and_validate_flexibility_bounds,
    read_flexibility_bounds_csv,
)
from flex_dep_opt.io.prices_io import build_fees_from_settings, build_prices_from_settings
from flex_dep_opt.io.results_io import (
    make_run_dir,
    save_dispatch_to_csv,
    save_run_info_txt,
    save_table_to_csv,
    write_latest_run_pointer,
)
from flex_dep_opt.market.fcr import (
    FCR_DROOP_COL,
    fcr_gate_closure_timestamp,
    get_fcr_frequency_data,
    get_fcr_prices,
)
from flex_dep_opt.market.trading_rules import (
    build_market_activity_mask_for_time,
    gate_closure_timestamp,
)
from flex_dep_opt.opt.model import flexibility_commercialization
from flex_dep_opt.opt.solve import extract_dispatch, solve_model

logger = logging.getLogger(__name__)


class _TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


_root_logger = logging.getLogger()
if not any(isinstance(h, _TqdmLoggingHandler) for h in _root_logger.handlers):
    for h in list(_root_logger.handlers):
        _root_logger.removeHandler(h)
    _handler = _TqdmLoggingHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    _root_logger.addHandler(_handler)
    _root_logger.setLevel(logging.INFO)
logging.getLogger("gurobipy").setLevel(logging.WARNING)
logging.getLogger("pyomo").setLevel(logging.WARNING)
logging.getLogger("pyomo.core").setLevel(logging.ERROR)

def run_mpc(settings: Settings) -> None:
    """
    Rolling-horizon Model Predictive Control (MPC).

    At each simulation step:
      - Optimize a forward-looking time window (DA horizon)
      - Apply only the first decision (receding horizon)
      - Roll the energy state forward
      - Respect market gate closures via activity masks
      - Permanently commit market positions once gate closure is passed

    Notes
    -----
    Gate-closure times are computed in the market layer via `gate_closure_timestamp()`.
    This ensures mask enforcement and commit logging use the exact same rules.
    """
    run_started_at = datetime.now(ZoneInfo("Europe/Berlin"))

    # ============================================================
    # 1) Read configuration blocks
    # ============================================================
    sim_cfg = settings.simulation
    opt_cfg = settings.optimization
    mpc_cfg = opt_cfg.mpc
    flex_cfg = opt_cfg.flexibility
    dep_cfg = opt_cfg.depot

    imb_cfg = opt_cfg.imbalance
    imb_enabled = imb_cfg.enabled

    terminal_enabled = bool(mpc_cfg.terminal_condition)
    terminal_weight = float(mpc_cfg.terminal_weight_eur_per_kwh)

    # ============================================================
    # 2) Load flexibility bounds (flex bands)
    # ============================================================
    flexibility_bounds_full = read_flexibility_bounds_csv(flex_cfg.bounds_file)

    fcr_prices_full = (
        get_fcr_prices(opt_cfg.trading.fcr.prices_source)
        if opt_cfg.trading.fcr.enabled
        else pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="Europe/Berlin"))
    )
    fcr_acceptance_rate = opt_cfg.trading.fcr.acceptance_rate
    fcr_acceptance_seed = opt_cfg.trading.fcr.acceptance_seed
    if fcr_acceptance_seed is not None:
        random.seed(fcr_acceptance_seed)

    fcr_freq_source = opt_cfg.trading.fcr.frequency_source
    fcr_frequency_column = opt_cfg.trading.fcr.frequency_column
    fcr_frequency_data_full: pd.DataFrame | None = None
    if opt_cfg.trading.fcr.enabled and fcr_freq_source:
        fcr_frequency_data_full = get_fcr_frequency_data(
            fcr_freq_source, column=fcr_frequency_column
        )

    frequency_nominal_hz = opt_cfg.trading.fcr.frequency_nominal_hz
    frequency_deadband_hz = opt_cfg.trading.fcr.deadband_hz
    frequency_full_activation_hz = opt_cfg.trading.fcr.full_activation_hz

    fcr_product_hours = float(opt_cfg.trading.fcr.product_hours)
    fcr_bid_block_kw = float(opt_cfg.trading.fcr.bid_block_mw) * 1000.0

    fcr_energy_reserve_kwh_per_kw = float(opt_cfg.trading.fcr.energy_reserve_minutes) / 60.0
    fcr_reserve_penalty = float(opt_cfg.trading.fcr.reserve_penalty_eur_per_kwh)
    fcr_balance_penalty = float(opt_cfg.trading.fcr.balance_penalty_eur_per_kwh)

    def _fcr_gate_ts(slot_start: pd.Timestamp) -> pd.Timestamp:
        return fcr_gate_closure_timestamp(
            slot_start,
            hour=opt_cfg.trading.fcr.gate_closure_hour,
            closes_previous_day=opt_cfg.trading.fcr.gate_closure_closes_previous_day,
            timezone=opt_cfg.trading.fcr.gate_closure_timezone,
        )

    # ============================================================
    # 3) Time settings
    # ============================================================
    step_hours = float(sim_cfg.timestep_hours)
    da_horizon_hours = float(mpc_cfg.da_horizon_hours)
    id_horizon_hours = float(mpc_cfg.id_horizon_hours)
    fcr_price_horizon_hours = float(mpc_cfg.fcr_price_horizon_hours)
    fcr_frequency_horizon_minutes = float(mpc_cfg.fcr_frequency_horizon_minutes)
    da_horizon_steps = int(da_horizon_hours / step_hours)

    # ============================================================
    # 4) Load prices, fees and simulation time index
    # ============================================================
    prices_by_market = build_prices_from_settings(settings)
    fees_by_market = build_fees_from_settings(settings)

    start = pd.to_datetime(sim_cfg.start).tz_localize("Europe/Berlin")
    end = pd.to_datetime(sim_cfg.end).tz_localize("Europe/Berlin")

    for mk in prices_by_market:
        prices_by_market[mk] = prices_by_market[mk].loc[start:end]

    full_index = prices_by_market[next(iter(prices_by_market))].index

    # Full state index (N+1): add terminal state timestamp
    dt = pd.Timedelta(hours=step_hours)
    full_state_index = full_index.append(pd.DatetimeIndex([full_index[-1] + dt]))

    if flexibility_bounds_full is not None:
        flexibility_bounds_full = flexibility_bounds_full.loc[full_state_index]

    imb_pos_full = prices_by_market.pop("IMB_POS", None)
    imb_neg_full = prices_by_market.pop("IMB_NEG", None)

    # slice fcr prices to simulation window
    fcr_prices_full = fcr_prices_full.loc[
        (fcr_prices_full.index >= start) & (fcr_prices_full.index <= end)
    ]

    all_fcr_slot_starts = list(fcr_prices_full.index)

    # Slot-local cap_max and slot_hours, computed against the *full* simulation
    # horizon (not the rolling window) so:
    #   - cap_max[j] reflects headroom over the whole 4 h slot (avoids
    #     bidding more MW than later steps can physically deliver), and
    #   - slot_hours[j] = 4 for slots fully covered by the sim, so the
    #     optimizer does not under-bid partial-overlap slots that the next
    #     rolling window will finish covering.
    fcr_cap_max_by_slot: dict[pd.Timestamp, float] = {}
    fcr_slot_hours_by_slot: dict[pd.Timestamp, float] = {}
    if not fcr_prices_full.empty and flexibility_bounds_full is not None:
        P_up_full = flexibility_bounds_full["Power_upper_kW"]
        P_lo_full = flexibility_bounds_full["Power_lower_kW"]
        for slot_start in all_fcr_slot_starts:
            slot_end = slot_start + pd.Timedelta(hours=fcr_product_hours)
            slot_mask = (full_index >= slot_start) & (full_index < slot_end)
            n_covered = int(slot_mask.sum())
            if n_covered == 0:
                fcr_cap_max_by_slot[slot_start] = 0.0
                fcr_slot_hours_by_slot[slot_start] = 0.0
                continue
            covered_idx = full_index[slot_mask]
            P_up = P_up_full.loc[covered_idx].to_numpy(dtype=float)
            P_lo = P_lo_full.loc[covered_idx].to_numpy(dtype=float)
            cap = float(np.minimum(P_up, -P_lo).min())
            fcr_cap_max_by_slot[slot_start] = max(cap, 0.0)
            fcr_slot_hours_by_slot[slot_start] = n_covered * step_hours

    # ============================================================
    # 5) Model options
    # ============================================================
    virtual_arbitrage = opt_cfg.virtual_arbitrage

    cyc_cfg = flex_cfg.cycle_regularization
    c_cyc = cyc_cfg.cost_eur_per_kwh_throughput if cyc_cfg.enabled else 0.0

    # ============================================================
    # 6) Initialize committed market positions
    # ============================================================
    committed_positions = {mk: pd.Series(0.0, index=full_index) for mk in prices_by_market.keys()}

    committed_fcr_slots: dict[pd.Timestamp, float] = {
        slot: 0.0 for slot in all_fcr_slot_starts
    }

    # ============================================================
    # 7) Initialize energy state (band-consistent)
    # ============================================================
    if flexibility_bounds_full is not None:
        E_state = 0.5 * (
            flexibility_bounds_full["Capacity_lower_kWh"].iloc[0]
            + flexibility_bounds_full["Capacity_upper_kWh"].iloc[0])
    else:
        E_state = 0.0

    # ============================================================
    # 8) Result containers
    # ============================================================
    rows = []
    commit_rows = []
    fcr_commit_rows = []

    # ============================================================
    # 9) Rolling-horizon MPC loop
    # ============================================================
    try:
        pbar = tqdm(total=len(full_index), desc="MPC", unit="step", dynamic_ncols=True)

        for i in range(len(full_index)):
            current_time = full_index[i]
            next_time = current_time + pd.Timedelta(hours=step_hours)
            pbar.set_postfix(time=str(current_time), E=f"{E_state:.1f} kWh")

            # --------------------------------------------------------
            # 9.1) Define rolling optimization window
            # --------------------------------------------------------
            N_total = len(full_index)
            window_end = min(i + da_horizon_steps, N_total)   # exclusive end for decisions
            window_idx = full_index[i:window_end]             # decision timestamps
            window_state_idx = full_state_index[i:window_end + 1]  # state timestamps (decisions + 1)

            if len(window_idx) == 0:
                break

            # --------------------------------------------------------
            # 9.2) Slice prices and flexibility bounds
            # --------------------------------------------------------
            window_prices = {mk: prices_by_market[mk].loc[window_idx] for mk in prices_by_market}

            window_flexibility_bounds = (
                align_and_validate_flexibility_bounds(
                    bounds=flexibility_bounds_full,
                    time_index=window_state_idx,
                    expected_len=len(window_state_idx),)
                if flexibility_bounds_full is not None
                else None)

            window_imb_pos = (imb_pos_full.loc[window_idx] if (imb_enabled and imb_pos_full is not None) else None)
            window_imb_neg = (imb_neg_full.loc[window_idx] if (imb_enabled and imb_neg_full is not None) else None)

            # fcr gates
            window_start = window_idx[0]
            window_end_ts = window_idx[-1] + pd.Timedelta(hours=fcr_product_hours)  # last slot may start up to one product length before window end

            # Include any slot that *overlaps* the window, not only those starting
            # inside it. An already-running slot (gate closed, bid committed) still
            # needs its droop activation applied to every step it covers — otherwise
            # only the slot-start step ever sees nonzero p_droop.
            slot_duration = pd.Timedelta(hours=fcr_product_hours)
            window_fcr_prices = fcr_prices_full.loc[
                (fcr_prices_full.index + slot_duration > window_start) &
                (fcr_prices_full.index < window_end_ts)
            ]

            # gate-open mask for slots in this window
            # a slot is biddable if (a) its D-1 08:00 gate closure is still in
            # the future and (b) it starts within the FCR price foresight
            # horizon. Slots beyond the horizon are pinned to their committed
            # value (0 if never committed), i.e. not biddable yet.
            fcr_price_horizon_end = current_time + pd.Timedelta(hours=fcr_price_horizon_hours)
            fcr_gate_open_window: dict[pd.Timestamp, bool] = {}
            for slot in window_fcr_prices.index:
                gate_ts = _fcr_gate_ts(slot)
                within_horizon = slot <= fcr_price_horizon_end
                fcr_gate_open_window[slot] = (current_time < gate_ts) and bool(within_horizon)

            # --------------------------------------------------------
            # 9.3) Build market activity masks (gate-closures enforced here)
            # --------------------------------------------------------
            trading_masks = build_market_activity_mask_for_time(
                current_time=current_time,
                delivery_times=window_idx,
                optimization_cfg=opt_cfg,
            )

            # --------------------------------------------------------
            # 9.3b) Build DECISION masks = trading masks + price foresight
            # --------------------------------------------------------
            decision_masks = {mk: s.copy() for mk, s in trading_masks.items()}

            id_horizon_end = current_time + pd.Timedelta(hours=id_horizon_hours)
            if "ID" in decision_masks:
                decision_masks["ID"][window_idx > id_horizon_end] = False

            # frequency data covering this window
            if fcr_frequency_data_full is not None:
                window_freq_data = fcr_frequency_data_full.loc[
                    (fcr_frequency_data_full.index >= window_idx[0])
                    & (fcr_frequency_data_full.index <= window_idx[-1] + pd.Timedelta(hours=1))
                ].copy()
                # FCR frequency foresight: grid frequency is unpredictable, so
                # only the near term is treated as known. Beyond the frequency
                # horizon the signal is reset to a zero-droop value so the
                # optimizer plans for no FCR activation there. The current step
                # (at current_time) always keeps its real value. For a droop
                # column zero droop is 0.0; for a frequency column it is nominal.
                fcr_freq_horizon_end = current_time + pd.Timedelta(
                    minutes=fcr_frequency_horizon_minutes
                )
                beyond_horizon = window_freq_data.index > fcr_freq_horizon_end
                zero_droop_value = (
                    0.0 if fcr_frequency_column == FCR_DROOP_COL else frequency_nominal_hz
                )
                window_freq_data.loc[beyond_horizon, fcr_frequency_column] = zero_droop_value
            else:
                window_freq_data = None

            # --------------------------------------------------------
            # 9.4) Build and parameterize optimization model
            # --------------------------------------------------------
            def _build_model(allow_imbalance: bool, imb_penalty: float) -> object:
                _m = flexibility_commercialization(
                    depot=Depot(dep_cfg.eta_grid2depot, dep_cfg.eta_depot2grid, dep_cfg.grid_connection_limit),
                    prices_by_market=window_prices,
                    fee_eur_per_kwh_by_market=fees_by_market,
                    timestep_hours=step_hours,
                    virtual_arbitrage=virtual_arbitrage,
                    cycling_cost_eur_per_kwh=c_cyc,
                    market_activity_mask=decision_masks,
                    committed_positions={
                        mk: committed_positions[mk].loc[window_idx]
                        for mk in committed_positions
                    },
                    flexibility_bounds=window_flexibility_bounds,
                    allow_imbalance=allow_imbalance,
                    imbalance_prices_pos=window_imb_pos,
                    imbalance_prices_neg=window_imb_neg,
                    imbalance_volume_penalty_eur_per_kwh=imb_penalty,
                    fcr_prices=window_fcr_prices if not window_fcr_prices.empty else None,
                    fcr_frequency_data=window_freq_data,
                    fcr_frequency_column=fcr_frequency_column,
                    frequency_nominal_hz=frequency_nominal_hz,
                    frequency_deadband_hz=frequency_deadband_hz,
                    frequency_full_activation_hz=frequency_full_activation_hz,
                    fcr_product_hours=fcr_product_hours,
                    fcr_bid_block_kw=fcr_bid_block_kw,
                    fcr_cap_max_by_slot=fcr_cap_max_by_slot or None,
                    fcr_slot_hours_by_slot=fcr_slot_hours_by_slot or None,
                    fcr_energy_reserve_kwh_per_kw=fcr_energy_reserve_kwh_per_kw,
                    fcr_reserve_penalty_eur_per_kwh=fcr_reserve_penalty,
                    fcr_balance_penalty_eur_per_kwh=fcr_balance_penalty,
                )

                if hasattr(_m, "S_FCR"):
                    # _m._fcr_slot_starts is set inside the model (only the slots
                    # that actually overlap this window), so S_FCR indices map to it.
                    for j in _m.S_FCR:
                        slot = _m._fcr_slot_starts[j]
                        is_open = fcr_gate_open_window.get(slot, False)
                        committed_val = committed_fcr_slots.get(slot, 0.0)
                        _m.fcr_gate_open[j].set_value(1 if is_open else 0)
                        if not is_open:
                            _m.x_fcr_committed[j].set_value(committed_val)

                return _m

            model = _build_model(allow_imbalance=False, imb_penalty=0.0)

            # Helper to set E0 consistently on a given model
            def _set_E0(_model):
                if window_flexibility_bounds is not None:
                    lb0 = float(window_flexibility_bounds["Capacity_lower_kWh"].iloc[0])
                    ub0 = float(window_flexibility_bounds["Capacity_upper_kWh"].iloc[0])
                    E0 = float(E_state if i > 0 else 0.5 * (lb0 + ub0))
                    if not (lb0 - 1e-6 <= E0 <= ub0 + 1e-6):
                        logger.warning(
                            f"E0 out of bounds at {current_time}: E0={E0}, lb={lb0}, ub={ub0} (clamping)"
                        )
                    _model.E0.set_value(float(min(max(E0, lb0), ub0)))
                else:
                    _model.E0.set_value(float(E_state))

            # Helper to set terminal condition
            def _set_terminal_terms(_model):
                _model.w_term.set_value(0.0)
                _model.energy_term_hard.deactivate()

                if not terminal_enabled or window_flexibility_bounds is None:
                    return

                goal_time = full_state_index[-1]
                if goal_time not in window_state_idx:
                    return

                lb = float(window_flexibility_bounds.loc[goal_time, "Capacity_lower_kWh"])
                ub = float(window_flexibility_bounds.loc[goal_time, "Capacity_upper_kWh"])
                _model.Eterm.set_value(0.5 * (lb + ub))

                remaining_steps = N_total - i
                frac = min(1.0, da_horizon_steps / max(1, remaining_steps))
                _model.w_term.set_value(terminal_weight * frac)

                if remaining_steps == 1:
                    _model.energy_term_hard.activate()

            # --------------------------------------------------------
            # 9.5) Solve optimization problem (Two-pass)
            # --------------------------------------------------------
            solved = False
            used_rebap = False

            try:
                _set_E0(model)
                _set_terminal_terms(model)
                solve_model(model, solver_name=sim_cfg.solver)
                solved = True
                used_rebap = False

            except RuntimeError as e1:
                if not imb_enabled:
                    tqdm.write("ERROR - PASS1 infeasible and imbalance disabled → abort")
                    logger.error(f"PASS1 infeasible at {current_time} and imbalance disabled. Details: {e1}")
                    solved = False
                else:
                    tqdm.write("PASS1 infeasible → Imbalance activated (PASS2)")
                    logger.info(f"PASS1 infeasible at {current_time} → trying PASS2 (imbalance). Details: {e1}")

                    model2 = _build_model(
                        allow_imbalance=True,
                        imb_penalty=imb_cfg.imbalance_volume_penalty_eur_per_kwh,
                    )

                    try:
                        _set_E0(model2)
                        _set_terminal_terms(model2)
                        solve_model(model2, solver_name=sim_cfg.solver)
                        solved = True
                        used_rebap = True
                        model = model2
                    except RuntimeError as e2:
                        tqdm.write("ERROR - PASS2 also infeasible → aborting")
                        logger.error(f"PASS2 also infeasible at {current_time}. Details: {e2}")
                        solved = False

            if not solved:
                logger.error(f"Both passes infeasible at current_time={current_time} → aborting MPC loop.")
                break

            # --------------------------------------------------------
            # 9.6) Extract first-step dispatch
            # --------------------------------------------------------
            dispatch_window = extract_dispatch(model, window_idx)
            first_row = dispatch_window.iloc[0].copy()
            first_row["used_rebap"] = used_rebap
            first_row.name = window_idx[0]
            rows.append(first_row)

            # --------------------------------------------------------
            # 9.7) Commit market positions at gate closure
            # --------------------------------------------------------
            trading_masks_next = build_market_activity_mask_for_time(
                current_time=next_time,
                delivery_times=window_idx,
                optimization_cfg=opt_cfg,
            )

            for mk in committed_positions:
                # Robustness: only process markets that exist in the masks
                if mk not in trading_masks or mk not in trading_masks_next:
                    continue

                p_col = f"p_{mk.lower()}_kw"
                if p_col not in dispatch_window.columns:
                    continue

                for tau in window_idx:
                    now_open = bool(trading_masks[mk].loc[tau])
                    next_open = bool(trading_masks_next[mk].loc[tau])

                    row = {
                        "market": mk,
                        "delivery_time": tau,
                        "current_time": current_time,
                        "next_time": next_time,
                        # Gate-closure timestamp from market layer (single source of truth)
                        "gate_closure_time": gate_closure_timestamp(mk, tau, opt_cfg),
                        "p_opt": float(dispatch_window.loc[tau, p_col]),
                        "committed_old": committed_positions[mk].loc[tau],
                        "committed_new": committed_positions[mk].loc[tau],
                        "commit_now": False,
                    }

                    # Mode "none": commit only the immediate first decision (as before)
                    if opt_cfg.trading.mode == "none" and tau == window_idx[0]:
                        committed_positions[mk].loc[tau] = row["p_opt"]
                        row["committed_new"] = row["p_opt"]
                        row["commit_now"] = True

                    # Realistic trading: commit exactly when the gate closes between now and next step
                    elif now_open and not next_open:
                        committed_positions[mk].loc[tau] = row["p_opt"]
                        row["committed_new"] = row["p_opt"]
                        row["commit_now"] = True

                    commit_rows.append(row)

            # commit fcr slots at their gate closure
            if hasattr(model, "S_FCR"):
                for j in model.S_FCR:
                    slot = model._fcr_slot_starts[j]
                    gate_ts = _fcr_gate_ts(slot)
                    slot_was_open = current_time < gate_ts
                    slot_now_closed = next_time >= gate_ts

                    if slot_was_open and slot_now_closed:
                        bid_val = pyo.value(model.x_fcr[j])
                        accepted = bid_val > 0 and random.random() < fcr_acceptance_rate
                        committed_val = bid_val if accepted else 0.0
                        committed_fcr_slots[slot] = committed_val
                        fcr_price_val = float(window_fcr_prices.loc[slot])
                        # Hours of this 4 h slot covered by the sim horizon;
                        # revenue is prorated to match the optimizer's
                        # fcr_revenue term (price is for the full 4 h product).
                        slot_hours = float(fcr_slot_hours_by_slot.get(slot, 4.0))

                        fcr_commit_rows.append({
                            "slot_start": slot,
                            "gate_closure_time": gate_ts,
                            "committed_at": current_time,
                            "bid_kw": bid_val,
                            "x_fcr_kw": committed_val,
                            "x_fcr_mw": committed_val / 1000.0,
                            "accepted": accepted,
                            "fcr_price": fcr_price_val,
                            "slot_hours": slot_hours,
                            "fcr_revenue_eur": (committed_val / 1000.0) * fcr_price_val * (slot_hours / fcr_product_hours),
                        })

                        if bid_val > 0:
                            if accepted:
                                logger.info(
                                    f"[{current_time}] FCR slot accepted: {slot} - {committed_val:.1f} kW "
                                    f"@ {fcr_price_val:.2f} €/MW"
                                )
                            else:
                                logger.info(
                                    f"[{current_time}] FCR slot rejected: {slot} - bid {bid_val:.1f} kW not accepted"
                                )

            # --------------------------------------------------------
            # 9.8) Roll energy state forward
            # --------------------------------------------------------
            E_state = float(first_row["E_next_kWh"])
            pbar.update(1)

        pbar.close()

    # ============================================================
    # 10) Export results
    # ============================================================
    finally:
        # --- Read simulation "name" from settings ---
        name = sim_cfg.name.strip()
        if not name:
            raise ValueError("simulation.name must be set in the settings file (e.g. 'illustrative_example').")

        # --- Create per-run output directory (Option A: name + timestamp) ---
        run_dir = make_run_dir("results", name, tz="Europe/Berlin")
        write_latest_run_pointer(run_dir, results_root="results")

        # --- Save the exact YAML that was passed via CLI / batch (1:1 copy) ---
        settings.save_to_toml(run_dir / "settings.toml")

        # --- Output filenames inside run_dir ---
        out_dispatch = run_dir / "dispatch.csv"
        out_commit = run_dir / "commit.csv"
        out_fcr_commit = run_dir / "fcr_commit.csv"

        # --- Write dispatch ---
        if rows:
            result = pd.DataFrame(rows)
            result.index.name = "time"
            save_dispatch_to_csv(result, out_dispatch, include_time_column=True)

        # --- Write commit ---
        if commit_rows:
            commit_df = pd.DataFrame(commit_rows)
            commit_df = commit_df.sort_values(["delivery_time", "current_time"])
            save_table_to_csv(commit_df, out_commit)

        # fcr commits log
        if fcr_commit_rows:
            fcr_commit_df = pd.DataFrame(fcr_commit_rows).sort_values("slot_start")
            save_table_to_csv(fcr_commit_df, out_fcr_commit)

        run_finished_at = datetime.now(ZoneInfo("Europe/Berlin"))

        save_run_info_txt(
            run_dir=run_dir,
            simulation_name=name,
            config_path=settings.get_source_path(),
            solver_name=sim_cfg.solver,
            start_time=run_started_at,
            end_time=run_finished_at,
            tz="Europe/Berlin",
        )

        print("MPC finished → Postprocessing starts")

