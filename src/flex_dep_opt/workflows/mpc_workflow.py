import logging
from pathlib import Path

from flex_dep_opt.market.fcr import get_fcr_prices
import pandas as pd
from tqdm.auto import tqdm
import pyomo.environ as pyo

from datetime import datetime
from zoneinfo import ZoneInfo

from flex_dep_opt.domain.depot import Depot
from flex_dep_opt.io.prices_io import build_prices_from_settings, build_fees_from_settings
from flex_dep_opt.io.flexibility_io import (
    read_flexibility_bounds_csv,
    align_and_validate_flexibility_bounds,
)
from flex_dep_opt.io.results_io import save_dispatch_to_csv, save_table_to_csv, save_settings_yaml_file, make_run_dir, write_latest_run_pointer, save_run_info_txt
from flex_dep_opt.opt.model import flexibility_commercialization
from flex_dep_opt.opt.solve import solve_model, extract_dispatch
from flex_dep_opt.market.trading_rules import (
    build_market_activity_mask_for_time,
    gate_closure_timestamp,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("gurobipy").setLevel(logging.WARNING)
logging.getLogger("pyomo").setLevel(logging.WARNING)
logging.getLogger("pyomo.core").setLevel(logging.ERROR)

def _fcr_gate_closure_timestamp(slot_start: pd.Timestamp) -> pd.Timestamp:
    # d-1 08:00 CET gate closure for slot
    d_minus_1 = slot_start.normalize() - pd.Timedelta(days=1)
    return d_minus_1.replace(hour=8, tzinfo=ZoneInfo("Europe/Berlin"))


def _extract_fcr_from_model(model, window_idx: pd.DatetimeIndex) -> pd.Series:
    result = pd.Series(0.0, index=window_idx, name="x_fcr_kw")

    if not hasattr(model, "S_FCR"):
        return result

    if not hasattr(model, "_fcr_slot_starts"):
        return result

    fcr_slot_starts = model._fcr_slot_starts
    for j in model.S_FCR:
        slot_start = fcr_slot_starts[j]
        slot_end = slot_start + pd.Timedelta(hours=4)
        committed_kw = pyo.value(model.x_fcr[j])
        mask = (window_idx >= slot_start) & (window_idx < slot_end)
        result.loc[mask] = committed_kw

    return result


def run_mpc(cfg: dict, config_path: str | Path | None = None) -> None:
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
    sim_cfg = cfg["simulation"]
    opt_cfg = cfg["optimization"]
    mpc_cfg = opt_cfg["mpc"]
    flex_cfg = opt_cfg.get("flexibility", {})
    dep_cfg = opt_cfg["depot"]

    imb_cfg = opt_cfg.get("imbalance", {})
    imb_enabled = bool(imb_cfg.get("enabled", False))

    terminal_enabled = bool(mpc_cfg.get("terminal_condition", False))
    terminal_weight = float(mpc_cfg.get("terminal_weight_eur_per_kwh", 50.0))

    # ============================================================
    # 2) Load flexibility bounds (flex bands)
    # ============================================================
    flexibility_bounds_full = read_flexibility_bounds_csv(flex_cfg["bounds_file"])

    fcr_prices_full = get_fcr_prices()

    # ============================================================
    # 3) Time settings
    # ============================================================
    step_hours = float(sim_cfg["timestep_hours"])
    da_horizon_hours = float(mpc_cfg["da_horizon_hours"])
    id_horizon_hours = float(mpc_cfg["id_horizon_hours"])
    da_horizon_steps = int(da_horizon_hours / step_hours)

    # ============================================================
    # 4) Load prices, fees and simulation time index
    # ============================================================
    prices_by_market = build_prices_from_settings(cfg)
    fees_by_market = build_fees_from_settings(cfg)

    start = pd.to_datetime(sim_cfg["start"]).tz_localize("Europe/Berlin")
    end = pd.to_datetime(sim_cfg["end"]).tz_localize("Europe/Berlin")

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

    # ============================================================
    # 5) Model options
    # ============================================================
    virtual_arbitrage = opt_cfg.get("virtual_arbitrage", False)

    cyc_cfg = flex_cfg.get("cycle_regularization", {})
    c_cyc = float(cyc_cfg["cost_eur_per_kwh_throughput"]) if cyc_cfg.get("enabled", False) else 0.0

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
            window_end_ts = window_idx[-1] + pd.Timedelta(hours=4)  # last slot may start up to 4h before window end

            window_fcr_prices = fcr_prices_full.loc[
                (fcr_prices_full.index >= window_start) &
                (fcr_prices_full.index < window_end_ts)
            ]

            # gate-open mask for slots in this window
            # a slot is still open if its D-1 08:00 gate closure is in the future
            fcr_gate_open_window: dict[pd.Timestamp, bool] = {}
            for slot in window_fcr_prices.index:
                gate_ts = _fcr_gate_closure_timestamp(slot)
                fcr_gate_open_window[slot] = current_time < gate_ts

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
                id_mask = decision_masks["ID"].copy()
                id_mask[window_idx > id_horizon_end] = False
                decision_masks["ID"] = id_mask

            # --------------------------------------------------------
            # 9.4) Build and parameterize optimization model
            # --------------------------------------------------------
            def _build_model(allow_imbalance: bool, imb_penalty: float) -> object:
                _m = flexibility_commercialization(
                    depot=Depot(**dep_cfg),
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
                )

                if hasattr(_m, "S_FCR"):
                    _m._fcr_slot_starts = list(window_fcr_prices.index)
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
                solve_model(model, solver_name=sim_cfg["solver"])
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
                        imb_penalty=float(imb_cfg.get("imbalance_volume_penalty_eur_per_kwh", 1000.0)),
                    )

                    try:
                        _set_E0(model2)
                        _set_terminal_terms(model2)
                        solve_model(model2, solver_name=sim_cfg["solver"])
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
            fcr_kw_window = _extract_fcr_from_model(model, window_idx)
            dispatch_window["x_fcr_kw"] = fcr_kw_window
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
                    if opt_cfg["trading"]["mode"] == "none" and tau == window_idx[0]:
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
                    gate_ts = _fcr_gate_closure_timestamp(slot)
                    slot_was_open = current_time < gate_ts
                    slot_now_closed = next_time >= gate_ts

                    if slot_was_open and slot_now_closed:
                        committed_val = pyo.value(model.x_fcr[j])
                        committed_fcr_slots[slot] = committed_val
                        fcr_price_val = float(window_fcr_prices.loc[slot])

                        fcr_commit_rows.append({
                            "slot_start": slot,
                            "gate_closure_time": gate_ts,
                            "committed_at": current_time,
                            "x_fcr_kw": committed_val,
                            "x_fcr_mw": committed_val / 1000.0,
                            "fcr_price": fcr_price_val,
                            "fcr_revenue_eur": (committed_val / 1000.0) * fcr_price_val,
                        })

                        logger.info(
                            f"FCR slot committed: {slot} - {committed_val:.1f} kW "
                            f"@ {fcr_price_val:.2f} €/MW"
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
        name = str(sim_cfg.get("name", "")).strip()
        if not name:
            raise ValueError("settings.yaml: simulation.name must be set (e.g. 'illustrative_example').")

        # --- Create per-run output directory (Option A: name + timestamp) ---
        run_dir = make_run_dir("results", name, tz="Europe/Berlin")
        write_latest_run_pointer(run_dir, results_root="results")

        # --- Save the exact YAML that was passed via CLI / batch (1:1 copy) ---
        if config_path is not None:
            save_settings_yaml_file(config_path, run_dir)

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
            config_path=config_path,
            solver_name=str(sim_cfg.get("solver", "")),
            start_time=run_started_at,
            end_time=run_finished_at,
            tz="Europe/Berlin",
        )

        print(f"MPC finished → Postprocessing starts")

