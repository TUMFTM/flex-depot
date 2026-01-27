import logging
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from flex_dep_opt.domain.depot import Depot
from flex_dep_opt.io.prices_io import build_prices_from_settings, build_fees_from_settings
from flex_dep_opt.io.flexibility_io import (
    read_flexibility_bounds_csv,
    align_and_validate_flexibility_bounds,
)
from flex_dep_opt.io.results_io import save_dispatch_to_csv, save_summary_to_csv
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


def run_mpc(cfg: dict) -> None:
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

    # ============================================================
    # 9) Rolling-horizon MPC loop
    # ============================================================
    try:
        pbar = tqdm(total=len(full_index), desc="MPC", unit="step", dynamic_ncols=True)

        for i in range(len(full_index)):
            current_time = full_index[i]
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
            model = flexibility_commercialization(
                depot=Depot(**dep_cfg),
                prices_by_market=window_prices,
                fee_eur_per_kwh_by_market=fees_by_market,
                timestep_hours=step_hours,
                virtual_arbitrage=virtual_arbitrage,
                cycling_cost_eur_per_kwh=c_cyc,
                market_activity_mask=decision_masks,
                committed_positions={mk: committed_positions[mk].loc[window_idx] for mk in committed_positions},
                flexibility_bounds=window_flexibility_bounds,
                allow_imbalance=False,
                imbalance_volume_penalty_eur_per_kwh=0.0,
            )

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
                solve_model(model)
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

                    model2 = flexibility_commercialization(
                        depot=Depot(**dep_cfg),
                        prices_by_market=window_prices,
                        fee_eur_per_kwh_by_market=fees_by_market,
                        timestep_hours=step_hours,
                        virtual_arbitrage=virtual_arbitrage,
                        cycling_cost_eur_per_kwh=c_cyc,
                        market_activity_mask=decision_masks,
                        committed_positions={mk: committed_positions[mk].loc[window_idx] for mk in committed_positions},
                        flexibility_bounds=window_flexibility_bounds,
                        allow_imbalance=True,
                        imbalance_prices_pos=window_imb_pos,
                        imbalance_prices_neg=window_imb_neg,
                        imbalance_volume_penalty_eur_per_kwh=float(
                            imb_cfg.get("imbalance_volume_penalty_eur_per_kwh", 1000.0)
                        ),
                    )

                    try:
                        _set_E0(model2)
                        _set_terminal_terms(model2)
                        solve_model(model2)
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
            next_time = current_time + pd.Timedelta(hours=step_hours)

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
        if rows:
            result = pd.DataFrame(rows)
            result.index.name = "time"

            out_dispatch = Path(sim_cfg["out_dispatch"]).with_suffix(".csv")
            out_dispatch.parent.mkdir(parents=True, exist_ok=True)
            save_dispatch_to_csv(result.reset_index(), out_dispatch)

        if commit_rows:
            commit_df = pd.DataFrame(commit_rows)

            out_commit = Path(sim_cfg["out_commit"]).with_suffix(".csv")
            out_commit.parent.mkdir(parents=True, exist_ok=True)
            save_dispatch_to_csv(commit_df.sort_values(["delivery_time", "current_time"]), out_commit)

        logger.info("MPC finished → results/dispatch.csv & results/commit.csv")

