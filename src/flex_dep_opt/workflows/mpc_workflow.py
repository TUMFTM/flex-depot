from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

from flex_dep_opt.domain.vehicle import Vehicle
from flex_dep_opt.domain.site import Site
from flex_dep_opt.io.prices_io import build_prices_from_settings
from flex_dep_opt.io.mobility_io import (
    read_mobility_bounds_csv,
    align_and_validate_mobility_bounds,
)
from flex_dep_opt.opt.model import fleet_commercialization
from flex_dep_opt.opt.solve import solve_model, extract_dispatch
from flex_dep_opt.market.trading_rules import build_market_activity_mask_for_time


def run_mpc(cfg: dict):
    """
    Rolling-horizon Model Predictive Control (MPC).

    At each simulation step:
      - Optimize a forward-looking time window (DA horizon)
      - Apply only the first decision
      - Roll the energy state forward
      - Respect real market gate closures via activity masks
      - Permanently commit market positions once gate closure is passed
    """

    # ============================================================
    # 1) Read configuration blocks
    # ============================================================
    sim_cfg = cfg["simulation"]
    opt_cfg = cfg["optimize"]
    opt_conf = cfg["optimization"]
    trading_cfg = opt_conf["trading"]
    mpc_cfg = opt_conf["mpc"]
    mob_cfg = opt_conf.get("mobility", {})

    # ============================================================
    # 2) Load mobility bounds (flex bands)
    # ============================================================
    mobility_bounds_full = None
    if mob_cfg.get("enabled", False):
        mobility_bounds_full = read_mobility_bounds_csv(mob_cfg["bounds_file"])

    # ============================================================
    # 3) Helper: compute market gate closure timestamp
    # ============================================================
    def compute_gate_closure(market: str, delivery_time: pd.Timestamp) -> pd.Timestamp:
        """
        Compute the gate-closure timestamp for a given market and delivery time.

        This must be consistent with build_market_activity_mask_for_time().
        """
        if market == "ID":
            id_cfg = trading_cfg.get("intraday", {})
            offset_min = int(id_cfg.get("offset_minutes_before_delivery", 30))
            return delivery_time - pd.Timedelta(minutes=offset_min)

        if market == "DA":
            da_cfg = trading_cfg.get("dayahead", {})
            gc_hour_str = da_cfg.get("gate_closure_hour", "12:00")
            closes_prev = da_cfg.get("closes_previous_day", True)

            try:
                hh, mm = map(int, gc_hour_str.split(":"))
            except Exception:
                hh, mm = 12, 0  # robust fallback

            base_date = (
                delivery_time.normalize() - pd.Timedelta(days=1)
                if closes_prev
                else delivery_time.normalize()
            )
            return base_date + pd.Timedelta(hours=hh, minutes=mm)

        raise ValueError(f"Unknown market: {market}")

    # ============================================================
    # 4) Time settings
    # ============================================================
    step_hours = float(sim_cfg["timestep_hours"])
    da_horizon_hours = float(mpc_cfg["da_horizon_hours"])
    id_horizon_hours = float(mpc_cfg["id_horizon_hours"])
    da_horizon_steps = int(da_horizon_hours / step_hours)

    # ============================================================
    # 5) Load prices and simulation time index
    # ============================================================
    prices_by_market = build_prices_from_settings(cfg)

    start = pd.to_datetime(sim_cfg["start"]).tz_localize("Europe/Berlin")
    end = pd.to_datetime(sim_cfg["end"]).tz_localize("Europe/Berlin")

    for mk in prices_by_market:
        prices_by_market[mk] = prices_by_market[mk].loc[start:end]

    full_index = prices_by_market[next(iter(prices_by_market))].index

    if mobility_bounds_full is not None:
        mobility_bounds_full = mobility_bounds_full.loc[full_index]

    # ============================================================
    # 6) Model options
    # ============================================================
    virtual_arbitrage = opt_conf.get("virtual_arbitrage", False)

    # Degradation cost (€/kWh throughput)
    deg_cfg = opt_conf.get("degradation", {})
    c_deg = (
        float(deg_cfg["cost_eur_per_mwh_throughput"]) / 1000.0
        if deg_cfg.get("enabled", False)
        else 0.0
    )

    # ============================================================
    # 7) Initialize committed market positions
    # ============================================================
    committed_positions = {
        mk: pd.Series(0.0, index=full_index)
        for mk in prices_by_market.keys()
    }

    # ============================================================
    # 8) Initialize energy state (band-consistent)
    # ============================================================
    if mobility_bounds_full is not None:
        E_state = 0.5 * (
            mobility_bounds_full["Capacity_lower_kWh"].iloc[0]
            + mobility_bounds_full["Capacity_upper_kWh"].iloc[0]
        )
    else:
        E_state = 0.0

    # ============================================================
    # 9) Result containers
    # ============================================================
    rows = []
    commit_rows = []

    # ============================================================
    # 10) Rolling-horizon MPC loop
    # ============================================================
    try:
        pbar = tqdm(range(len(full_index)), desc="MPC", unit="step")
        for i in pbar:
            current_time = full_index[i]
            pbar.set_postfix(time=str(current_time), E=f"{E_state:.1f} kWh")

            # --------------------------------------------------------
            # 10.1) Define rolling optimization window
            # --------------------------------------------------------
            window_end = min(i + da_horizon_steps, len(full_index))
            window_idx = full_index[i:window_end]
            if len(window_idx) == 0:
                break

            # --------------------------------------------------------
            # 10.2) Slice prices and mobility bounds
            # --------------------------------------------------------
            window_prices = {
                mk: prices_by_market[mk].loc[window_idx]
                for mk in prices_by_market
            }

            window_mobility_bounds = (
                align_and_validate_mobility_bounds(
                    bounds=mobility_bounds_full,
                    time_index=window_idx,
                )
                if mobility_bounds_full is not None
                else None
            )

            # --------------------------------------------------------
            # 10.3) Build market activity masks (GATE CLOSURES only)
            # --------------------------------------------------------
            # NOTE:
            # We distinguish between two different masks:
            #
            # trading_mask:
            #   Encodes exogenous market rules (gate closures).
            #   Determines whether a market position is still legally tradable.
            #   Used to trigger commitments once the market closes.
            #
            # decision_mask:
            #   Encodes endogenous information constraints of the optimizer.
            #   Limits the optimizer's freedom to choose new positions when
            #   price information is unavailable (e.g. intraday price foresight).
            #
            # A market may be open for trading while the optimizer deliberately
            # refrains from trading due to missing information.
            # Therefore, decision_mask is always a subset of trading_mask.
            # In general: decision_mask ⊆ trading_mask

            trading_masks = build_market_activity_mask_for_time(
                current_time=current_time,
                delivery_times=window_idx,
                optimization_cfg=opt_conf,
            )

            # --------------------------------------------------------
            # 10.3b) Build DECISION masks = trading masks + price foresight
            # --------------------------------------------------------
            decision_masks = {mk: s.copy() for mk, s in trading_masks.items()}

            # Intraday price-foresight cut-off (information constraint, not gate-closure!)
            id_horizon_end = current_time + pd.Timedelta(hours=id_horizon_hours)
            if "ID" in decision_masks:
                id_mask = decision_masks["ID"].copy()
                id_mask[window_idx > id_horizon_end] = False
                decision_masks["ID"] = id_mask

            # --------------------------------------------------------
            # 10.4) Build and parameterize optimization model
            # --------------------------------------------------------
            model = fleet_commercialization(
                vehicle=Vehicle(**opt_cfg["vehicle"]),
                site=Site(**opt_cfg["site"]),
                prices_by_market=window_prices,
                timestep_hours=step_hours,
                virtual_arbitrage=virtual_arbitrage,
                degradation_cost_eur_per_kwh=c_deg,
                market_activity_mask=decision_masks,
                committed_positions={
                    mk: committed_positions[mk].loc[window_idx]
                    for mk in committed_positions
                },
                mobility_bounds=window_mobility_bounds,
            )

            # Set initial energy state for this MPC window
            if window_mobility_bounds is not None:
                lb0 = window_mobility_bounds["Capacity_lower_kWh"].iloc[0]
                ub0 = window_mobility_bounds["Capacity_upper_kWh"].iloc[0]
                E0 = E_state if i > 0 else 0.5 * (lb0 + ub0)
                if not (lb0 - 1e-6 <= E0 <= ub0 + 1e-6):
                    logger.warning(f"E0 out of bounds at {current_time}: E0={E0}, lb={lb0}, ub={ub0} (clamping)")
                model.E0.set_value(float(min(max(E0, lb0), ub0)))
            else:
                model.E0.set_value(float(E_state))

            # --------------------------------------------------------
            # 10.5) Solve optimization problem
            # --------------------------------------------------------
            try:
                solve_model(model)
                solved = True
            except RuntimeError as e:
                solved = False
                logger.error(
                    f"MPC infeasible at current_time={current_time}: {e}"
                )
                break

            # --------------------------------------------------------
            # 10.6) Extract first-step dispatch
            # --------------------------------------------------------
            dispatch_window = extract_dispatch(model, window_idx)
            first_row = dispatch_window.iloc[0].copy()
            first_row.name = window_idx[0]
            rows.append(first_row)

            # --------------------------------------------------------
            # 10.7) Commit market positions at gate closure (TRADING masks only)
            # --------------------------------------------------------
            next_time = current_time + pd.Timedelta(hours=step_hours)

            trading_masks_next = build_market_activity_mask_for_time(
                current_time=next_time,
                delivery_times=window_idx,
                optimization_cfg=opt_conf,
            )

            for mk in committed_positions:
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
                        "gate_closure_time": compute_gate_closure(mk, tau),
                        "p_opt": float(dispatch_window.loc[tau, p_col]),
                        "committed_old": committed_positions[mk].loc[tau],
                        "committed_new": committed_positions[mk].loc[tau],
                        "commit_now": False,
                    }

                    if now_open and not next_open:
                        committed_positions[mk].loc[tau] = row["p_opt"]
                        row["committed_new"] = row["p_opt"]
                        row["commit_now"] = True

                    commit_rows.append(row)

            # --------------------------------------------------------
            # 10.8) Roll energy state forward
            # --------------------------------------------------------
            E_state = float(first_row["E_next_kWh"])

    # ============================================================
    # 11) Export results
    # ============================================================
    finally:
        if rows:
            result = pd.DataFrame(rows)
            result.index.name = "time"
            result.reset_index().to_csv("results/dispatch_mpc.csv", index=False)

        if commit_rows:
            commit_df = pd.DataFrame(commit_rows)
            commit_df.sort_values(["delivery_time", "current_time"]).to_csv(
                "results/commit_mpc.csv", index=False
            )

        print("MPC finished → results/dispatch_mpc.csv")
