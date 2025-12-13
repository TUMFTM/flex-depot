from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import pyomo.environ as pyo

from flex_dep_opt.domain.vehicle import Vehicle
from flex_dep_opt.io.prices_io import build_prices_from_settings
from flex_dep_opt.io.mobility_io import read_mobility_bounds_csv, slice_mobility_bounds
from flex_dep_opt.opt.model import fleet_commercialization
from flex_dep_opt.opt.solve import solve_model, extract_dispatch
from flex_dep_opt.market.trading_rules import build_market_activity_mask_for_time


def run_mpc(cfg: dict):
    """
    Rolling-Horizon MPC:
      - In jedem Zeitschritt ein 24h-Fenster optimieren
      - Nur den ersten Zeitschritt umsetzen
      - SOC in die Zukunft weiterrollen
      - Echte Markt-Gate-Closures anwenden
    """

    sim = cfg["simulation"]
    opt = cfg["optimize"]
    opt_conf = cfg["optimization"]
    trading_cfg = opt_conf["trading"]
    mpc_cfg = opt_conf["mpc"]
    mob_cfg = opt_conf.get("mobility", {})
    mobility_bounds_full = None
    if mob_cfg.get("enabled", False):
        bounds_path = mob_cfg["bounds_file"]
        mobility_bounds_full = read_mobility_bounds_csv(bounds_path)

    def compute_gate_closure(market: str, tau: pd.Timestamp) -> pd.Timestamp:
        """
        Berechnet die Gate-Closure-Zeit für einen Lieferzeitpunkt tau
        und einen Markt (z.B. "DA", "ID"), konsistent zu
        build_market_activity_mask_for_time().
        """
        if market == "ID":
            id_cfg = trading_cfg.get("intraday", {})
            offset_min = int(id_cfg.get("offset_minutes_before_delivery", 30))
            return tau - pd.Timedelta(minutes=offset_min)

        if market == "DA":
            da_cfg = trading_cfg.get("dayahead", {})
            gc_hour_str = da_cfg.get("gate_closure_hour", "12:00")
            closes_prev = da_cfg.get("closes_previous_day", True)
            try:
                hh_str, mm_str = gc_hour_str.split(":")
                gc_h = int(hh_str)
                gc_m = int(mm_str)
            except Exception:
                gc_h, gc_m = 12, 0  # Fallback
            # Basisdatum = Lieferdatum oder Vortag
            if closes_prev:
                gc_date = tau.normalize() - pd.Timedelta(days=1)
            else:
                gc_date = tau.normalize()
            gc_ts = gc_date + pd.Timedelta(hours=gc_h, minutes=gc_m)
            return gc_ts


    step_hours = float(sim["timestep_hours"])
    da_horizon_hours = float(mpc_cfg["da_horizon_hours"])
    id_horizon_hours = float(mpc_cfg["id_horizon_hours"])
    da_horizon_steps = int(da_horizon_hours / step_hours)

    # Vehicle
    vehicle = Vehicle(**opt["vehicle"])

    # Preise für alle Märkte laden und auf Simulationsfenster zuschneiden
    prices_by_market = build_prices_from_settings(cfg)

    start = pd.to_datetime(sim["start"]).tz_localize("Europe/Berlin")
    end = pd.to_datetime(sim["end"]).tz_localize("Europe/Berlin")

    for mkt in prices_by_market:
        prices_by_market[mkt] = prices_by_market[mkt].loc[start:end]

    full_index = prices_by_market[next(iter(prices_by_market))].index

    # Mobility auf Simulationsfenster zuschneiden
    if mobility_bounds_full is not None:
        mobility_bounds_full = mobility_bounds_full.loc[full_index]

    # Virtual arbitrage
    virt_arb = opt_conf.get("virtual_arbitrage", False)

    # Committed market positions über die gesamte Simulation ---
    committed_positions = {}
    for mk in prices_by_market.keys():
        # initial keine Verpflichtungen: alles 0
        committed_positions[mk] = pd.Series(0.0, index=full_index)

    # Degradation
    deg_cfg = opt_conf.get("degradation", {})
    if deg_cfg.get("enabled", False):
        c_deg = float(deg_cfg["cost_eur_per_mwh_throughput"]) / 1000.0  # €/kWh
    else:
        c_deg = 0.0

    # Anfangs-SOC in kWh
    soc = vehicle.soc0 * vehicle.capacity_kwh

    # Ergebnisse sammeln (eine Zeile pro sim-Schritt)
    rows = []
    commit_rows = []

    n_steps = len(full_index)

    pbar = tqdm(range(n_steps), desc="MPC", unit="step")
    for i in pbar:
        current_time = full_index[i]
        pbar.set_postfix(time=str(current_time), soc=f"{soc:.1f} kWh")

        # 1) Rolling-Fenster definieren
        window_start = full_index[i]
        window_end_idx = min(i + da_horizon_steps, len(full_index))
        window_idx = full_index[i:window_end_idx]
        is_last_window = (window_end_idx == len(full_index))

        if len(window_idx) == 0:
            break

        # 2) Preisfenster bauen
        window_prices = {mk: prices_by_market[mk].loc[window_idx]for mk in prices_by_market}

        # 2a) Mobilitäts-Bounds für Fenster bauen
        window_mobility_bounds = None
        if mobility_bounds_full is not None:
            window_mobility_bounds = slice_mobility_bounds(mobility_bounds_full, window_idx)

        # 2b) Handelsmasken entsprechend GATE CLOSURE REGELN
        window_masks = build_market_activity_mask_for_time(current_time=current_time,delivery_times=window_idx,optimization_cfg=opt_conf)
        id_horizon_end = current_time + pd.Timedelta(hours=id_horizon_hours)

        if "ID" in window_masks:
            id_mask = window_masks["ID"].copy()
            # alles jenseits des ID-Horizonts auf "geschlossen" setzen
            id_mask[window_idx > id_horizon_end] = False
            window_masks["ID"] = id_mask

        # 3) Modell bauen
        model = fleet_commercialization(
            vehicle=vehicle,
            prices_by_market=window_prices,
            timestep_hours=step_hours,
            virtual_arbitrage=virt_arb,
            degradation_cost_eur_per_kwh=c_deg,
            market_activity_mask=window_masks,
            committed_positions={mk: committed_positions[mk].loc[window_idx] for mk in committed_positions},
            enforce_terminal_soc=False,
            mobility_bounds=window_mobility_bounds,
        )

        # Start-SOC für dieses Fenster überschreiben
        if window_mobility_bounds is not None:
            E0_mid = 0.5 * (float(window_mobility_bounds["Capacity_lower_kWh"].iloc[0])+float(window_mobility_bounds["Capacity_upper_kWh"].iloc[0]))
            model.E0.set_value(E0_mid)
        else:
            model.E0.set_value(0.0)

        # 4) Lösen
        solve_model(model)

        # 5) Dispatch extrahieren und nur erste Zeile benutzen
        dispatch_window = extract_dispatch(model, window_idx)
        first_row = dispatch_window.iloc[0].copy()
        first_row.name = window_idx[0]  # Zeitindex

        # === NEU: Marktpositionen committen, wenn Gate Closure überschritten wird ===
        # Wir vergleichen Maske zum aktuellen Zeitpunkt und zum "nächsten" MPC-Schritt
        next_time = current_time + pd.Timedelta(hours=step_hours)

        masks_now = window_masks
        masks_next = build_market_activity_mask_for_time(
            current_time=next_time,
            delivery_times=window_idx,
            optimization_cfg=opt_conf,
        )

        for mk in committed_positions.keys():
            # passende Spaltennamen in dispatch_window: z.B. "p_id_kw" für "ID"
            p_col = f"p_{mk.lower()}_kw"
            if p_col not in dispatch_window.columns:
                continue  # diesen Markt ignorieren, falls nicht im Dispatch

            mask_now_mk = masks_now.get(mk)
            mask_next_mk = masks_next.get(mk)
            if mask_now_mk is None or mask_next_mk is None:
                continue

            for tau in window_idx:
                now_open = bool(mask_now_mk.loc[tau])
                next_open = bool(mask_next_mk.loc[tau])

                gc_ts = compute_gate_closure(mk, tau)
                val = float(dispatch_window.loc[tau, p_col])

                row = {
                    "market": mk,
                    "delivery_time": tau,
                    "current_time": current_time,
                    "next_time": next_time,
                    "gate_closure_time": gc_ts,
                    "p_opt": val,
                    "open_now": now_open,
                    "open_next": next_open,
                    "committed_old": committed_positions[mk].loc[tau],
                    "committed_new": committed_positions[mk].loc[tau],  # evtl. gleich, wird unten überschrieben
                    "commit_now": False,
                }

                # Gate Closure: bisher handelbar, im nächsten Schritt nicht mehr
                if now_open and (not next_open):
                    committed_positions[mk].loc[tau] = val
                    row["commit_now"] = True
                    row["committed_new"] = val

                    # Optionales Debugging für die ersten paar Commits:
                    #if i < 5:
                    #    print(f"[COMMIT] mk={mk}, delivery={tau}, "
                    #          f"p={val:.2f} kW (current_time={current_time}, next_time={next_time})")

                commit_rows.append(row)

        rows.append(first_row)

        # 6) SOC updaten für nächsten Schritt
        soc = float(first_row["E_kWh"])

    # Alles zu einem DataFrame zusammenbauen
    result = pd.DataFrame(rows)
    result.index.name = "time"
    commit_df = pd.DataFrame(commit_rows)
    commit_df.sort_values(["delivery_time", "current_time"], inplace=True)

    # Zeitindex auch als Spalte (UTC) für die Plot-Workflows
    result_reset = result.reset_index()

    # Export
    out_path = Path("results/dispatch_mpc.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_reset.to_csv(out_path, index=False)
    out_path_commit = Path("results/commit_mpc.csv")
    out_path_commit.parent.mkdir(parents=True, exist_ok=True)
    commit_df.to_csv(out_path_commit, index=False)

    print(f"MPC finished → {out_path.resolve()}")