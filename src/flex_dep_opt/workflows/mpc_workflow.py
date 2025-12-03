from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import pyomo.environ as pyo

from flex_dep_opt.domain.vehicle import Vehicle
from flex_dep_opt.io.prices_io import build_prices_from_settings
from flex_dep_opt.opt.model import vehicle_commercialization
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

    step_hours = float(sim["timestep_hours"])
    horizon_hours = float(opt_conf["mpc"]["horizon_hours"])
    horizon_steps = int(horizon_hours / step_hours)

    # Vehicle
    vehicle = Vehicle(**opt["vehicle"])

    # Preise für alle Märkte laden und auf Simulationsfenster zuschneiden
    prices_by_market = build_prices_from_settings(cfg)

    start = pd.to_datetime(sim["start"]).tz_localize("Europe/Berlin")
    end = pd.to_datetime(sim["end"]).tz_localize("Europe/Berlin")

    for mkt in prices_by_market:
        prices_by_market[mkt] = prices_by_market[mkt].loc[start:end]

    full_index = prices_by_market[next(iter(prices_by_market))].index

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
        window_end_idx = min(i + horizon_steps, len(full_index))
        window_idx = full_index[i:window_end_idx]

        if len(window_idx) == 0:
            break

        # 2) Preisfenster bauen
        window_prices = {mk: prices_by_market[mk].loc[window_idx]for mk in prices_by_market}

        # 2b) Handelsmasken entsprechend GATE CLOSURE REGELN
        window_masks = build_market_activity_mask_for_time(current_time=current_time,delivery_times=window_idx,optimization_cfg=opt_conf)

        # 3) Modell bauen
        model = vehicle_commercialization(
            vehicle=vehicle,
            prices_by_market=window_prices,
            timestep_hours=step_hours,
            virtual_arbitrage=virt_arb,
            degradation_cost_eur_per_kwh=c_deg,
            market_activity_mask=window_masks,
            committed_positions={mk: committed_positions[mk].loc[window_idx] for mk in committed_positions},
        )

        # Start-SOC für dieses Fenster überschreiben
        model.soc0.set_value(float(soc))

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

                gc_ts = tau - pd.Timedelta(
                    minutes=int(opt_conf["trading"]["intraday"]["offset_minutes_before_delivery"])
                )
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
                    if i < 3:
                        print(f"[COMMIT] mk={mk}, delivery={tau}, "
                              f"p={val:.2f} kW (current_time={current_time}, next_time={next_time})")

                commit_rows.append(row)

        rows.append(first_row)

        # 6) SOC updaten für nächsten Schritt
        soc = float(first_row["soc_kwh"])

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
    out_path_commit = Path("results/commit_mpc.csv", index=False)
    out_path_commit.parent.mkdir(parents=True, exist_ok=True)
    commit_df.to_csv(out_path_commit, index=False)

    print(f"MPC finished → {out_path.resolve()}")