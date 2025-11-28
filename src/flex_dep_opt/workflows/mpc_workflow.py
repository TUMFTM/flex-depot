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

        # DEBUG: Bounds der Marktvariablen prüfen
        #mk_test = list(window_masks.keys())[0]  # z.B. "ID"
        #allowed_slots = window_masks[mk_test][window_masks[mk_test]].index
        #if len(allowed_slots) > 0:
        #    debug_time = allowed_slots[0]  # erste erlaubte Stunde
        #    debug_t = list(window_idx).index(debug_time)  # Index im MPC-Fenster
        #    print("DEBUG MARKET SLOT:", mk_test, debug_time, "→ t =", debug_t)
        #    print("p_market bounds:",
        #          model.p_market[mk_test, debug_t].lb,
        #          model.p_market[mk_test, debug_t].ub)
        #    print("p_ch bounds:",
        #          model.p_ch[debug_t].lb,
        #          model.p_ch[debug_t].ub)
        #    print("p_dis bounds:",
        #          model.p_dis[debug_t].lb,
        #          model.p_dis[debug_t].ub)

        # 4) Lösen
        solve_model(model)

        # === DEBUG: erste MPC-Optimierung inspizieren ===
        #if i == 0:
        #    print("\n=== DEBUG: Lösung für erstes MPC-Fenster ===")
        #    print("Solver-Objective:", pyo.value(model.obj))
        #    # ein paar Zeitschritte anschauen (z.B. t = 0..5)
        #    for t in range(min(6, len(window_idx))):
        #        ts = window_idx[t]
        #        print(f"\n--- t = {t}, time = {ts} ---")
        #        print("price_ID =", model.price["ID", t])
        #        print("p_ch      =", model.p_ch[t].value)
        #        print("p_dis     =", model.p_dis[t].value)
        #        if hasattr(model, "p_market_pos"):  # MILP-Zweig (virtual_arbitrage=False)
        #            print("p_market_pos[ID] =", model.p_market_pos["ID", t].value)
        #            print("p_market_neg[ID] =", model.p_market_neg["ID", t].value)
        #        print("p_market[ID]   =", model.p_market["ID", t].value)
        #    print("=== ENDE DEBUG FENSTER 1 ===\n")

        # 5) Dispatch extrahieren und nur erste Zeile benutzen
        dispatch_window = extract_dispatch(model, window_idx)
        first_row = dispatch_window.iloc[0].copy()
        first_row.name = window_idx[0]  # Zeitindex

        #if i == 0:
        #    print("\n=== DEBUG: extract_dispatch – erstes Fenster ===")
        #    print(dispatch_window.head(6))  # zeigt alle Spalten
        #    print("FIRST ROW (t=0):")
        #    print(first_row)
        #    print("=== ENDE DEBUG extract_dispatch ===\n")

        rows.append(first_row)

        # 6) SOC updaten für nächsten Schritt
        soc = float(first_row["soc_kwh"])


    # Alles zu einem DataFrame zusammenbauen
    result = pd.DataFrame(rows)
    result.index.name = "time"

    # Zeitindex auch als Spalte (UTC) für die Plot-Workflows
    result_reset = result.reset_index()

    # Export
    out_path = Path("results/dispatch_mpc.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_reset.to_csv(out_path, index=False)

    print(f"MPC finished → {out_path.resolve()}")