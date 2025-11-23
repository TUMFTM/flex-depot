from pathlib import Path
import pandas as pd

from flex_dep_opt.domain.vehicle import Vehicle
from flex_dep_opt.io.prices_io import build_prices_from_settings
from flex_dep_opt.opt.model import vehicle_commercialization
from flex_dep_opt.opt.solve import solve_model, extract_dispatch


def run_mpc(cfg: dict):
    """
    Rolling-Horizon MPC:
      - In jedem Zeitschritt ein 24h-Fenster optimieren
      - Nur den ersten 15min-Schritt übernehmen
      - SOC in die Zukunft weiterrollen
    """

    sim = cfg["simulation"]
    opt = cfg["optimize"]
    opt_conf = cfg["optimization"]

    step_hours = float(sim["timestep_hours"])
    horizon_hours = float(opt_conf["mpc"]["horizon_hours"])
    horizon_steps = int(horizon_hours / step_hours)

    # Vehicle
    vehicle = Vehicle(**opt["vehicle"])

    # Market price series
    prices_by_market = build_prices_from_settings(cfg)

    # Simulation time window
    start = pd.to_datetime(sim["start"]).tz_localize("Europe/Berlin")
    end = pd.to_datetime(sim["end"]).tz_localize("Europe/Berlin")

    # Trim prices
    for mk in prices_by_market:
        prices_by_market[mk] = prices_by_market[mk].loc[start:end]

    full_index = prices_by_market[next(iter(prices_by_market))].index

    # Initial SoC
    soc = vehicle.soc0 * vehicle.capacity_kwh

    # Storage for results
    all_results = []

    # Loop über alle Zeitschritte
    for i in range(len(full_index) - 1):

        # 1) Rolling window definieren
        window_start = full_index[i]
        window_end_idx = min(i + horizon_steps, len(full_index))
        window_idx = full_index[i:window_end_idx]

        # 2) Preise für dieses Fenster ausschneiden
        window_prices = {mk: prices_by_market[mk].loc[window_idx] for mk in prices_by_market}

        # 3) Modell bauen
        model = vehicle_commercialization(
            vehicle,
            window_prices,
            timestep_hours=step_hours,
            virtual_arbitrage=opt_conf.get("virtual_arbitrage", False),
            degradation_cost_eur_per_kwh=0.0,   # wird später ergänzt
            market_activity_mask=None,          # kommt später
        )

        # <-- Neuer SOC für Startzeitpunkt einsetzen
        model.soc0 = soc

        # 4) Solve
        solve_model(model)

        # 5) Lösung extrahieren (nur 1. Schritt übernehmen!)
        dispatch_window = extract_dispatch(model, window_idx)

        first_row = dispatch_window.iloc[0]

        all_results.append(first_row)

        # 6) SOC updaten
        soc = first_row["soc_kwh"]

        # Simulation abbrechen, wenn wir am Ende sind
        if window_end_idx == len(full_index):
            break

    # Ergebnisse zusammenbauen
    result = pd.DataFrame(all_results, index=full_index[:len(all_results)])

    # Export
    out_path = Path(opt["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path)

    print(f"MPC finished → {out_path.resolve()}")