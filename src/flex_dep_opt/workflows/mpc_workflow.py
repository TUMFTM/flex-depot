from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

from flex_dep_opt.domain.vehicle import Vehicle
from flex_dep_opt.io.prices_io import build_prices_from_settings
from flex_dep_opt.opt.model import vehicle_commercialization
from flex_dep_opt.opt.solve import solve_model, extract_dispatch
from flex_dep_opt.market.trading_rules import build_market_activity_mask


def run_mpc(cfg: dict):
    """
    Rolling-Horizon MPC:
      - In jedem Zeitschritt ein 24h-Fenster optimieren
      - Nur den ersten Zeitschritt umsetzen
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

    # Preise für alle Märkte laden und auf Simulationsfenster zuschneiden
    prices_by_market = build_prices_from_settings(cfg)

    start = pd.to_datetime(sim["start"]).tz_localize("Europe/Berlin")
    end = pd.to_datetime(sim["end"]).tz_localize("Europe/Berlin")

    for mkt in prices_by_market:
        prices_by_market[mkt] = prices_by_market[mkt].loc[start:end]

    full_index = prices_by_market[next(iter(prices_by_market))].index

    # Handelsmasken für gesamten Horizont (auch wenn trading.mode="none" → alles True)
    market_activity_mask_full = build_market_activity_mask(full_index, opt_conf)

    # Virtual arbitrage
    virt_arb = opt_conf.get("virtual_arbitrage", False)

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

        # 2) Preise und Masken für dieses Fenster zuschneiden
        window_prices = {mk: prices_by_market[mk].loc[window_idx] for mk in prices_by_market}
        window_masks = {
            mk: market_activity_mask_full[mk].loc[window_idx]
            for mk in market_activity_mask_full
        }

        # 3) Modell bauen
        model = vehicle_commercialization(
            vehicle=vehicle,
            prices_by_market=window_prices,
            timestep_hours=step_hours,
            virtual_arbitrage=virt_arb,
            degradation_cost_eur_per_kwh=c_deg,
            market_activity_mask=window_masks,
        )

        # WICHTIG: Start-SOC für dieses Fenster überschreiben
        model.soc0.set_value(float(soc))

        # 4) Lösen
        solve_model(model)

        # 5) Dispatch extrahieren und nur erste Zeile benutzen
        dispatch_window = extract_dispatch(model, window_idx)
        first_row = dispatch_window.iloc[0].copy()
        first_row.name = window_idx[0]  # Zeitindex

        rows.append(first_row)

        # 6) SOC updaten für nächsten Schritt
        soc = float(first_row["soc_kwh"])

        #if window_end_idx == len(full_index):
         #   break

    # Alles zu einem DataFrame zusammenbauen
    result = pd.DataFrame(rows)
    result.index.name = "time"

    # Zeitindex auch als Spalte (UTC) für die Plot-Workflows
    result_reset = result.reset_index()
    # ggf. nach UTC konvertieren, falls Index lokal ist:
    if result_reset["time"].dt.tz is not None:
        result_reset["time"] = result_reset["time"].dt.tz_convert("UTC")

    # Export
    out_path = Path("results/dispatch_mpc.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_reset.to_csv(out_path, index=False)

    print(f"MPC finished → {out_path.resolve()}")