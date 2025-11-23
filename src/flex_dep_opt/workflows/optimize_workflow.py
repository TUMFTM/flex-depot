from pathlib import Path
import pandas as pd

from flex_dep_opt.domain.vehicle import Vehicle
from flex_dep_opt.io.prices_io import build_prices_from_settings
from flex_dep_opt.opt.model import vehicle_commercialization
from flex_dep_opt.opt.solve import solve_model, extract_dispatch
from flex_dep_opt.market.trading_rules import build_market_activity_mask


def run_optimize(cfg: dict):
    """
    End-to-end optimization workflow:
    Load → Cut timeframe → Build model → Solve → Export results
    """

    sim = cfg["simulation"]
    opt = cfg["optimize"]
    opt_conf = cfg["optimization"]

    # Virtual arbitrage
    virt_arb = opt_conf.get("virtual_arbitrage", False)

    # Degradation
    deg_cfg = opt_conf.get("degradation", {})
    if deg_cfg.get("enabled", False):
        c_deg = float(deg_cfg["cost_eur_per_mwh_throughput"]) / 1000.0  # €/MWh --> €/kWh
    else:
        c_deg = 0.0

    # 1) Load all enabled market price series
    prices_by_market = build_prices_from_settings(cfg)
    # Referenz-Zeitachse (erste Marktserie)
    first_mkt = next(iter(prices_by_market.keys()))
    ref = prices_by_market[first_mkt]
    idx = ref.index

    # 2) Cut to simulation time window
    start = pd.to_datetime(sim["start"]).tz_localize("Europe/Berlin")
    end = pd.to_datetime(sim["end"]).tz_localize("Europe/Berlin")

    for mkt in prices_by_market:
        prices_by_market[mkt] = prices_by_market[mkt].loc[start:end]

    time_index = prices_by_market[first_mkt].index

    # 2) Handelsmasken bauen
    market_activity_mask = build_market_activity_mask(time_index=time_index, optimization_cfg=opt_conf)

    # 3) Vehicle object
    vehicle = Vehicle(**opt["vehicle"])

    # 4) Build generic multi-market model
    model = vehicle_commercialization(
        vehicle=vehicle,
        prices_by_market=prices_by_market,
        timestep_hours=sim["timestep_hours"],
        virtual_arbitrage=virt_arb,
        degradation_cost_eur_per_kwh=c_deg,
        market_activity_mask=market_activity_mask,
    )

    # 5) Solve
    solve_model(model)

    # 6) Export results
    out_path = Path(opt["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dispatch = extract_dispatch(model, time_index)
    dispatch.to_csv(out_path)

    print(f"Optimization finished → {out_path.resolve()}")