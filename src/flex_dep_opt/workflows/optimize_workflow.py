from pathlib import Path
import pandas as pd

from flex_dep_opt.domain.vehicle import Vehicle
from flex_dep_opt.io.prices_io import build_prices_from_settings
from flex_dep_opt.opt.model import vehicle_commercialization
from flex_dep_opt.opt.solve import solve_model, extract_dispatch


from pathlib import Path
import pandas as pd

from flex_dep_opt.domain.vehicle import Vehicle
from flex_dep_opt.io.prices_io import build_prices_from_settings
from flex_dep_opt.opt.model import vehicle_commercialization
from flex_dep_opt.opt.solve import solve_model, extract_dispatch


def run_optimize(cfg: dict):
    """
    End-to-end optimization workflow:
    Load → Cut timeframe → Build model → Solve → Export results
    """

    sim = cfg["simulation"]
    opt = cfg["optimize"]
    opt_conf = cfg["optimization"]
    virt_arb = opt_conf.get("virtual_arbitrage", False)

    # 1) Load all enabled market price series
    prices_by_market = build_prices_from_settings(cfg)

    # 2) Cut to simulation time window
    start = pd.to_datetime(sim["start"]).tz_localize("Europe/Berlin")
    end = pd.to_datetime(sim["end"]).tz_localize("Europe/Berlin")

    for mkt in prices_by_market:
        prices_by_market[mkt] = prices_by_market[mkt].loc[start:end]

    # 3) Vehicle object
    vehicle = Vehicle(**opt["vehicle"])

    # 4) Build generic multi-market model
    model = vehicle_commercialization(
        vehicle=vehicle,
        prices_by_market=prices_by_market,
        timestep_hours=sim["timestep_hours"],
        virtual_arbitrage=virt_arb,
    )

    # 5) Solve
    solve_model(model)

    # 6) Export results
    out_path = Path(opt["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Take time index from any market
    any_series = next(iter(prices_by_market.values()))
    time_index = any_series.index

    dispatch = extract_dispatch(model, time_index)
    dispatch.to_csv(out_path)

    print(f"Optimization finished → {out_path.resolve()}")