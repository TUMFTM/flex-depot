from pathlib import Path
import pandas as pd

from flex_dep_opt.domain.vehicle import Vehicle
from flex_dep_opt.opt.model import build_single_vehicle_model
from flex_dep_opt.opt.solve import solve_model, extract_dispatch


def run_optimize(cfg: dict):
    """
    End-to-end optimization workflow:
    Load → Cut timeframe → Build model → Solve → Export results
    """

    sim = cfg["simulation"]
    opt = cfg["optimize"]
    veh_cfg = opt["vehicle"]

    # Load prices
    df = pd.read_csv(opt["prices"])
    idx = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Berlin")
    prices = pd.Series(df["price"].values, index=idx).sort_index()

    # Timeframe
    start = pd.to_datetime(sim["start"]).tz_localize("Europe/Berlin")
    end = pd.to_datetime(sim["end"]).tz_localize("Europe/Berlin")
    prices = prices.loc[start:end]

    # Vehicle
    vehicle = Vehicle(**veh_cfg)

    # Build model
    model = build_single_vehicle_model(
        vehicle,
        prices,
        timestep_hours=sim["timestep_hours"]
    )

    # Solve
    solve_model(model)

    # Save results
    out_path = Path(opt["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dispatch = extract_dispatch(model, prices.index)
    dispatch.to_csv(out_path)

    print(f"Optimization finished → {out_path.resolve()}")