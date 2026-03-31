from __future__ import annotations

from pathlib import Path
import webbrowser

import pandas as pd

from flex_dep_opt.io.prices_io import build_prices_from_settings, build_fees_from_settings
from flex_dep_opt.post.metrics import (
    compute_cashflows_per_step,
    compute_market_aggregates,
    compute_kpis,
)
from flex_dep_opt.post.plots import plot_market_cashflows_plotly, plot_mpc_fcr_plotly, plot_mpc_dispatch_plotly
from flex_dep_opt.io.results_io import save_dispatch_to_csv, save_summary_to_csv, read_latest_run_pointer

from flex_dep_opt.market.fcr import generate_fcr_availability_df


def postprocess_mpc_results(cfg: dict) -> None:
    """
    Postprocessing workflow for MPC results.

    Steps
    -----
    1) Read dispatch.csv and commit.csv from cfg["simulation"]
    2) Load prices and fees from settings and slice to [start, end]
    3) Compute metrics (cashflows, aggregates, KPIs)
    4) Export optional postprocessing CSVs (cashflows + KPI summary)
    5) Generate interactive Plotly HTML plots
    """
    sim = cfg["simulation"]

    # ------------------------------------------------------------------
    # Name-based I/O (new convention)
    # ------------------------------------------------------------------
    name = str(sim.get("name", "")).strip()
    if not name:
        raise ValueError("settings.yaml: simulation.name must be set.")

    results_root = Path("results")
    run_dir = read_latest_run_pointer(results_root)

    dispatch_csv = run_dir / "dispatch.csv"
    commit_csv = run_dir / "commit.csv"

    if not dispatch_csv.exists():
        raise FileNotFoundError(f"Dispatch CSV not found: {dispatch_csv.resolve()}")
    if not commit_csv.exists():
        raise FileNotFoundError(f"Commit CSV not found: {commit_csv.resolve()}")

    # ------------------------------------------------------------------
    # Time window
    # ------------------------------------------------------------------
    start = pd.to_datetime(sim["start"]).tz_localize("Europe/Berlin")
    end = pd.to_datetime(sim["end"]).tz_localize("Europe/Berlin")

    # ------------------------------------------------------------------
    # Load dispatch
    # ------------------------------------------------------------------
    df = pd.read_csv(dispatch_csv)
    if "time" not in df.columns:
        raise ValueError(f"{dispatch_csv} must contain a 'time' column.")
    idx = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Berlin")

    dispatch = df.drop(columns=["time"])
    dispatch.index = idx
    dispatch = dispatch.loc[start:end]

    # ------------------------------------------------------------------
    # Load commit
    # ------------------------------------------------------------------
    cdf = pd.read_csv(commit_csv)

    # robust parsing if saved as strings
    if "delivery_time" in cdf.columns:
        cdf["delivery_time"] = pd.to_datetime(cdf["delivery_time"], utc=True).dt.tz_convert("Europe/Berlin")
    if "current_time" in cdf.columns:
        cdf["current_time"] = pd.to_datetime(cdf["current_time"], utc=True).dt.tz_convert("Europe/Berlin")

    commit_df = cdf
    if "delivery_time" in commit_df.columns:
        commit_df = commit_df[(commit_df["delivery_time"] >= start) & (commit_df["delivery_time"] <= end)]

    # -------------------------------------------------------------------------
    # Prices + fees
    # -------------------------------------------------------------------------
    prices_by_market = build_prices_from_settings(cfg)
    for mk, s in list(prices_by_market.items()):
        prices_by_market[mk] = s.loc[start:end]

    fees_by_market = build_fees_from_settings(cfg)

    dt = float(sim["timestep_hours"])

    # -------------------------------------------------------------------------
    # Compute metrics
    # -------------------------------------------------------------------------
    cf_df = compute_cashflows_per_step(dispatch, prices_by_market, timestep_hours=dt)
    energy_by_mk, cash_by_mk, energy_data, cash_data = compute_market_aggregates(
        dispatch, prices_by_market, timestep_hours=dt
    )
    kpis = compute_kpis(cf_df, energy_by_mk, fees_by_market, commit=commit_df)

    # -------------------------------------------------------------------------
    # Optional: persist postprocessing results next to dispatch/commit outputs
    # -------------------------------------------------------------------------
    #out_dir = dispatch_csv.parent
    cashflow_csv = run_dir / "cashflow.csv"
    kpi_csv = run_dir / "kpis.csv"

    # cashflows: keep DatetimeIndex in the CSV for later analysis
    save_dispatch_to_csv(cf_df, cashflow_csv, include_time_column=True)

    # KPIs: single row
    save_summary_to_csv(kpis, kpi_csv)

    # -------------------------------------------------------------------------
    # HTML output paths (same base names as config entries)
    # -------------------------------------------------------------------------
    dispatch_html = run_dir / "dispatch.html"
    cashflow_html = run_dir / "cashflow.html"
    dev_html = run_dir / "dev.html"

    # -------------------------------------------------------------------------
    # Plot 1: dispatch report
    # -------------------------------------------------------------------------
    fig_dispatch = plot_mpc_dispatch_plotly(
        dispatch=dispatch,
        prices_by_market=prices_by_market,
        commit_df=commit_df,
        title="MPC Flexband Dispatch and Market Positions",
    )
    fig_dispatch.write_html(dispatch_html, include_plotlyjs="cdn")
    webbrowser.open(dispatch_html.resolve().as_uri())

    # -------------------------------------------------------------------------
    # Plot 2: cashflow report (plots consume precomputed metrics)
    # -------------------------------------------------------------------------
    fig_cf = plot_market_cashflows_plotly(
        cf_df=cf_df,
        energy_data=energy_data,
        cash_data=cash_data,
        kpis=kpis,
        title="Market Cashflows",
    )
    fig_cf.write_html(cashflow_html, include_plotlyjs="cdn")
    webbrowser.open(cashflow_html.resolve().as_uri())

    symmetric_limit, fcr_grouped_capacity, fcr_result = generate_fcr_availability_df()

    fig_dev = plot_mpc_fcr_plotly(
        symmetric_limit=symmetric_limit,
        fcr_grouped_capacity=fcr_grouped_capacity,
        fcr_result=fcr_result,
        title="FCR Test Plot",
    )
    fig_dev.write_html(dev_html, include_plotlyjs="cdn")
    webbrowser.open(dev_html.resolve().as_uri())

    print(f"Result CSV files saved → {run_dir.as_posix()}")
    print(f"Result HTML plots saved → {run_dir.as_posix()}")
    print(f"Postprocessing finished")

