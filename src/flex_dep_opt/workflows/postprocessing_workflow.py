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
from flex_dep_opt.post.plots import plot_market_cashflows_plotly, plot_mpc_dispatch_plotly
from flex_dep_opt.io.results_io import save_dispatch_to_csv, save_summary_to_csv


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
    tz = "Europe/Berlin"

    start = pd.to_datetime(sim["start"]).tz_localize(tz)
    end = pd.to_datetime(sim["end"]).tz_localize(tz)

    # -------------------------------------------------------------------------
    # Resolve CSV inputs (produced by MPC workflow)
    # -------------------------------------------------------------------------
    dispatch_csv = Path(sim["out_dispatch"]).with_suffix(".csv")
    commit_csv = Path(sim["out_commit"]).with_suffix(".csv")

    # -------------------------------------------------------------------------
    # Load dispatch.csv
    # MPC exporter wrote a "time" column; parse in UTC and convert (robust to DST).
    # -------------------------------------------------------------------------
    df = pd.read_csv(dispatch_csv)
    idx = pd.to_datetime(df["time"], utc=True).dt.tz_convert(tz)
    dispatch = df.drop(columns=["time"])
    dispatch.index = idx
    dispatch = dispatch.loc[start:end]

    # -------------------------------------------------------------------------
    # Load commit.csv
    # -------------------------------------------------------------------------
    cdf = pd.read_csv(commit_csv)
    cdf["delivery_time"] = pd.to_datetime(cdf["delivery_time"], utc=True).dt.tz_convert(tz)
    cdf["current_time"] = pd.to_datetime(cdf["current_time"], utc=True).dt.tz_convert(tz)
    commit_df = cdf[(cdf["delivery_time"] >= start) & (cdf["delivery_time"] <= end)]

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
    out_dir = dispatch_csv.parent
    cashflow_csv = out_dir / "cashflows.csv"
    kpi_csv = out_dir / "kpis.csv"

    # cashflows: keep DatetimeIndex in the CSV for later analysis
    save_dispatch_to_csv(cf_df, cashflow_csv)

    # KPIs: single row
    save_summary_to_csv(kpis, kpi_csv)

    # -------------------------------------------------------------------------
    # HTML output paths (same base names as config entries)
    # -------------------------------------------------------------------------
    dispatch_html = Path(sim["out_dispatch"]).with_suffix(".html")
    cashflow_html = Path(sim["out_commit"]).with_suffix(".html")
    dispatch_html.parent.mkdir(parents=True, exist_ok=True)

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
    print(f"MPC plot saved → {dispatch_html.resolve()}")
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
    print(f"Cashflow plot saved → {cashflow_html.resolve()}")
    webbrowser.open(cashflow_html.resolve().as_uri())

