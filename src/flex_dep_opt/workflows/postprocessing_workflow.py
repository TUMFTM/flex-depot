from __future__ import annotations

from pathlib import Path
import webbrowser

from flex_dep_opt.config.settings import Settings
import pandas as pd

from flex_dep_opt.io.prices_io import build_prices_from_settings, build_fees_from_settings
from flex_dep_opt.post.metrics import (
    compute_cashflows_per_step,
    compute_market_aggregates,
    compute_kpis,
)
from flex_dep_opt.post.plots import plot_market_cashflows_plotly, plot_mpc_dispatch_plotly
from flex_dep_opt.io.results_io import save_dispatch_to_csv, save_summary_to_csv, read_latest_run_pointer
from flex_dep_opt.market.fcr import get_fcr_frequency_data

def postprocess_mpc_results(settings: Settings) -> None:
    """
    Postprocessing workflow for MPC results.

    Steps
    -----
    1) Read dispatch.csv and commit.csv from settings.simulation
    2) Load prices and fees from settings and slice to [start, end]
    3) Compute metrics (cashflows, aggregates, KPIs)
    4) Export optional postprocessing CSVs (cashflows + KPI summary)
    5) Generate interactive Plotly HTML plots
    """
    sim = settings.simulation


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
    start = pd.to_datetime(sim.start).tz_localize("Europe/Berlin")
    end = pd.to_datetime(sim.end).tz_localize("Europe/Berlin")

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

    fcr_commit_df: pd.DataFrame | None = None
    fcr_commit_csv = run_dir / "fcr_commit.csv"
    if fcr_commit_csv.exists():
        fcr_cdf = pd.read_csv(fcr_commit_csv)
        if "slot_start" in fcr_cdf.columns:
            fcr_cdf["slot_start"] = pd.to_datetime(fcr_cdf["slot_start"], utc=True).dt.tz_convert("Europe/Berlin")
        if "committed_at" in fcr_cdf.columns:
            fcr_cdf["committed_at"] = pd.to_datetime(fcr_cdf["committed_at"], utc=True).dt.tz_convert("Europe/Berlin")
        fcr_commit_df = fcr_cdf

    # -------------------------------------------------------------------------
    # Prices + fees
    # -------------------------------------------------------------------------
    prices_by_market = build_prices_from_settings(settings)
    for mk, s in list(prices_by_market.items()):
        prices_by_market[mk] = s.loc[start:end]

    fees_by_market = build_fees_from_settings(settings)

    dt = sim.timestep_hours

    fcr_kpis: dict = {}
    if "x_fcr_kw" in dispatch.columns and "fcr" in prices_by_market:
        x_fcr_kw = dispatch["x_fcr_kw"]
        fcr_prices = prices_by_market["fcr"]

        committed_mw = x_fcr_kw / 1000.0
        slot_revenue = committed_mw * fcr_prices.reindex(committed_mw.index, method="ffill")

        fcr_kpis = {
            "fcr_revenue_eur": float(slot_revenue.sum()),
            "fcr_slots_committed": int((committed_mw > 0).sum()),
            "fcr_avg_capacity_mw": float(committed_mw[committed_mw > 0].mean())
            if (committed_mw > 0).any() else 0.0,
        }

    # -------------------------------------------------------------------------
    # Compute metrics
    # -------------------------------------------------------------------------
    cf_df = compute_cashflows_per_step(dispatch, prices_by_market, timestep_hours=dt)
    energy_by_mk, cash_by_mk, energy_data, cash_data = compute_market_aggregates(
        dispatch, prices_by_market, timestep_hours=dt
    )
    kpis = compute_kpis(cf_df, energy_by_mk, fees_by_market, commit=commit_df)
    kpis.update(fcr_kpis)

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

    # -------------------------------------------------------------------------
    # Plot 1: dispatch report
    # -------------------------------------------------------------------------

    fcr_cfg = settings.optimization.trading.fcr
    fcr_frequency_data_pp: pd.DataFrame | None = None
    if fcr_cfg.enabled and getattr(fcr_cfg, "frequency_source", None):
        try:
            fcr_frequency_data_pp = get_fcr_frequency_data(fcr_cfg.frequency_source)
            fcr_frequency_data_pp = fcr_frequency_data_pp.loc[
                (fcr_frequency_data_pp.index >= start) &
                (fcr_frequency_data_pp.index <= end)
            ]
        except Exception:
            fcr_frequency_data_pp = None

    fig_dispatch = plot_mpc_dispatch_plotly(
        dispatch=dispatch,
        prices_by_market=prices_by_market,
        commit_df=commit_df,
        fcr_commit_df=fcr_commit_df,
        title="MPC Flexband Dispatch and Market Positions",
        fcr_energy_req_hours=float(fcr_cfg.energy_req_hours),
        fcr_frequency_data=fcr_frequency_data_pp,
    )
    fig_dispatch.write_html(dispatch_html, include_plotlyjs="cdn")
    #webbrowser.open(dispatch_html.resolve().as_uri())

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
    #webbrowser.open(cashflow_html.resolve().as_uri())

    print(f"Result CSV files saved → {run_dir.as_posix()}")
    print(f"Result HTML plots saved → {run_dir.as_posix()}")
    print(f"Postprocessing finished")