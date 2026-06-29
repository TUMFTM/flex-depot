from __future__ import annotations

from pathlib import Path
import webbrowser

import pandas as pd

from flex_dep_opt.io.prices_io import build_prices_from_settings, build_fees_from_settings
from flex_dep_opt.io.time_utils import LOCAL_TIMEZONE, local_config_timestamp_to_utc, validate_regular_index
from flex_dep_opt.post.metrics import (
    compute_cashflows_per_step,
    compute_market_aggregates,
    compute_kpis,
)
from flex_dep_opt.post.reference_energy_costs import (
    DEFAULT_REFERENCE_ENERGY_COLUMN,
    compute_reference_driving_energy_costs,
)
from flex_dep_opt.post.plots import plot_market_cashflows_plotly, plot_mpc_dispatch_plotly
from flex_dep_opt.io.results_io import save_dispatch_to_csv, save_summary_to_csv, read_latest_run_pointer


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
    start = local_config_timestamp_to_utc(sim["start"], local_tz=LOCAL_TIMEZONE)
    end = local_config_timestamp_to_utc(sim["end"], local_tz=LOCAL_TIMEZONE)

    # ------------------------------------------------------------------
    # Load dispatch
    # ------------------------------------------------------------------
    df = pd.read_csv(dispatch_csv)
    if "time" not in df.columns:
        raise ValueError(f"{dispatch_csv} must contain a 'time' column.")
    idx = pd.to_datetime(df["time"], utc=True).dt.tz_convert("UTC")

    dispatch = df.drop(columns=["time"])
    dispatch.index = idx
    dispatch = dispatch.loc[start:end]
    validate_regular_index(dispatch.index, timestep_hours=float(sim["timestep_hours"]), name="dispatch index")

    # ------------------------------------------------------------------
    # Load commit
    # ------------------------------------------------------------------
    cdf = pd.read_csv(commit_csv)

    # robust parsing if saved as strings
    if "delivery_time" in cdf.columns:
        cdf["delivery_time"] = pd.to_datetime(cdf["delivery_time"], utc=True).dt.tz_convert("UTC")
    if "current_time" in cdf.columns:
        cdf["current_time"] = pd.to_datetime(cdf["current_time"], utc=True).dt.tz_convert("UTC")

    commit_df = cdf
    if "delivery_time" in commit_df.columns:
        commit_df = commit_df[(commit_df["delivery_time"] >= start) & (commit_df["delivery_time"] <= end)]

    # -------------------------------------------------------------------------
    # Prices + fees
    # -------------------------------------------------------------------------
    prices_by_market = build_prices_from_settings(cfg)
    for mk, s in list(prices_by_market.items()):
        prices_by_market[mk] = s.loc[start:end]
        validate_regular_index(
            prices_by_market[mk].index,
            timestep_hours=float(sim["timestep_hours"]),
            name=f"{mk} price index",
        )

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
    save_dispatch_to_csv(cf_df, cashflow_csv, include_time_column=True, output_tz=LOCAL_TIMEZONE)

    # -------------------------------------------------------------------------
    # Optional static-price reference scenario for driving energy
    # -------------------------------------------------------------------------
    reference_df = None
    reference_summary = None

    post_cfg = cfg.get("postprocessing", {}) or {}
    if not isinstance(post_cfg, dict):
        raise ValueError("settings.yaml: postprocessing must be a mapping if provided.")
    ref_cfg = post_cfg.get("reference_driving_energy_costs", {}) or {}
    if not isinstance(ref_cfg, dict):
        raise ValueError("settings.yaml: postprocessing.reference_driving_energy_costs must be a mapping.")
    if ref_cfg.get("enabled", False):
        static_price = ref_cfg.get("static_price_eur_per_kwh")
        if static_price is None:
            raise ValueError(
                "settings.yaml: postprocessing.reference_driving_energy_costs.static_price_eur_per_kwh "
                "must be set when reference calculation is enabled."
            )

        flex_file = cfg["optimization"].get("flexibility", {}).get("bounds_file")
        if not flex_file:
            raise ValueError("settings.yaml: optimization.flexibility.bounds_file must be set.")

        reference_df, reference_summary = compute_reference_driving_energy_costs(
            flex_file,
            start=start,
            end=end,
            static_price_eur_per_kwh=float(static_price),
            energy_column=str(ref_cfg.get("energy_column", DEFAULT_REFERENCE_ENERGY_COLUMN)),
        )
        reference_csv = run_dir / "reference_driving_energy_costs.csv"
        save_dispatch_to_csv(reference_df, reference_csv, include_time_column=True, output_tz=LOCAL_TIMEZONE)

        reference_gross_profit_eur = -float(reference_summary["ref_energy_cost_eur"])
        kpis.update({
            "ref_gross_profit_eur": reference_gross_profit_eur,
            "ref_driving_energy_kwh": float(reference_summary["ref_driving_energy_kwh"]),
            "ref_static_price_eur_per_kwh": float(reference_summary["ref_static_price_eur_per_kwh"]),
            "ref_energy_cost_eur": float(reference_summary["ref_energy_cost_eur"]),
            "total_potential_gross_profit_delta_eur": (
                float(kpis["gross_profit_eur"]) - reference_gross_profit_eur
            ),
        })

        print(
            "Reference driving energy costs: "
            f"{reference_summary['ref_energy_cost_eur']:.2f} EUR "
            f"for {reference_summary['ref_driving_energy_kwh']:.2f} kWh"
        )

    # KPIs: single row, including optional reference scenario fields
    save_summary_to_csv(kpis, kpi_csv)

    # -------------------------------------------------------------------------
    # HTML output paths (same base names as config entries)
    # -------------------------------------------------------------------------
    dispatch_html = run_dir / "dispatch.html"
    cashflow_html = run_dir / "cashflow.html"

    # -------------------------------------------------------------------------
    # Plot 1: dispatch report
    # -------------------------------------------------------------------------
    dispatch_plot = dispatch.copy()
    dispatch_plot.index = dispatch_plot.index.tz_convert(LOCAL_TIMEZONE)
    prices_plot = {mk: s.tz_convert(LOCAL_TIMEZONE) for mk, s in prices_by_market.items()}
    commit_plot = commit_df.copy()
    for col in ("delivery_time", "current_time", "next_time", "gate_closure_time"):
        if col in commit_plot.columns and pd.api.types.is_datetime64_any_dtype(commit_plot[col]):
            if getattr(commit_plot[col].dt, "tz", None) is not None:
                commit_plot[col] = commit_plot[col].dt.tz_convert(LOCAL_TIMEZONE)

    fig_dispatch = plot_mpc_dispatch_plotly(
        dispatch=dispatch_plot,
        prices_by_market=prices_plot,
        commit_df=commit_plot,
        title="MPC Flexband Dispatch and Market Positions",
    )
    fig_dispatch.write_html(dispatch_html, include_plotlyjs="cdn")
    webbrowser.open(dispatch_html.resolve().as_uri())

    # -------------------------------------------------------------------------
    # Plot 2: cashflow report (plots consume precomputed metrics)
    # -------------------------------------------------------------------------
    cf_plot = cf_df.copy()
    cf_plot.index = cf_plot.index.tz_convert(LOCAL_TIMEZONE)
    fig_cf = plot_market_cashflows_plotly(
        cf_df=cf_plot,
        energy_data=energy_data,
        cash_data=cash_data,
        kpis=kpis,
        reference_df=(
            reference_df.tz_convert(LOCAL_TIMEZONE)
            if reference_df is not None
            else None
        ),
        reference_summary=reference_summary,
        title="Market Cashflows",
    )
    fig_cf.write_html(cashflow_html, include_plotlyjs="cdn")
    webbrowser.open(cashflow_html.resolve().as_uri())

    print(f"Result CSV files saved → {run_dir.as_posix()}")
    print(f"Result HTML plots saved → {run_dir.as_posix()}")
    print(f"Postprocessing finished")

