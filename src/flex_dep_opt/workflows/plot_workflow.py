from pathlib import Path
import pandas as pd
import webbrowser

from flex_dep_opt.viz.plots import plot_market_cashflows_plotly,plot_mpc_dispatch_plotly
from flex_dep_opt.io.prices_io import build_prices_from_settings, build_fees_from_settings


def run_plot_mpc(cfg: dict):
    """
    Plot-Workflow speziell für MPC-Dispatch.
    Nutzt eine andere Plot-Funktion, die MPC-Charakter hervorheben kann.
    """

    sim = cfg["simulation"]

    start = pd.to_datetime(sim["start"]).tz_localize("Europe/Berlin")
    end = pd.to_datetime(sim["end"]).tz_localize("Europe/Berlin")

    # --- Resolve input paths (CSV) ---
    dispatch_csv = Path(sim["out_dispatch"]).with_suffix(".csv")
    commit_csv = Path(sim["out_commit"]).with_suffix(".csv")

    # --- Load dispatch.csv (robust) ---
    df = pd.read_csv(dispatch_csv)
    idx = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Berlin")
    dispatch = df.drop(columns=["time"])
    dispatch.index = idx
    dispatch = dispatch.loc[start:end]

    # --- Load commit_mpc.csv (optional/robust) ---
    cdf = pd.read_csv(commit_csv)
    # Parse times; delivery_time/current_time are timestamps written by MPC
    cdf["delivery_time"] = pd.to_datetime(cdf["delivery_time"], utc=True).dt.tz_convert("Europe/Berlin")
    cdf["current_time"] = pd.to_datetime(cdf["current_time"], utc=True).dt.tz_convert("Europe/Berlin")
    commit_df = cdf
    # Optional: restrict to visible window for speed
    commit_df = commit_df[(commit_df["delivery_time"] >= start) & (commit_df["delivery_time"] <= end)]

    # --- Prices ---
    prices_by_market = build_prices_from_settings(cfg)
    for mkt, s in prices_by_market.items():
        prices_by_market[mkt] = s.loc[start:end]

    # --- Fees ---
    fees_by_market = build_fees_from_settings(cfg)

    # --- Output directory: same folder as the input CSVs ---
    # --- Base paths (from config, no suffix) ---
    dispatch_base = Path(sim["out_dispatch"])
    commit_base = Path(sim["out_commit"])

    # --- CSV paths (already used for reading) ---
    dispatch_csv = dispatch_base.with_suffix(".csv")
    commit_csv = commit_base.with_suffix(".csv")

    # --- Output directory ---
    out_dir = dispatch_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- HTML output paths (same base name!) ---
    dispatch_html = dispatch_base.with_suffix(".html")
    commit_html = commit_base.with_suffix(".html")

    # --- Create dispatch figure (html) ---
    fig = plot_mpc_dispatch_plotly(
        dispatch=dispatch,
        prices_by_market=prices_by_market,
        commit_df=commit_df,
        title="MPC Flexband Dispatch and Market Positions",
    )
    fig.write_html(dispatch_html, include_plotlyjs="cdn")
    print(f"MPC Plot saved → {dispatch_html.resolve()}")
    webbrowser.open(dispatch_html.resolve().as_uri())

    # --- Create cashflow figure (html) ---
    fig_cf = plot_market_cashflows_plotly(
        dispatch=dispatch,
        commit=commit_df,
        prices_by_market=prices_by_market,
        fee_eur_per_kwh_by_market=fees_by_market,
        timestep_hours=sim["timestep_hours"],
    )
    fig_cf.write_html(commit_html, include_plotlyjs="cdn")
    print(f"CF Plot (MPC) saved → {commit_html.resolve()}")
    webbrowser.open(commit_html.resolve().as_uri())

