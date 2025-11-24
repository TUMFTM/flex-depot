from pathlib import Path
import pandas as pd
import webbrowser

from flex_dep_opt.viz.plots import plot_dispatch_multimarket_plotly,plot_market_cashflows_plotly,plot_mpc_dispatch_plotly
from flex_dep_opt.io.prices_io import build_prices_from_settings


def run_plot(cfg: dict):
    """
    End-to-end plot workflow:
    Load dispatch → Load prices (all markets) → Create Plot → Export HTML
    """

    sim = cfg["simulation"]
    plot_cfg = cfg["plot"]

    # Timeframe
    start = pd.to_datetime(sim["start"]).tz_localize("Europe/Berlin")
    end = pd.to_datetime(sim["end"]).tz_localize("Europe/Berlin")

    # --- 1) Dispatch laden ---
    df = pd.read_csv(plot_cfg["dispatch"])
    idx = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Berlin")
    dispatch = df.drop(columns=["time"])
    dispatch.index = idx
    dispatch = dispatch.loc[start:end]

    # --- 2) Preise für alle aktivierten Märkte laden ---
    prices_by_market = build_prices_from_settings(cfg)

    for mkt, s in prices_by_market.items():
        prices_by_market[mkt] = s.loc[start:end]

    # --- 3) Plot erzeugen (Multi-Markt) ---
    fig = plot_dispatch_multimarket_plotly(
        dispatch=dispatch,
        prices_by_market=prices_by_market,
        capacity_kwh=plot_cfg["capacity_kwh"],
        title=plot_cfg.get("title", "Dispatch and Market Positions"),
    )

    # --- 4) Export ---
    out = Path(plot_cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")

    print(f"Plot saved → {out.resolve()}")

    if plot_cfg.get("open", False):
        webbrowser.open(out.resolve().as_uri())

    # --- 5) Cashflow-Plot (optional) ---
    if "cashflow_out" in plot_cfg:
        out_cf = Path(plot_cfg["cashflow_out"])
        out_cf.parent.mkdir(parents=True, exist_ok=True)
        fig_cf = plot_market_cashflows_plotly(
            dispatch=dispatch,
            prices_by_market=prices_by_market,
            timestep_hours=sim["timestep_hours"],
        )
        fig_cf.write_html(out_cf, include_plotlyjs="cdn")

        print(f"CF Plot saved → {out_cf.resolve()}")
        if plot_cfg.get("open", False):
            webbrowser.open(out_cf.resolve().as_uri())



def run_plot_mpc(cfg: dict):
    """
    Plot-Workflow speziell für MPC-Dispatch.
    Nutzt eine andere Plot-Funktion, die MPC-Charakter hervorheben kann.
    """

    sim = cfg["simulation"]
    plot_cfg = cfg["plot"]

    start = pd.to_datetime(sim["start"]).tz_localize("Europe/Berlin")
    end = pd.to_datetime(sim["end"]).tz_localize("Europe/Berlin")

    df = pd.read_csv(plot_cfg["dispatch_mpc"])
    idx = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Berlin")
    dispatch = df.drop(columns=["time"])
    dispatch.index = idx
    dispatch = dispatch.loc[start:end]

    prices_by_market = build_prices_from_settings(cfg)
    for mkt, s in prices_by_market.items():
        prices_by_market[mkt] = s.loc[start:end]

    fig = plot_mpc_dispatch_plotly(
        dispatch=dispatch,
        prices_by_market=prices_by_market,
        capacity_kwh=plot_cfg["capacity_kwh"],
        title=plot_cfg.get("title", "MPC Dispatch and Market Positions"),
    )

    out = Path(plot_cfg.get("mpc_out", "results/dispatch_mpc_plot.html"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")

    print(f"MPC Plot saved → {out.resolve()}")

    if plot_cfg.get("open", False):
        webbrowser.open(out.resolve().as_uri())

    # Cashflow-Plot kannst du bei MPC genauso verwenden, wenn du willst:
    if "cashflow_out" in plot_cfg:
        out_cf = Path(plot_cfg["cashflow_out"])
        out_cf.parent.mkdir(parents=True, exist_ok=True)
        fig_cf = plot_market_cashflows_plotly(
            dispatch=dispatch,
            prices_by_market=prices_by_market,
            timestep_hours=sim["timestep_hours"],
        )
        fig_cf.write_html(out_cf, include_plotlyjs="cdn")

        print(f"CF Plot (MPC) saved → {out_cf.resolve()}")
        if plot_cfg.get("open", False):
            webbrowser.open(out_cf.resolve().as_uri())