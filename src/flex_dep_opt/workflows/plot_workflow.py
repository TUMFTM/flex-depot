from pathlib import Path
import pandas as pd
import webbrowser

from flex_dep_opt.viz.plots import plot_dispatch_plotly


def run_plot(cfg: dict):
    """
    End-to-end plot workflow:
    Load dispatch → Load prices → Create Plot → Export HTML
    """

    sim = cfg["simulation"]
    plot_cfg = cfg["plot"]

    # Timeframe
    start = pd.to_datetime(sim["start"]).tz_localize("Europe/Berlin")
    end = pd.to_datetime(sim["end"]).tz_localize("Europe/Berlin")

    # Load dispatch
    df = pd.read_csv(plot_cfg["dispatch"])
    idx = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Berlin")
    dispatch = df.drop(columns=["time"])
    dispatch.index = idx
    dispatch = dispatch.loc[start:end]

    # Load prices
    pf = pd.read_csv(plot_cfg["prices"])
    ts = pd.to_datetime(pf["time"], utc=True).dt.tz_convert("Europe/Berlin")
    prices = pd.Series(pf["price"], index=ts)
    prices = prices.loc[start:end]

    # Create plot
    fig = plot_dispatch_plotly(dispatch, prices, capacity_kwh=plot_cfg["capacity_kwh"])

    # Export
    out = Path(plot_cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")

    print(f"Plot saved → {out.resolve()}")

    if plot_cfg.get("open", False):
        webbrowser.open(out.resolve().as_uri())