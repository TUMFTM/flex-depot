# Interactive Plotly visualization of dispatch and prices using subplots.

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict

# Einheitliche Farbcodierung je Markt (R, G, B)
MARKET_COLORS: Dict[str, tuple[int, int, int]] = {
    "DA": (0,101,189),
    "ID": (227,114,34),
    # später z.B. "FCR": (0, 120, 255),
}


def _rgb(market: str) -> str:
    """Hilfsfunktion: rgb()-String aus MARKET_COLORS."""
    r, g, b = MARKET_COLORS.get(market, (100, 100, 100))
    return f"rgb({r},{g},{b})"


def _rgba(market: str, alpha: float) -> str:
    """Hilfsfunktion: rgba()-String aus MARKET_COLORS + Alpha."""
    r, g, b = MARKET_COLORS.get(market, (100, 100, 100))
    return f"rgba({r},{g},{b},{alpha})"


def plot_dispatch_multimarket_plotly(
    dispatch: pd.DataFrame,
    prices_by_market: dict[str, pd.Series] | None = None,
    *,
    capacity_kwh: float,
    title: str = "Dispatch and Market Positions",
) -> go.Figure:
    """
    Multi-market visualization:

      Subplot 1 (oben):   Prices per market [€/MWh]
      Subplot 2 (mitte):  State of Charge [%] + Net Power [kW]
      Subplot 3 (unten):  Market positions per market [kW] as bars
    """
    if not isinstance(dispatch.index, pd.DatetimeIndex):
        raise ValueError("Dispatch index must be a DatetimeIndex")

    # Net power (positive = discharging)
    net_power = dispatch["p_dis_kw"] - dispatch["p_ch_kw"]

    # SoC in %
    soc_kwh = dispatch["soc_kwh"]
    soc_percent = (soc_kwh / float(capacity_kwh)) * 100.0

    # Marktspalten automatisch erkennen: p_{mk}_kw, aber p_ch/p_dis ignorieren
    market_cols = [
        c for c in dispatch.columns
        if c.startswith("p_")
        and c.endswith("_kw")
        and c not in ("p_ch_kw", "p_dis_kw")
    ]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        specs=[
            [{"secondary_y": False}],   # Prices
            [{"secondary_y": True}],    # Net Power + SoC
            [{"secondary_y": False}],   # Market positions (bars)
        ],
        subplot_titles=(
            "Market Prices",
            "State of Charge & Net Power",
            "Market Positions",
        ),
    )

    # === Subplot 1: nur Preise ===
    if prices_by_market is not None:
        for mk, s in prices_by_market.items():
            price_mwh = s * 1000.0
            color = _rgb(mk)

            fig.add_trace(
                go.Scatter(
                    x=price_mwh.index,
                    y=price_mwh.values,
                    mode="lines",
                    name=f"{mk} Price [€/MWh]",
                    line=dict(width=2, color=color),
                ),
                row=1,
                col=1,
            )

    # === Subplot 2: Net Power + SoC ===
    fig.add_trace(
        go.Scatter(
            x=dispatch.index,
            y=net_power,
            mode="lines",
            name="Net Power [kW]",
            line=dict(width=2),
            fill="tozeroy",
            fillcolor="rgba(65,105,225,0.2)",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=dispatch.index,
            y=soc_percent,
            mode="lines",
            name="State of Charge [%]",
            line=dict(width=3),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    # === Subplot 3: Market Positions als Bars ===
    for col in market_cols:
        inner = col[2:-3]       # "p_da_kw" -> "da"
        mk_code = inner.upper() # "DA", "ID", ...
        pretty_name = f"{mk_code} Position [kW]"

        values = dispatch[col]

        # Farben: Marktfarbe + Alpha nach Vorzeichen (Sell vs. Buy)
        colors = [
            _rgba(mk_code, 1.0 if v > 0 else 0.3) for v in values
        ]

        # Labels pro Balken → Buy / Sell / Neutral
        labels = ["Sell" if v > 0 else "Buy" if v < 0 else "Neutral" for v in values]

        fig.add_trace(
            go.Bar(
                x=dispatch.index,
                y=values,
                marker_color=colors,
                name=pretty_name,
                customdata=labels,
                hovertemplate="%{y:.1f} kW<br>%{customdata}",
            ),
            row=3,
            col=1,
        )

    # Achsentitel
    fig.update_yaxes(title_text="Price [€/MWh]", row=1, col=1)
    fig.update_yaxes(title_text="Net Power [kW]", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="State of Charge [%]", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Market Position [kW] (+Sell / –Buy)",row=3,col=1)

    fig.update_layout(
        title=title,
        xaxis=dict(title="Time"),
        template="plotly_white",
        height=950,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        barmode="relative",  # positive/negative Beiträge relativ
    )

    return fig


def plot_market_cashflows_plotly(
    dispatch: pd.DataFrame,
    prices_by_market: dict[str, pd.Series],
    *,
    timestep_hours: float,
    title: str = "Market Cashflows per Timestep",
) -> go.Figure:
    """
    Plot cashflows [€/timestep] per market as bars, plus cumulative profit line.

    - Bars: Cashflow je Markt und Zeitschritt (DA, ID, ...)
    - Line: Kumulierte Summe über alle Märkte
    """
    if not isinstance(dispatch.index, pd.DatetimeIndex):
        raise ValueError("Dispatch index must be a DatetimeIndex")

    dt = float(timestep_hours)

    market_cols = [
        c for c in dispatch.columns
        if c.startswith("p_")
        and c.endswith("_kw")
        and c not in ("p_ch_kw", "p_dis_kw")
    ]

    cf_df = pd.DataFrame(index=dispatch.index)

    # Cashflows pro Markt berechnen
    for col in market_cols:
        inner = col[2:-3]       # "p_da_kw" -> "da"
        mk_code = inner.upper() # "DA", "ID", ...
        pretty_name = f"{mk_code} Cashflow [€/step]"

        if mk_code not in prices_by_market:
            continue

        p_series = dispatch[col]                 # kW
        price_series = prices_by_market[mk_code] # €/kWh

        # auf gemeinsamen Index ausrichten
        p_series, price_series = p_series.align(price_series, join="inner")

        cf = price_series * p_series * dt        # €/step
        cf_df[pretty_name] = cf

    # Gesamt-Cashflow & kumulative Summe
    cf_df["Total Cashflow [€/step]"] = cf_df.sum(axis=1)
    cf_df["Cumulative Profit [€]"] = cf_df["Total Cashflow [€/step]"].cumsum()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        specs=[
            [{"secondary_y": False}],  # Cashflow-Bars
            [{"secondary_y": False}],  # Cumulative Profit
        ],
        subplot_titles=("Market Cashflows per Timestep", "Cumulative Profit"),
    )

    # Oben: NUR Markt-Cashflows als Bars (ohne Total)
    cash_cols = [
        c for c in cf_df.columns
        if c.endswith("Cashflow [€/step]") and not c.startswith("Total")
    ]
    for col in cash_cols:
        # Marktcode aus Spaltennamen holen ("DA Cashflow ...")
        mk_code = col.split()[0]  # "DA", "ID", ...
        base_color = _rgb(mk_code)

        fig.add_trace(
            go.Bar(
                x=cf_df.index,
                y=cf_df[col],
                name=col,
                marker_color=base_color,
            ),
            row=1,
            col=1,
        )

    # Unten: kumulativer Gewinn als Linie
    fig.add_trace(
        go.Scatter(
            x=cf_df.index,
            y=cf_df["Cumulative Profit [€]"],
            mode="lines",
            name="Cumulative Profit [€]",
            line=dict(width=3, color="black"),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Cashflow [€/step]", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Profit [€]", row=2, col=1)

    fig.update_layout(
        title=title,
        xaxis=dict(title="Time"),
        template="plotly_white",
        height=900,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        barmode="relative",
    )

    return fig
