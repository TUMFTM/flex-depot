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
    #"DA": (228,0,69),
    #"ID": (145,185,0),
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
    E_kWh = dispatch["E_kWh"]
    soc_percent = (E_kWh / float(capacity_kwh)) * 100.0

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
            line=dict(width=3, color="rgb(162,173,0)"),
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


def plot_mpc_dispatch_plotly(
    dispatch: pd.DataFrame,
    prices_by_market: dict[str, pd.Series] | None = None,
    *,
    commit_df: pd.DataFrame | None = None,
    title: str = "MPC Flexband Dispatch and Market Positions",
) -> go.Figure:
    """
    MPC Visualisierung für Flexband-Modell:

      Subplot 1: Marktpreise
      Subplot 2: Powerband (P_lower/P_upper) + p_net
      Subplot 3: Energyband (E_lower/E_upper) + E
      Subplot 4: Marktpositionen pro Markt (Bars)
    """

    if not isinstance(dispatch.index, pd.DatetimeIndex):
        raise ValueError("Dispatch index must be a DatetimeIndex")

    required = ["p_net_kW", "P_lower_kW", "P_upper_kW", "E_kWh", "E_lower_kWh", "E_upper_kWh"]
    missing = [c for c in required if c not in dispatch.columns]
    if missing:
        raise ValueError(f"Dispatch missing required columns for flexband plotting: {missing}")

    # Market columns
    market_cols = [
        c for c in dispatch.columns
        if c.startswith("p_") and c.endswith("_kw")
    ]

    commit_time_by_market: dict[str, pd.Series] = {}

    if commit_df is not None and not commit_df.empty:
        tmp = commit_df.copy()
        tmp["delivery_time"] = pd.to_datetime(tmp["delivery_time"])
        tmp["current_time"] = pd.to_datetime(tmp["current_time"])
        if "market" in tmp.columns and "commit_now" in tmp.columns:
            for mk in tmp["market"].unique():
                mk_rows = tmp[(tmp["market"] == mk) & (tmp["commit_now"] == True)]
                if mk_rows.empty:
                    continue
                # one commit per delivery_time expected; take first if duplicates
                s = mk_rows.sort_values("current_time").drop_duplicates("delivery_time")
                commit_time_by_market[mk] = s.set_index("delivery_time")["current_time"]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        specs=[
            [{"secondary_y": False}],  # Prices
            [{"secondary_y": False}],  # Power band + p_net
            [{"secondary_y": False}],  # Energy band + E
            [{"secondary_y": False}],  # Market positions
        ],
        subplot_titles=(
            "Market Prices",
            "Power Flexband (p_net within [P_lower, P_upper])",
            "Energy Flexband (E within [E_lower, E_upper])",
            "Market Positions",
        ),
    )

    # --- Row 1: Prices ---
    if prices_by_market is not None:
        for mk, s in prices_by_market.items():
            price_mwh = s * 1000.0
            fig.add_trace(
                go.Scatter(
                    x=price_mwh.index,
                    y=price_mwh.values,
                    mode="lines",
                    name=f"{mk} Price [€/MWh]",
                    line=dict(width=3, color=_rgb(mk)),
                ),
                row=1, col=1
            )

    # --- Row 2: Power band + p_net ---
    fig.add_trace(
        go.Scatter(
            x=dispatch.index, y=dispatch["P_upper_kW"],
            mode="lines", name="P upper [kW]",
            line=dict(width=1, color="rgba(0,0,0,0.5)"),
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dispatch.index, y=dispatch["P_lower_kW"],
            mode="lines", name="P lower [kW]",
            fill="tonexty",
            line=dict(width=1, color="rgba(0,0,0,0.5)"),
            fillcolor="rgba(0,0,0,0.08)",
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dispatch.index, y=dispatch["p_net_kW"],
            mode="lines", name="p_net [kW]",
            line=dict(width=3, color="rgb(162,173,0)"),
        ),
        row=2, col=1
    )

    # --- Row 3: Energy band + E ---
    fig.add_trace(
        go.Scatter(
            x=dispatch.index, y=dispatch["E_upper_kWh"],
            mode="lines", name="E upper [kWh]",
            line=dict(width=1, color="rgba(0,0,0,0.5)"),
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dispatch.index, y=dispatch["E_lower_kWh"],
            mode="lines", name="E lower [kWh]",
            fill="tonexty",
            line=dict(width=1, color="rgba(0,0,0,0.5)"),
            fillcolor="rgba(0,0,0,0.08)",
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dispatch.index, y=dispatch["E_kWh"],
            mode="lines", name="E [kWh]",
            line=dict(width=3, color="rgb(162,173,0)"),
        ),
        row=3, col=1
    )

    # --- Row 4: Market positions ---
    for col in market_cols:
        inner = col[2:-3]
        mk_code = inner.upper()
        values = dispatch[col]

        colors = [_rgba(mk_code, 1.0 if v > 0 else 0.3) for v in values]
        labels = ["Sell" if v > 0 else "Buy" if v < 0 else "Neutral" for v in values]

        # Build commit time strings aligned to dispatch.index (delivery_time axis)
        commit_times = None
        if mk_code in commit_time_by_market:
            ct = commit_time_by_market[mk_code].reindex(dispatch.index)
            # Format nicely; NaT -> "not committed"
            commit_times = ct.dt.strftime("%Y-%m-%d %H:%M").fillna("not committed")
        else:
            commit_times = pd.Series(["not committed"] * len(dispatch.index), index=dispatch.index)

        custom = pd.DataFrame({
            "side": labels,
            "commit_time": commit_times.values,
        }).values

        fig.add_trace(
            go.Bar(
                x=dispatch.index,
                y=values,
                name=f"{mk_code} Position [kW]",
                marker_color=colors,
                customdata=custom,
                hovertemplate=(
                    "%{y:.1f} kW<br>"
                    "%{customdata[0]}<br>"
                    "Committed at: %{customdata[1]}<extra></extra>"
                ),
            ),
            row=4, col=1
        )

    fig.update_yaxes(title_text="Price [€/MWh]", row=1, col=1)
    fig.update_yaxes(title_text="Power [kW]", row=2, col=1)
    fig.update_yaxes(title_text="Energy [kWh]", row=3, col=1)
    fig.update_yaxes(title_text="Position [kW]", row=4, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=1200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=80, b=60),
        barmode="relative",
        hovermode="x unified",
    )

    return fig



# Presentation one-pager
def plot_mpc_onepager(
    dispatch: pd.DataFrame,
    prices_by_market: dict[str, pd.Series] | None = None,
    *,
    capacity_kwh: float,
    title: str = "MPC Dispatch and Market Positions",
) -> Dict[str, go.Figure]:
    """
    Erzeugt 4 einzelne Plots (je eine Figure), statt Subplots:

      - fig_prices:        Marktpreise (alle Märkte)
      - fig_power_pos:     Net Power + Marktpositionen
      - fig_soc:           State of Charge [%]
      - fig_empty:         leerer Platzhalter (z.B. für spätere Ergänzungen)

    Rückgabe: dict mit Schlüsseln {"prices", "power_positions", "soc", "empty"}
    """

    if not isinstance(dispatch.index, pd.DatetimeIndex):
        raise ValueError("Dispatch index must be a DatetimeIndex")

    # Net Power
    net_power = dispatch["p_dis_kw"] - dispatch["p_ch_kw"]

    # SoC [%]
    E_kWh = dispatch["E_kWh"]
    soc_percent = (E_kWh / float(capacity_kwh)) * 100.0

    # Marktspalten erkennen
    market_cols = [
        c for c in dispatch.columns
        if c.startswith("p_")
        and c.endswith("_kw")
        and c not in ("p_ch_kw", "p_dis_kw")
    ]

    # ---------- Helper für einheitlichen PPT-Style ----------
    def _style_for_ppt(fig: go.Figure, subtitle: str | None = None) -> go.Figure:
        fig.update_layout(
            title=subtitle,
            template="plotly_white",
            width=2000,
            height=1100,
            margin=dict(l=60, r=60, t=80, b=60),
            font=dict(size=45),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,  # Legend im PPT nachbauen, wenn gewünscht
        )
        fig.update_xaxes(showgrid=True, gridcolor="black", gridwidth=1)

        return fig

    # ---------- Plot 1: Preise ----------
    fig_prices = go.Figure()
    if prices_by_market is not None:
        for mk, s in prices_by_market.items():
            price_mwh = s * 1000.0
            color = _rgb(mk)

            fig_prices.add_trace(
                go.Scatter(
                    x=price_mwh.index,
                    y=price_mwh.values,
                    mode="lines",
                    name=f"{mk} Price [€/MWh]",
                    line=dict(width=5, color=color),
                )
            )



    # ---------- Plot 2: Net Power + Marktpositionen ----------
    fig_power_pos = go.Figure()

    # Bars je Markt
    for col in market_cols:
        inner = col[2:-3]   # p_da_kw -> da
        mk_code = inner.upper()
        pretty_name = f"{mk_code} Position [kW]"

        values = dispatch[col]
        labels = ["Sell" if v > 0 else "Buy" if v < 0 else "Neutral" for v in values]
        colors = [_rgba(mk_code, 1.0 if v > 0 else 0.3) for v in values]

        fig_power_pos.add_trace(
            go.Bar(
                x=dispatch.index,
                y=values,
                name=pretty_name,
                marker_color=colors,
                customdata=labels,
                hovertemplate="%{y:.1f} kW<br>%{customdata}",
            )
        )

    # Net Power als Linie obendrauf
    fig_power_pos.add_trace(
        go.Scatter(
            x=dispatch.index,
            y=net_power,
            mode="lines",
            name="Net Power [kW]",
            line=dict(width=4, color="rgb(106, 117, 126)"),
        )
    )



    # ---------- Plot 3: SoC ----------
    fig_soc = go.Figure()
    fig_soc.add_trace(
        go.Scatter(
            x=dispatch.index,
            y=soc_percent,
            mode="lines",
            name="State of Charge [%]",
            line=dict(width=8, color="rgb(228,0,69)"),
        )
    )


    # ========= Plot 4: Cashflows (Bars) + kumulative Summe (Linie) =========
    fig_cash = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"secondary_y": False}]],
    )

    cf_df = None
    if prices_by_market is not None:
        # Δt aus Index ableiten (h)
        if len(dispatch.index) >= 2:
            dt_seconds = (dispatch.index[1] - dispatch.index[0]).total_seconds()
            dt = dt_seconds / 3600.0
        else:
            dt = 1.0  # Fallback

        cf_df = pd.DataFrame(index=dispatch.index)

        # Cashflows pro Markt berechnen
        for col in market_cols:
            inner = col[2:-3]  # "p_da_kw" -> "da"
            mk_code = inner.upper()  # "DA", "ID", ...
            pretty_name = f"{mk_code} Cashflow [€/step]"

            if mk_code not in prices_by_market:
                continue

            p_series = dispatch[col]  # kW
            price_series = prices_by_market[mk_code]  # €/kWh

            # auf gemeinsamen Index ausrichten
            p_series, price_series = p_series.align(price_series, join="inner")

            cf = price_series * p_series * dt  # €/step
            cf_df[pretty_name] = cf

        # Gesamt-Cashflow & kumulative Summe
        if not cf_df.empty:
            cf_df["Total Cashflow [€/step]"] = cf_df.sum(axis=1)
            cf_df["Cumulative Profit [€]"] = cf_df["Total Cashflow [€/step]"].cumsum()

            # Bars: Markt-Cashflows
            cash_cols = [
                c for c in cf_df.columns
                if c.endswith("Cashflow [€/step]") and not c.startswith("Total")
            ]
            for col in cash_cols:
                mk_code = col.split()[0]  # "DA", "ID", ...
                base_color = _rgb(mk_code)

                #fig_cash.add_trace(
                #    go.Bar(
                #        x=cf_df.index,
                #        y=cf_df[col],
                #        name=col,
                #        marker_color=base_color,
                #    ),
                #    row=1,
                #    col=1,
                #    secondary_y=True,  # linke Achse
                #)

            # Linie: kumulativer Profit (rechte Achse)
            fig_cash.add_trace(
                go.Scatter(
                    x=cf_df.index,
                    y=cf_df["Cumulative Profit [€]"],
                    mode="lines",
                    name="Cumulative Profit [€]",
                    line=dict(width=8, color="rgb(228,0,69)"),
                ),
                row=1,
                col=1,
                secondary_y=False,  # rechte Achse
            )


    # ---------- Layout ----------

    fig_prices.update_yaxes(title_text=None, range=[0, 200], showgrid=True, gridcolor="black", gridwidth=1,)
    fig_prices.update_xaxes(title_text=None)
    _style_for_ppt(fig_prices, subtitle=None)

    fig_power_pos.update_yaxes(title_text=None, range=[-500, 500], showgrid=True, gridcolor="black", gridwidth=1,)
    fig_power_pos.update_xaxes(title_text=None)
    _style_for_ppt(fig_power_pos, subtitle=None)

    fig_soc.update_yaxes(title_text=None, range=[0, 100], showgrid=True, gridcolor="black", gridwidth=1,)
    fig_soc.update_xaxes(title_text=None)
    _style_for_ppt(fig_soc, subtitle=None)

    fig_cash.update_xaxes(title_text=None)
    fig_cash.update_yaxes(title_text="", range=[0, 400], showgrid=True, gridcolor="black", gridwidth=1, secondary_y=False)
    #fig_cash.update_yaxes(title_text=None,range=[-30, 30], showgrid=False, secondary_y=True)
    fig_cash.update_xaxes(title_text=None)
    _style_for_ppt(fig_cash, subtitle=None)

    return {
        "prices": fig_prices,
        "power_positions": fig_power_pos,
        "soc": fig_soc,
        "cash": fig_cash,
    }
