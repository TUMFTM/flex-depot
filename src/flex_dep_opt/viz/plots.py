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
    "IMB": (120, 80, 160),
}


def _rgb(market: str) -> str:
    """Hilfsfunktion: rgb()-String aus MARKET_COLORS."""
    r, g, b = MARKET_COLORS.get(market, (100, 100, 100))
    return f"rgb({r},{g},{b})"


def _rgba(market: str, alpha: float) -> str:
    """Hilfsfunktion: rgba()-String aus MARKET_COLORS + Alpha."""
    r, g, b = MARKET_COLORS.get(market, (100, 100, 100))
    return f"rgba({r},{g},{b},{alpha})"


def plot_market_cashflows_plotly(
    dispatch: pd.DataFrame,
    commit: pd.DataFrame,
    prices_by_market: dict[str, pd.Series],
    fee_eur_per_kwh_by_market: dict[str, float],
    *,
    timestep_hours: float,
    title: str = "Market Cashflows",
) -> go.Figure:
    """
    HTML Layout (3 Zeilen):
      Row 1 (full width): Market cashflows per timestep (stacked bars)
      Row 2 (full width): Cumulative profit (line)
      Row 3 (3 columns): Energy sunburst | Cashflow sunburst | KPI table
    """

    if not isinstance(dispatch.index, pd.DatetimeIndex):
        raise ValueError("Dispatch index must be a DatetimeIndex")

    dt = float(timestep_hours)

    market_cols = [
        c for c in dispatch.columns
        if c.startswith("p_")
        and c.endswith("_kw")
        and c not in ("p_ch_kw", "p_dis_kw", "p_net_kw", "p_imb_pos_kw", "p_imb_neg_kw")
    ]

    # --- Imbalance / reBAP columns (optional) ---
    has_imb = ("p_imb_pos_kw" in dispatch.columns) and ("p_imb_neg_kw" in dispatch.columns)
    p_imb_net = (dispatch["p_imb_pos_kw"] - dispatch["p_imb_neg_kw"]) if has_imb else None

    # =========================================================
    # 1) Cashflow time series
    # =========================================================
    cf_df = pd.DataFrame(index=dispatch.index)

    for col in market_cols:
        mk = col[2:-3].upper()
        if mk not in prices_by_market:
            continue

        p = dispatch[col]
        price = prices_by_market[mk]
        p, price = p.align(price, join="inner")
        if p.empty:
            continue

        cf_df[f"{mk} Cashflow [€/step]"] = (-price * p) * dt

    # =========================================================
    # 1b) Add Imbalance cashflow (optional)
    # =========================================================
    if has_imb and ("IMB_POS" in prices_by_market) and ("IMB_NEG" in prices_by_market):
        # lokale Kopien (damit garantiert nichts am dispatch “klebt”)
        p_pos = dispatch["p_imb_pos_kw"].astype(float).copy()
        p_neg = dispatch["p_imb_neg_kw"].astype(float).copy()

        p_pos_mag = p_pos.clip(lower=0.0)  # oder: p_pos.abs()
        p_neg_mag = p_neg.clip(lower=0.0)  # oder: p_neg.abs()

        pos_price = prices_by_market["IMB_POS"]
        neg_price = prices_by_market["IMB_NEG"]

        # gemeinsamer Index
        idx = dispatch.index.intersection(pos_price.index).intersection(neg_price.index)
        if len(idx) > 0:
            p_pos_mag = p_pos_mag.reindex(idx).fillna(0.0)
            p_neg_mag = p_neg_mag.reindex(idx).fillna(0.0)
            pos_price = pos_price.reindex(idx)
            neg_price = neg_price.reindex(idx)

            # IMB immer als Kosten (negativ)
            cf_df["IMB Cashflow [€/step]"] = - (pos_price * p_pos_mag + neg_price * p_neg_mag) * dt


    if cf_df.empty:
        raise ValueError("No cashflows could be computed (check market columns and prices_by_market).")

    cf_df["Total Cashflow [€/step]"] = cf_df.sum(axis=1)
    cf_df["Cumulative Profit [€]"] = cf_df["Total Cashflow [€/step]"].cumsum()

    # =========================================================
    # 2) Aggregates for sunbursts + KPIs
    # =========================================================
    # sunburst raw lists (mk, side, value>=0)
    energy_data: list[tuple[str, str, float]] = []
    cash_data: list[tuple[str, str, float]] = []

    # for KPI fees + table
    energy_by_mk: dict[str, tuple[float, float]] = {}  # mk -> (buy_kwh, sell_kwh)
    cash_by_mk: dict[str, tuple[float, float]] = {}    # mk -> (buy_eur, sell_eur)

    for col in market_cols:
        mk = col[2:-3].upper()
        if mk not in prices_by_market:
            continue

        p = dispatch[col]
        price = prices_by_market[mk]
        p, price = p.align(price, join="inner")
        if p.empty:
            continue

        energy_kwh = p * dt
        cash_eur = (-price * p) * dt

        buy_mask = p > 0
        sell_mask = p < 0

        buy_e = float((energy_kwh[buy_mask]).sum()) if buy_mask.any() else 0.0
        sell_e = float((-energy_kwh[sell_mask]).sum()) if sell_mask.any() else 0.0
        buy_c = float((-cash_eur[buy_mask]).sum()) if buy_mask.any() else 0.0  # positive cost magnitude
        sell_c = float((cash_eur[sell_mask]).sum()) if sell_mask.any() else 0.0  # positive revenue magnitude

        energy_by_mk[mk] = (buy_e, sell_e)
        cash_by_mk[mk] = (buy_c, sell_c)

        if buy_e > 0:
            energy_data.append((mk, "Buy", buy_e))
        if sell_e > 0:
            energy_data.append((mk, "Sell", sell_e))

        if buy_c > 0:
            cash_data.append((mk, "Buy", buy_c))
        if sell_c > 0:
            cash_data.append((mk, "Sell", sell_c))

    # =========================================================
    # 2b) Add Imbalance aggregates for sunbursts + KPIs (IMB always cost)
    # =========================================================
    if has_imb and ("IMB_POS" in prices_by_market) and ("IMB_NEG" in prices_by_market):
        p_pos = dispatch["p_imb_pos_kw"].astype(float)
        p_neg = dispatch["p_imb_neg_kw"].astype(float)

        pos_price = prices_by_market["IMB_POS"]
        neg_price = prices_by_market["IMB_NEG"]

        idx = dispatch.index.intersection(pos_price.index).intersection(neg_price.index)
        if len(idx) > 0:
            p_pos = p_pos.reindex(idx).fillna(0.0)
            p_neg = p_neg.reindex(idx).fillna(0.0)
            pos_price = pos_price.reindex(idx)
            neg_price = neg_price.reindex(idx)

            # energy (kWh): direction can still be represented if you want:
            # net power sign: p_net = p_pos - p_neg
            p_net = (p_pos - p_neg)

            energy_kwh_net = p_net * dt

            buy_mask = energy_kwh_net > 0
            sell_mask = energy_kwh_net < 0

            buy_e = float((energy_kwh_net[buy_mask]).sum()) if buy_mask.any() else 0.0
            sell_e = float((-energy_kwh_net[sell_mask]).sum()) if sell_mask.any() else 0.0

            # cash (€): ALWAYS a penalty (negative), but sunburst expects positive magnitudes
            imb_cost_eur = float((pos_price * p_pos + neg_price * p_neg).sum() * dt)  # positive magnitude

            # store for KPIs
            energy_by_mk["IMB"] = (buy_e, sell_e)
            cash_by_mk["IMB"] = (imb_cost_eur, 0.0)  # treat as Buy(cost), no Sell

            # sunburst raw lists (values >= 0)
            if buy_e > 0:
                energy_data.append(("IMB", "Buy", buy_e))
            if sell_e > 0:
                energy_data.append(("IMB", "Sell", sell_e))

            if imb_cost_eur > 0:
                cash_data.append(("IMB", "Cost", imb_cost_eur))  # always Buy (cost)

    # =========================================================
    # 3) KPI computation
    # =========================================================
    buy_kwh = float(sum(v[0] for mk, v in energy_by_mk.items() if mk != "IMB"))
    sell_kwh = float(sum(v[1] for mk, v in energy_by_mk.items() if mk != "IMB"))
    net_kwh = sell_kwh - buy_kwh

    # fees only for DA/ID
    def _mk_fee(mk: str) -> float:
        if mk not in energy_by_mk:
            return 0.0
        buy_e, sell_e = energy_by_mk[mk]
        return float(fee_eur_per_kwh_by_market.get(mk, 0.0)) * float(buy_e + sell_e)
    fees_eur = -(_mk_fee("DA") + _mk_fee("ID"))

    trade_steps = 0
    if commit is not None and not commit.empty:
        # same logic as plots_paper.py
        if ("committed_new" in commit.columns) and ("commit_now" in commit.columns):
            trade_steps = int(commit[(commit["committed_new"] != 0.0) & (commit["commit_now"] == True)].shape[0])

    # Imbalance cost
    imb_cost_eur = 0.0
    if "IMB Cashflow [€/step]" in cf_df.columns:
        # cashflow is negative -> cost magnitude is -sum
        imb_cost_eur = -float(cf_df["IMB Cashflow [€/step]"].sum())

    # Trading profit
    gross_profit_eur = float(cf_df["Total Cashflow [€/step]"].sum()) + fees_eur
    trading_profit_eur = gross_profit_eur - fees_eur - imb_cost_eur

    # =========================================================
    # 4) Sunburst builder (same hierarchy idea as before)
    # =========================================================
    def _build_sunburst(data: list[tuple[str, str, float]]):
        """
        data: list of (mk, side, value) with side in {"Buy","Sell"} and value >= 0
        Returns labels, parents, values, colors for a 2-level sunburst:
          Total -> MK -> MK Side
        """
        total = sum(v for _, _, v in data)
        labels = ["Total"]
        parents = [""]
        values = [total]
        colors = ["rgba(200,200,200,0.0)"]  # root almost invisible

        markets = sorted({mk for mk, _, _ in data})
        for mk in markets:
            mk_total = sum(v for m, _, v in data if m == mk)
            if mk_total <= 0:
                continue

            labels.append(mk)
            parents.append("Total")
            values.append(mk_total)
            colors.append(_rgba(mk, 0.7))

            for side in sorted({s for m, s, _ in data if m == mk}):
                side_total = sum(v for m, s, v in data if m == mk and s == side)
                if side_total <= 0:
                    continue

                labels.append(f"{mk} {side}")
                parents.append(mk)
                values.append(side_total)

                alpha = 1.0 if side == "Sell" else 0.3
                colors.append(_rgba(mk, alpha))

        return labels, parents, values, colors

    e_labels, e_parents, e_values, e_colors = _build_sunburst(energy_data)
    c_labels, c_parents, c_values, c_colors = _build_sunburst(cash_data)

    # =========================================================
    # 5) Figure layout (3 rows, 3 cols; rows 1/2 span all cols)
    # =========================================================
    fig = make_subplots(
        rows=3,
        cols=3,
        shared_xaxes=True,
        vertical_spacing=0.07,
        horizontal_spacing=0.06,
        specs=[
            [{"colspan": 3}, None, None],                 # Row 1: Bars
            [{"colspan": 3}, None, None],                 # Row 2: Cumulative line
            [{"type": "sunburst"}, {"type": "sunburst"}, {"type": "table"}],  # Row 3: 3 columns
        ],
        subplot_titles=(
            "Market Cashflows per Timestep",
            "Cumulative Profit",
            "Energy volumes by market (kWh)",
            "Cashflow volumes by market (€)",
            "KPIs",
        ),
    )

    # ---------------------------------------------------------
    # Row 1: Bars (per market cashflow)
    # ---------------------------------------------------------
    for col in cf_df.columns:
        if not col.endswith("Cashflow [€/step]") or col.startswith("Total"):
            continue

        mk = col.split()[0]
        values = cf_df[col]

        colors = [_rgba(mk, 1.0) if v > 0 else _rgba(mk, 0.3) for v in values]

        fig.add_trace(
            go.Bar(
                x=cf_df.index,
                y=values,
                name=col,
                marker_color=colors,
            ),
            row=1,
            col=1,
        )

    # ---------------------------------------------------------
    # Row 2: Cumulative profit (line)
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Row 3: Sunbursts + KPI table
    # ---------------------------------------------------------
    fig.add_trace(
        go.Sunburst(
            labels=e_labels,
            parents=e_parents,
            values=e_values,
            marker=dict(colors=e_colors),
            branchvalues="total",
            name="Energy",
            hovertemplate="%{label}<br>%{value:.2f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Sunburst(
            labels=c_labels,
            parents=c_parents,
            values=c_values,
            marker=dict(colors=c_colors),
            branchvalues="total",
            name="Cashflow",
            hovertemplate="%{label}<br>%{value:.2f}<extra></extra>",
        ),
        row=3,
        col=2,
    )

    kpi_rows = [
        ("Gross Profit", f"{gross_profit_eur:.1f} €"),
        ("      Trading Profit", f"{trading_profit_eur:.1f} €"),
        ("      Trading Fees", f"{fees_eur:.1f} €"),
        ("      Imbalance Cost", f"{imb_cost_eur:.1f} €"),
        ("Number of Trades", f"{trade_steps:d}"),
        ("Net Volume", f"{net_kwh:.1f} kWh"),
        ("      Sell Volume", f"{sell_kwh:.1f} kWh"),
        ("      Buy Volume", f"{buy_kwh:.1f} kWh"),
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=["KPI", "Value"],
                align=["left", "right"],
                fill_color="rgba(0,0,0,0.03)",
                line_color="rgba(0,0,0,0.15)",
                font=dict(size=12),
            ),
            cells=dict(
                values=[[r[0] for r in kpi_rows], [r[1] for r in kpi_rows]],
                align=["left", "right"],
                line_color="rgba(0,0,0,0.15)",
                height=24,
                font=dict(size=12),
            ),
        ),
        row=3,
        col=3,
    )

    # =========================================================
    # Layout tweaks
    # =========================================================
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=1150,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        barmode="relative",
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Cashflow [€/step]", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Profit [€]", row=2, col=1)

    # Sunbursts: etwas weniger “Plotly padding” (optional)
    fig.update_traces(insidetextorientation="radial", selector=dict(type="sunburst"))

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

    required = ["p_net_kw", "P_lower_kw", "P_upper_kw", "E_kWh", "E_lower_kWh", "E_upper_kWh"]
    missing = [c for c in required if c not in dispatch.columns]
    if missing:
        raise ValueError(f"Dispatch missing required columns for flexband plotting: {missing}")

    # Market columns
    market_cols = [
        c for c in dispatch.columns
        if c.startswith("p_")
           and c.endswith("_kw")
           and c not in ("p_ch_kw", "p_dis_kw", "p_net_kw", "p_imb_pos_kw", "p_imb_neg_kw")
    ]

    # --- Imbalance / reBAP columns (optional) ---
    has_imb = ("p_imb_pos_kw" in dispatch.columns) and ("p_imb_neg_kw" in dispatch.columns)
    if has_imb:
        p_imb_net = dispatch["p_imb_pos_kw"] - dispatch["p_imb_neg_kw"]
    else:
        p_imb_net = None
    has_used_flag = "used_rebap" in dispatch.columns

    # Commit times
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
            # Default visibility: hide reBAP/imbalance prices until legend click
            is_imb_price = mk in {"IMB_POS", "IMB_NEG", "IMB"}  # robust for your naming

            fig.add_trace(
                go.Scatter(
                    x=price_mwh.index,
                    y=price_mwh.values,
                    mode="lines",
                    name=f"{mk} Price [€/MWh]",
                    line=dict(width=3, color=_rgb(mk)),
                    visible="legendonly" if is_imb_price else True,
                ),
                row=1, col=1
            )

    # --- Row 2: Power band + p_net ---
    fig.add_trace(
        go.Scatter(
            x=dispatch.index, y=dispatch["P_upper_kw"],
            mode="lines", name="P upper [kW]",
            line=dict(width=1, color="rgba(0,0,0,0.5)"),
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dispatch.index, y=dispatch["P_lower_kw"],
            mode="lines", name="P lower [kW]",
            fill="tonexty",
            line=dict(width=1, color="rgba(0,0,0,0.5)"),
            fillcolor="rgba(0,0,0,0.08)",
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dispatch.index, y=dispatch["p_net_kw"],
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

        colors = [_rgba(mk_code, 1.0 if v < 0 else 0.3) for v in values]  # optional: highlight Sell
        labels = ["Buy" if v > 0 else "Sell" if v < 0 else "Neutral" for v in values]

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

    # --- Row 4b: Imbalance (reBAP) net position ---
    if has_imb:
        values = p_imb_net

        colors = [_rgba("IMB", 1.0 if v > 0 else 0.3) for v in values]
        labels = ["Buy" if v > 0 else "Sell" if v < 0 else "Neutral" for v in values]

        if has_used_flag:
            used = dispatch["used_rebap"].astype(bool)
            used_str = used.map(lambda x: "reBAP used" if x else "no reBAP").values
        else:
            used_str = ["reBAP unknown"] * len(dispatch)

        custom = pd.DataFrame({
            "side": labels,
            "used": used_str,
            "pos": dispatch["p_imb_pos_kw"].values,
            "neg": dispatch["p_imb_neg_kw"].values,
        }).values

        fig.add_trace(
            go.Bar(
                x=dispatch.index,
                y=values,
                name="IMB (reBAP) Net [kW]",
                marker_color=colors,
                customdata=custom,
                hovertemplate=(
                    "IMB net: %{y:.1f} kW<br>"
                    "%{customdata[0]}<br>"
                    "%{customdata[1]}<br>"
                    "pos: %{customdata[2]:.1f} kW<br>"
                    "neg: %{customdata[3]:.1f} kW"
                    "<extra></extra>"
                ),
            ),
            row=4, col=1
        )

    # --- Optional: highlight periods where reBAP was used ---
    if has_used_flag:
        used = dispatch["used_rebap"].astype(bool).fillna(False)

        # find contiguous True segments
        start = None
        for ts, flag in used.items():
            if flag and start is None:
                start = ts
            if (not flag) and start is not None:
                end = ts
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor="rgba(120,80,160,0.12)",
                    line_width=0,
                    layer="below",
                    row=4, col=1
                )
                start = None

        # handle if ends with True
        if start is not None:
            fig.add_vrect(
                x0=start, x1=dispatch.index[-1],
                fillcolor="rgba(120,80,160,0.12)",
                line_width=0,
                layer="below",
                row="all", col=1
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

