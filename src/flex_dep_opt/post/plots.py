from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flex_dep_opt.post.metrics import infer_market_position_columns, has_imbalance


# =============================================================================
# Color conventions
# =============================================================================
MARKET_COLORS: Dict[str, Tuple[int, int, int]] = {
    "DA": (0, 101, 189),
    "ID": (227, 114, 34),
    "IMB": (120, 80, 160),
    "FCR": (44, 160, 44),
}

FCR_GREEN = "rgb(44,160,44)"
FCR_GREEN_LIGHT = "rgba(44,160,44,0.12)"
FCR_GREEN_MID = "rgba(44,160,44,0.25)"

def _expand_slots_to_step(
    slot_series: pd.Series,
    dense_index: pd.DatetimeIndex,
    slot_duration: pd.Timedelta = pd.Timedelta(hours=4),
) -> pd.Series:
    result = pd.Series(0.0, index=dense_index)
    for slot_start, val in slot_series.items():
        slot_end = slot_start + slot_duration
        mask = (dense_index >= slot_start) & (dense_index < slot_end)
        result.loc[mask] = val
    return result


def _rgb(market: str) -> str:
    """Return 'rgb(r,g,b)' for a market code."""
    r, g, b = MARKET_COLORS.get(market, (100, 100, 100))
    return f"rgb({r},{g},{b})"


def _rgba(market: str, alpha: float) -> str:
    """Return 'rgba(r,g,b,a)' for a market code and transparency alpha."""
    r, g, b = MARKET_COLORS.get(market, (100, 100, 100))
    return f"rgba({r},{g},{b},{alpha})"


def _build_two_level_sunburst(data: list[tuple[str, str, float]]):
    """
    Build a 2-level sunburst hierarchy from tuples (market, side, value>=0).

    Hierarchy:
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

            alpha = 1.0 if side in {"Buy", "Cost"} else 0.3
            colors.append(_rgba(mk, alpha))

    return labels, parents, values, colors


# =============================================================================
# Plot 1: cashflows + sunbursts + KPI table
# =============================================================================
def plot_market_cashflows_plotly(
    *,
    cf_df: pd.DataFrame,
    energy_data: list[tuple[str, str, float]],
    cash_data: list[tuple[str, str, float]],
    kpis: Mapping[str, float | int],
    title: str = "Market Cashflows",
) -> go.Figure:
    """
    Create a cashflow report figure from *precomputed* postprocessing outputs.

    Parameters
    ----------
    cf_df:
        DataFrame as produced by `compute_cashflows_per_step(...)`.
    energy_data / cash_data:
        Sunburst tuples (mk, side, value>=0).
    kpis:
        KPI dict as produced by `compute_kpis(...)`.
    """
    if not isinstance(cf_df.index, pd.DatetimeIndex):
        raise ValueError("cf_df index must be a DatetimeIndex")

    e_labels, e_parents, e_values, e_colors = _build_two_level_sunburst(energy_data)
    c_labels, c_parents, c_values, c_colors = _build_two_level_sunburst(cash_data)

    fig = make_subplots(
        rows=3,
        cols=3,
        shared_xaxes=True,
        vertical_spacing=0.07,
        horizontal_spacing=0.06,
        specs=[
            [{"colspan": 3}, None, None],
            [{"colspan": 3}, None, None],
            [{"type": "sunburst"}, {"type": "sunburst"}, {"type": "table"}],
        ],
        subplot_titles=(
            "Market Cashflows per Timestep",
            "Cumulative Profit",
            "Energy volumes by market (kWh)",
            "Cashflow volumes by market (€)",
            "KPIs",
        ),
    )

    # Row 1: Cashflow bars
    for col in cf_df.columns:
        if not col.endswith("Cashflow [€/step]") or col.startswith("Total"):
            continue

        mk = col.split()[0]
        values = cf_df[col]
        colors = [_rgba(mk, 1.0) if v > 0 else _rgba(mk, 0.3) for v in values]

        fig.add_trace(go.Bar(x=cf_df.index, y=values, name=col, marker_color=colors), row=1, col=1)

    # Row 2: Cumulative profit
    fig.add_trace(
        go.Scatter(
            x=cf_df.index,
            y=cf_df["Cumulative Profit [€]"],
            mode="lines",
            name="Cumulative Profit [€]",
            line=dict(width=3, color="rgb(162,173,0)"),
        ),
        row=2, col=1
    )

    # Row 3: Sunbursts
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
        row=3, col=1
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
        row=3, col=2
    )

    kpi_rows = [
        ("Gross Profit", f"{float(kpis['gross_profit_eur']):.1f} €"),
        ("      Trading Profit", f"{float(kpis['trading_profit_eur']):.1f} €"),
        ("      Trading Fees", f"{float(kpis['fees_eur']):.1f} €"),
        ("      Imbalance Cost", f"{float(kpis['imb_cost_eur']):.1f} €"),
    ]
    if "fcr_revenue_eur" in kpis:
        kpi_rows.append(("      FCR Revenue", f"{float(kpis['fcr_revenue_eur']):.1f} €"))
    if "fcr_slots_committed" in kpis:
        kpi_rows.append(("      FCR Slots Committed", f"{int(kpis['fcr_slots_committed']):d}"))
    if "fcr_avg_capacity_mw" in kpis:
        kpi_rows.append(("      FCR Avg Capacity", f"{float(kpis['fcr_avg_capacity_mw']):.2f} MW"))

    kpi_rows += [
        ("Number of Trades", f"{int(kpis['trade_steps']):d}"),
        ("Net Volume", f"{float(kpis['net_kwh']):.1f} kWh"),
        ("      Sell Volume", f"{float(kpis['sell_kwh']):.1f} kWh"),
        ("      Buy Volume", f"{float(kpis['buy_kwh']):.1f} kWh"),
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
        row=3, col=3
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=1150,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=80, b=60),
        barmode="relative",
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Cashflow [€/step]", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Profit [€]", row=2, col=1)
    fig.update_traces(insidetextorientation="radial", selector=dict(type="sunburst"))

    return fig


# =============================================================================
# Plot 2: MPC dispatch + bands + market positions
# =============================================================================
def plot_mpc_dispatch_plotly(
    dispatch: pd.DataFrame,
    prices_by_market: Mapping[str, pd.Series] | None = None,
    *,
    commit_df: pd.DataFrame | None = None,
    fcr_result: pd.DataFrame | None = None,
    title: str = "MPC Flexband Dispatch and Market Positions",
) -> go.Figure:
    """
    MPC Visualisierung für Flexband-Modell.

    Subplots
    --------
    1) Market prices (optional)
    2) Power band + p_net
    3) Energy band + E
    4) Market positions (bars) + optional imbalance position
    """
    if not isinstance(dispatch.index, pd.DatetimeIndex):
        raise ValueError("Dispatch index must be a DatetimeIndex")

    required = ["p_net_kw", "P_lower_kw", "P_upper_kw", "E_kWh", "E_lower_kWh", "E_upper_kWh"]
    missing = [c for c in required if c not in dispatch.columns]
    if missing:
        raise ValueError(f"Dispatch missing required columns for flexband plotting: {missing}")

    market_cols = infer_market_position_columns(dispatch)
    has_imb = has_imbalance(dispatch)
    has_used_flag = "used_rebap" in dispatch.columns

    has_fcr = "x_fcr_kw" in dispatch.columns

    # Commit times (for hover annotation)
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
                s = mk_rows.sort_values("current_time").drop_duplicates("delivery_time")
                commit_time_by_market[mk] = s.set_index("delivery_time")["current_time"]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        specs=[[{}], [{}], [{}], [{}]],
        subplot_titles=(
            "Market Prices",
            "Power Flexband (p_net within [P_lower, P_upper])",
            "Energy Flexband (E within [E_lower, E_upper])",
            "Market Positions",
        ),
    )

    # Row 1: Prices
    if prices_by_market is not None:
        for mk, s in prices_by_market.items():
            price_mwh = s * 1000.0
            is_imb_price = mk in {"IMB_POS", "IMB_NEG", "IMB"}

            fig.add_trace(
                go.Scatter(
                    x=price_mwh.index,
                    y=price_mwh.values,
                    mode="lines",
                    name=f"{mk} Price [€/MWh]",
                    line=dict(width=3, color=_rgb(mk if mk in MARKET_COLORS else "IMB")),
                    visible="legendonly" if is_imb_price else True,
                ),
                row=1, col=1
            )

    if fcr_result is not None and "fcr_price" in fcr_result.columns:
        fig.add_trace(
            go.Scatter(
                x=fcr_result.index,
                y=fcr_result["fcr_price"],
                mode="lines",
                name="FCR Price [€/MW per 4h]",
                line=dict(width=2, color=FCR_GREEN, dash="dot"),
                line_shape="hv",
                visible="legendonly",
                yaxis="y1",
            ),
            row=1, col=1
        )

    # Row 2: Power band + p_net
    fig.add_trace(
        go.Scatter(x=dispatch.index, y=dispatch["P_upper_kw"], mode="lines", name="P upper [kW]",
                   line=dict(width=1, color="rgba(0,0,0,0.5)")),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=dispatch.index, y=dispatch["P_lower_kw"], mode="lines", name="P lower [kW]",
                   fill="tonexty", line=dict(width=1, color="rgba(0,0,0,0.5)"),
                   fillcolor="rgba(0,0,0,0.08)"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=dispatch.index, y=dispatch["p_net_kw"], mode="lines", name="p_net [kW]",
                   line=dict(width=3, color="rgb(162,173,0)")),
        row=2, col=1
    )

    if has_fcr:
        if fcr_result is not None and "fcr_capacity_kWh" in fcr_result.columns:
            committed_slots = dispatch["x_fcr_kw"].replace(0.0, float("nan")).dropna()
            slot_vals = {}
            for slot in fcr_result.index:
                slot_end = slot + pd.Timedelta(hours=4)
                in_slot = committed_slots[
                    (committed_slots.index >= slot) & (committed_slots.index < slot_end)
                ]
                slot_vals[slot] = in_slot.max() if not in_slot.empty else 0.0
            x_fcr_slots = pd.Series(slot_vals)
        else:
            x_fcr_slots = dispatch["x_fcr_kw"].replace(0.0, float("nan")).dropna()

        x_fcr_dense = _expand_slots_to_step(x_fcr_slots, dispatch.index)

        headroom_upper = dispatch["p_net_kw"] + x_fcr_dense
        headroom_lower = dispatch["p_net_kw"] - x_fcr_dense

        fig.add_trace(
            go.Scatter(
                x=dispatch.index,
                y=headroom_upper,
                mode="lines",
                name="FCR headroom upper",
                line=dict(width=1, color=FCR_GREEN, dash="dot"),
                showlegend=True,
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=dispatch.index,
                y=headroom_lower,
                mode="lines",
                name="FCR headroom lower / window",
                fill="tonexty",
                fillcolor=FCR_GREEN_LIGHT,
                line=dict(width=1, color=FCR_GREEN, dash="dot"),
                hovertemplate=(
                    "FCR window: %{customdata[0]:.0f} → %{customdata[1]:.0f} kW<br>"
                    "Reserved: %{customdata[2]:.0f} kW<extra></extra>"
                ),
                customdata=list(zip(headroom_upper, headroom_lower, x_fcr_dense)),
            ),
            row=2, col=1
        )

    # Row 3: Energy band + E
    fig.add_trace(
        go.Scatter(x=dispatch.index, y=dispatch["E_upper_kWh"], mode="lines", name="E upper [kWh]",
                   line=dict(width=1, color="rgba(0,0,0,0.5)")),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=dispatch.index, y=dispatch["E_lower_kWh"], mode="lines", name="E lower [kWh]",
                   fill="tonexty", line=dict(width=1, color="rgba(0,0,0,0.5)"),
                   fillcolor="rgba(0,0,0,0.08)"),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=dispatch.index, y=dispatch["E_kWh"], mode="lines", name="E [kWh]",
                   line=dict(width=3, color="rgb(162,173,0)")),
        row=3, col=1
    )

    # Row 4: Market positions
    for col in market_cols:
        mk_code = col[2:-3].upper()
        values = dispatch[col]

        colors = [_rgba(mk_code, 0.3 if v < 0 else 1.0) for v in values]
        labels = ["Buy" if v > 0 else "Sell" if v < 0 else "Neutral" for v in values]

        if mk_code in commit_time_by_market:
            ct = commit_time_by_market[mk_code].reindex(dispatch.index)
            commit_times = ct.dt.strftime("%Y-%m-%d %H:%M").fillna("not committed")
        else:
            commit_times = pd.Series(["not committed"] * len(dispatch.index), index=dispatch.index)

        custom = pd.DataFrame({"side": labels, "commit_time": commit_times.values}).values

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

    if has_fcr:
        if "x_fcr_dense" not in dir():
            committed_slots = dispatch["x_fcr_kw"].replace(0.0, float("nan")).dropna()
            x_fcr_dense = _expand_slots_to_step(committed_slots, dispatch.index)

        fig.add_trace(
            go.Scatter(
                x=dispatch.index,
                y=x_fcr_dense,
                mode="lines",
                name="FCR committed [kW]",
                line=dict(width=2, color=FCR_GREEN),
                hovertemplate="FCR: %{y:.0f} kW<extra></extra>",
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=dispatch.index,
                y=-x_fcr_dense,
                mode="lines",
                name="FCR committed [-kW]",
                line=dict(width=2, color=FCR_GREEN, dash="dot"),
                showlegend=False,
            ),
            row=4, col=1
        )

    # Imbalance net position
    if has_imb:
        p_imb_net = dispatch["p_imb_pos_kw"] - dispatch["p_imb_neg_kw"]
        colors = [_rgba("IMB", 1.0 if v > 0 else 0.3) for v in p_imb_net]
        labels = ["Buy" if v > 0 else "Sell" if v < 0 else "Neutral" for v in p_imb_net]

        if has_used_flag:
            used_str = dispatch["used_rebap"].astype(bool).map(lambda x: "reBAP used" if x else "no reBAP").values
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
                y=p_imb_net,
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

    # Highlight reBAP usage
    if has_used_flag:
        used = dispatch["used_rebap"].astype(bool).fillna(False)
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


def plot_mpc_fcr_plotly(
    *,
    symmetric_limit: pd.DataFrame,
    fcr_grouped_capacity: pd.Series,
    fcr_result: pd.DataFrame,
    dispatch: pd.DataFrame | None = None,
    title: str = "FCR Capacity & Results",
) -> go.Figure:
    has_dispatch = dispatch is not None and "x_fcr_kw" in dispatch.columns

    dense_index = symmetric_limit.index

    cap_dense = _expand_slots_to_step(fcr_grouped_capacity / 1000.0, dense_index)

    price_dense = _expand_slots_to_step(fcr_result["fcr_price"], dense_index)

    potential_per_slot = (fcr_grouped_capacity / 1000.0) * fcr_result["fcr_price"]
    potential_dense = _expand_slots_to_step(potential_per_slot, dense_index)

    realised_mw: pd.Series | None = None
    realised_dense: pd.Series | None = None
    realised_revenue_dense: pd.Series | None = None

    if has_dispatch:
        committed_slots_sparse = dispatch["x_fcr_kw"].replace(0.0, float("nan")).dropna()
        slot_vals: dict[pd.Timestamp, float] = {}
        for slot in fcr_result.index:
            slot_end = slot + pd.Timedelta(hours=4)
            in_slot = committed_slots_sparse[
                (committed_slots_sparse.index >= slot) &
                (committed_slots_sparse.index < slot_end)
            ]
            slot_vals[slot] = float(in_slot.max()) if not in_slot.empty else 0.0

        realised_mw = pd.Series(slot_vals) / 1000.0
        realised_dense = _expand_slots_to_step(realised_mw, dense_index)

        revenue_per_slot = realised_mw * fcr_result["fcr_price"]
        realised_revenue_dense = _expand_slots_to_step(revenue_per_slot, dense_index)

    n_rows = 4 if has_dispatch else 3
    row_heights = [0.35, 0.22, 0.22, 0.21] if has_dispatch else [0.45, 0.3, 0.25]

    subplot_titles = [
        "Symmetric FCR capacity: available vs. committed",
        "FCR settlement price (Germany)",
        "FCR revenue: potential vs. realised",
    ]
    if has_dispatch:
        subplot_titles.append("DA + ID volume displaced by FCR commitment")

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Scatter(
            x=symmetric_limit.index,
            y=symmetric_limit["inst_symmetric_limit"] / 1000.0,
            name="Available (15min) [MW]",
            line=dict(color="rgba(150,150,150,0.45)", width=1, dash="dot"),
            hovertemplate="%{y:.3f} MW<extra>available 15m</extra>",
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dense_index,
            y=cap_dense,
            name="Slot capacity (+) [MW]",
            line=dict(color=FCR_GREEN, width=2),
            hovertemplate="%{y:.3f} MW<extra>slot max +</extra>",
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dense_index,
            y=-cap_dense,
            name="Slot capacity (-) [MW]",
            line=dict(color=FCR_GREEN, width=2),
            fill="tonexty",
            fillcolor=FCR_GREEN_LIGHT,
            showlegend=False,
            hovertemplate="%{y:.3f} MW<extra>slot max -</extra>",
        ),
        row=1, col=1
    )

    if realised_dense is not None:
        fig.add_trace(
            go.Scatter(
                x=dense_index,
                y=realised_dense,
                name="Committed [MW]",
                line=dict(color=FCR_GREEN, width=3),
                fill="tozeroy",
                fillcolor=FCR_GREEN_MID,
                hovertemplate="%{y:.3f} MW committed<extra></extra>",
            ),
            row=1, col=1
        )
        avg_avail = cap_dense[cap_dense > 0].mean() if (cap_dense > 0).any() else 0.0
        avg_comm = realised_dense[realised_dense > 0].mean() if (realised_dense > 0).any() else 0.0
        util_pct = 100.0 * avg_comm / avg_avail if avg_avail > 0 else 0.0
        fig.add_annotation(
            text=f"Avg utilisation: {util_pct:.1f}%",
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            showarrow=False,
            font=dict(size=12, color=FCR_GREEN),
            bgcolor="rgba(255,255,255,0.7)",
        )

    fig.add_trace(
        go.Scatter(
            x=dense_index,
            y=price_dense,
            name="Settlement price [€/MW]",
            line=dict(color="rgb(214,39,40)", width=2),
            fill="tozeroy",
            fillcolor="rgba(214,39,40,0.07)",
            hovertemplate="%{y:.2f} €/MW<extra></extra>",
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dense_index,
            y=potential_dense,
            name="Potential revenue [€/slot]",
            line=dict(color="rgba(214,39,40,0.4)", width=1),
            fill="tozeroy",
            fillcolor="rgba(214,39,40,0.07)",
            hovertemplate="%{y:.2f} €<extra>potential</extra>",
        ),
        row=3, col=1
    )
    if realised_revenue_dense is not None:
        fig.add_trace(
            go.Scatter(
                x=dense_index,
                y=realised_revenue_dense,
                name="Realised revenue [€/slot]",
                line=dict(color=FCR_GREEN, width=2),
                fill="tozeroy",
                fillcolor=FCR_GREEN_MID,
                hovertemplate="%{y:.2f} €<extra>realised</extra>",
            ),
            row=3, col=1
        )
        total_potential = float(potential_per_slot.sum())
        total_realised = float((realised_mw * fcr_result["fcr_price"]).sum())
        capture = f"{100*total_realised/total_potential:.1f}%" if total_potential > 0 else "n/a"
        fig.add_annotation(
            text=(
                f"Total potential: {total_potential:.0f} €  |  "
                f"Realised: {total_realised:.0f} €  |  "
                f"Capture: {capture}"
            ),
            xref="paper", yref="paper",
            x=0.01, y=0.01,
            showarrow=False,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.7)",
        )

    if has_dispatch:
        da_id_cols = [
            c for c in infer_market_position_columns(dispatch)
            if any(mk in c.upper() for mk in ("DA", "ID"))
        ]

        if da_id_cols:
            da_id_abs = dispatch[da_id_cols].abs().sum(axis=1)
            slot_da_id: dict[pd.Timestamp, float] = {}
            for slot in fcr_result.index:
                slot_end = slot + pd.Timedelta(hours=4)
                in_slot = da_id_abs[
                    (da_id_abs.index >= slot) & (da_id_abs.index < slot_end)
                ]
                slot_da_id[slot] = float(in_slot.mean()) if not in_slot.empty else 0.0

            da_id_dense = _expand_slots_to_step(pd.Series(slot_da_id), dense_index)

            fig.add_trace(
                go.Scatter(
                    x=dense_index,
                    y=da_id_dense,
                    name="DA+ID volume [avg kW]",
                    fill="tozeroy",
                    fillcolor="rgba(100,100,100,0.15)",
                    line=dict(color="rgba(100,100,100,0.5)", width=1),
                    hovertemplate="%{y:.1f} kW avg DA+ID<extra></extra>",
                ),
                row=4, col=1
            )

        if realised_dense is not None:
            fig.add_trace(
                go.Scatter(
                    x=dense_index,
                    y=realised_dense * 1000.0,
                    name="FCR committed [kW]",
                    line=dict(color=FCR_GREEN, width=2),
                    hovertemplate="%{y:.1f} kW FCR<extra></extra>",
                ),
                row=4, col=1
            )

    fig.update_yaxes(title_text="Capacity [MW]", row=1, col=1)
    fig.update_yaxes(title_text="Price [€/MW]", row=2, col=1)
    fig.update_yaxes(title_text="Revenue [€]", row=3, col=1)
    if has_dispatch:
        fig.update_yaxes(title_text="Volume [kW]", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=n_rows, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=300 * n_rows,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=80, b=60),
    )

    return fig