from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flex_dep_opt.io.prices import FCR_DROOP_COL
from flex_dep_opt.post.metrics import has_imbalance, infer_market_position_columns

# =============================================================================
# Color conventions
# =============================================================================
MARKET_COLORS: dict[str, tuple[int, int, int]] = {
    "DA": (0, 101, 189),
    "ID": (227, 114, 34),
    "IMB": (120, 80, 160),
    "FCR": (44, 160, 44),
}

FCR_GREEN = "rgb(44,160,44)"
FCR_GREEN_LIGHT = "rgba(44,160,44,0.12)"
FCR_GREEN_MID = "rgba(44,160,44,0.25)"
FCR_RED = "rgb(214,39,40)"
FCR_RED_LIGHT = "rgba(214,39,40,0.12)"


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
    reference_df: pd.DataFrame | None = None,
    reference_summary: Mapping[str, float] | None = None,
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
    reference_df / reference_summary:
        Optional static-price reference scenario as produced by
        `compute_reference_driving_energy_costs(...)`.
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
        row=2,
        col=1,
    )

    if reference_df is not None and reference_summary is not None:
        if not isinstance(reference_df.index, pd.DatetimeIndex):
            raise ValueError("reference_df index must be a DatetimeIndex")
        required_col = "Cumulative Reference Energy Cost [EUR]"
        if required_col not in reference_df.columns:
            raise ValueError(f"reference_df must contain '{required_col}'")

        fig.add_trace(
            go.Scatter(
                x=reference_df.index,
                y=-reference_df[required_col],
                mode="lines",
                name="Reference Scenario [€]",
                line=dict(width=2, color="rgb(0,0,0)", dash="dash"),
                visible="legendonly",
                hovertemplate="%{y:.2f} €<extra></extra>",
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

    # Indented rows decompose Gross Profit:
    #   Gross = Trading Profit + FCR Revenue + Trading Fees + Imbalance Cost
    kpi_rows = [
        ("Flex-Depot Optimization (V2G & Driving)", ""),
        ("Gross Profit", f"{float(kpis['gross_profit_eur']):.1f} €"),
        ("      Trading Profit", f"{float(kpis['trading_profit_eur']):.1f} €"),
    ]
    if "fcr_revenue_eur" in kpis:
        kpi_rows.append(("      FCR Revenue", f"{float(kpis['fcr_revenue_eur']):.1f} €"))
    kpi_rows += [
        ("      Trading Fees", f"{float(kpis['fees_eur']):.1f} €"),
        ("      Imbalance Cost", f"{float(kpis['imb_cost_eur']):.1f} €"),
    ]
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

    if reference_summary is not None:
        reference_gross_profit_eur = -float(reference_summary["ref_energy_cost_eur"])
        total_potential_eur = float(kpis["gross_profit_eur"]) - reference_gross_profit_eur
        kpi_rows.extend([
            ("", ""),
            ("Reference Scenario (Driving only)", ""),
            ("Gross Profit", f"{reference_gross_profit_eur:.1f} €"),
            ("Driving Energy", f"{float(reference_summary['ref_driving_energy_kwh']):.1f} kWh"),
            ("Static Price", f"{float(reference_summary['ref_static_price_eur_per_kwh']):.3f} €/kWh"),
            ("", ""),
            ("Total Potential Flex-Depot Optimization", ""),
            ("Gross Profit Delta", f"{total_potential_eur:.1f} €"),
        ])

    section_labels = {
        "Flex-Depot Optimization (V2G & Driving)",
        "Reference Scenario (Driving only)",
        "Total Potential Flex-Depot Optimization",
    }
    section_rows = {
        i for i, (label, value) in enumerate(kpi_rows)
        if label in section_labels and value == ""
    }
    fill_colors = [
        [
            "rgba(0,0,0,0.06)" if i in section_rows else "rgba(255,255,255,1.0)"
            for i in range(len(kpi_rows))
        ],
        [
            "rgba(0,0,0,0.06)" if i in section_rows else "rgba(255,255,255,1.0)"
            for i in range(len(kpi_rows))
        ],
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
                fill_color=fill_colors,
                line_color="rgba(0,0,0,0.15)",
                height=24,
                font=dict(size=12),
            ),
        ),
        row=3,
        col=3,
    )

    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", y=0.99, yanchor="top", yref="container"),
        template="plotly_white",
        height=1150,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=150, b=60),
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
    fcr_commit_df: pd.DataFrame | None = None,
    title: str = "MPC Flexband Dispatch and Market Positions",
    fcr_frequency_data: pd.DataFrame | None = None,
    fcr_product_hours: float = 4.0,
) -> go.Figure:
    """
    MPC flexband dispatch visualisation.

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

    # All FCR series come pre-computed from extract_dispatch:
    #   - x_fcr_kw   : per-step bid power (kW, slot value broadcast across steps)
    #   - fcr_droop  : per-step droop signal in [-1, +1] (signed; + = up-reg)
    #   - p_droop_kw : per-step activation power (kW, import-positive, matches p_net)
    has_fcr = "x_fcr_kw" in dispatch.columns
    x_fcr_dense: pd.Series | None = dispatch["x_fcr_kw"].astype(float) if has_fcr else None
    p_droop_kw: pd.Series | None = (
        dispatch["p_droop_kw"].astype(float) if has_fcr and "p_droop_kw" in dispatch.columns else None
    )

    has_freq_subplot = has_fcr and fcr_frequency_data is not None and not fcr_frequency_data.empty
    freq_resampled: pd.Series | None = None

    if has_freq_subplot:
        freq_df = fcr_frequency_data.copy()
        if not isinstance(freq_df.index, pd.DatetimeIndex):
            has_freq_subplot = False
        else:
            droop_col = (
                FCR_DROOP_COL
                if FCR_DROOP_COL in freq_df.columns
                else next((c for c in freq_df.columns if "DROOP" in c.upper()), None)
            )
            if droop_col is None:
                has_freq_subplot = False
            else:
                freq_series = freq_df[droop_col].sort_index()
                aligned_idx = dispatch.index
                locs = freq_series.index.get_indexer(aligned_idx, method="nearest")
                freq_resampled = pd.Series(
                    [float(freq_series.iloc[loc]) if loc >= 0 else 0.0 for loc in locs],
                    index=aligned_idx,
                )

    # Commit times (for hover annotation)
    commit_time_by_market: dict[str, pd.Series] = {}
    if commit_df is not None and not commit_df.empty:
        tmp = commit_df.copy()
        tmp["delivery_time"] = pd.to_datetime(tmp["delivery_time"])
        tmp["current_time"] = pd.to_datetime(tmp["current_time"])
        if "market" in tmp.columns and "commit_now" in tmp.columns:
            for mk in tmp["market"].unique():
                mk_rows = tmp[(tmp["market"] == mk) & tmp["commit_now"].astype(bool)]
                if mk_rows.empty:
                    continue
                s = mk_rows.sort_values("current_time").drop_duplicates("delivery_time")
                commit_time_by_market[mk] = s.set_index("delivery_time")["current_time"]

    n_rows = 6 if has_freq_subplot else 4
    subplot_titles_list = [
        "Market Prices",
        "Power Flexband (p_net within [P_lower, P_upper])",
        "Energy Flexband (E within [E_lower, E_upper])",
        "Market Positions",
    ]
    if has_freq_subplot:
        subplot_titles_list.append("FCR Activation Power (droop · bid)")
        subplot_titles_list.append("FCR Droop Signal")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        specs=[[{}]] * n_rows,
        subplot_titles=tuple(subplot_titles_list),
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
                row=1,
                col=1,
            )

    # Row 1 overlay: FCR clearing price vs. breakeven bid per slot (€/MW for the 4 h product; the shaded gap is the margin).
    # Committed slots are green (breakeven below clearing, positive margin); declined-but-capable slots ("ok_entry_price") are red (breakeven above clearing, negative margin — the price the slot would have needed).
    if (
        fcr_commit_df is not None
        and not fcr_commit_df.empty
        and "breakeven_eur_per_mw" in fcr_commit_df.columns
    ):
        bdf = fcr_commit_df.copy()
        bdf["slot_start"] = pd.to_datetime(bdf["slot_start"], utc=True).dt.tz_convert(dispatch.index.tz)
        bdf = bdf[bdf["breakeven_eur_per_mw"].notna()].sort_values("slot_start")
        x_max = dispatch.index[-1]
        bdf = bdf[bdf["slot_start"] <= x_max]
        if not bdf.empty:
            status = bdf.get("breakeven_status")
            if status is not None:
                declined_mask = status == "ok_entry_price"
            else:
                declined_mask = bdf["margin_eur_per_mw"] < 0

            xs: list = []
            clearing_ys: list = []
            for _, b_row in bdf.iterrows():
                slot_end = b_row["slot_start"] + pd.Timedelta(hours=fcr_product_hours)
                s0, s1 = b_row["slot_start"].isoformat(), min(slot_end, x_max).isoformat()
                xs += [s0, s1, s1]
                clearing_ys += [float(b_row["fcr_price"]), float(b_row["fcr_price"]), None]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=clearing_ys,
                    mode="lines",
                    name="FCR Clearing [€/MW]",
                    legendgroup="fcr_breakeven",
                    line=dict(width=2, color=FCR_GREEN),
                    hovertemplate="FCR clearing: %{y:.2f} €/MW<extra></extra>",
                ),
                row=1,
                col=1,
            )

            def _add_breakeven_group(sub_df, *, color, fill, name, group):
                if sub_df.empty:
                    return
                gxs: list = []
                breakeven_ys: list = []
                margins: list = []
                poly_xs: list = []
                poly_ys: list = []
                for _, b_row in sub_df.iterrows():
                    slot_end = b_row["slot_start"] + pd.Timedelta(hours=fcr_product_hours)
                    s0, s1 = b_row["slot_start"].isoformat(), min(slot_end, x_max).isoformat()
                    clr = float(b_row["fcr_price"])
                    bev = float(b_row["breakeven_eur_per_mw"])
                    m_val = float(b_row.get("margin_eur_per_mw", float("nan")))
                    gxs += [s0, s1, s1]
                    breakeven_ys += [bev, bev, None]
                    margins += [m_val, m_val, None]
                    poly_xs += [s0, s1, s1, s0, s0, None]
                    poly_ys += [clr, clr, bev, bev, clr, None]
                fig.add_trace(
                    go.Scatter(
                        x=poly_xs,
                        y=poly_ys,
                        mode="lines",
                        fill="toself",
                        fillcolor=fill,
                        line=dict(width=0),
                        legendgroup=group,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=gxs,
                        y=breakeven_ys,
                        mode="lines",
                        name=name,
                        legendgroup=group,
                        line=dict(width=2, color=color, dash="dash"),
                        customdata=margins,
                        hovertemplate=(
                            "FCR breakeven: %{y:.2f} €/MW<br>margin: %{customdata:.2f} €/MW<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

            _add_breakeven_group(
                bdf[~declined_mask],
                color=FCR_GREEN,
                fill=FCR_GREEN_LIGHT,
                name="FCR Breakeven Bid [€/MW]",
                group="fcr_breakeven",
            )
            _add_breakeven_group(
                bdf[declined_mask],
                color=FCR_RED,
                fill=FCR_RED_LIGHT,
                name="FCR Entry Price (declined) [€/MW]",
                group="fcr_entry_price",
            )

    # Row 2: Power band + p_net
    fig.add_trace(
        go.Scatter(
            x=dispatch.index,
            y=dispatch["P_upper_kw"],
            mode="lines",
            name="P upper [kW]",
            line=dict(width=1, color="rgba(0,0,0,0.5)"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dispatch.index,
            y=dispatch["P_lower_kw"],
            mode="lines",
            name="P lower [kW]",
            fill="tonexty",
            line=dict(width=1, color="rgba(0,0,0,0.5)"),
            fillcolor="rgba(0,0,0,0.08)",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dispatch.index,
            y=dispatch["p_net_kw"],
            mode="lines",
            name="p_net [kW]",
            line=dict(width=3, color="rgb(162,173,0)"),
        ),
        row=2,
        col=1,
    )

    # Row 3: Energy band + E
    fig.add_trace(
        go.Scatter(
            x=dispatch.index,
            y=dispatch["E_upper_kWh"],
            mode="lines",
            name="E upper [kWh]",
            line=dict(width=1, color="rgba(0,0,0,0.5)"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dispatch.index,
            y=dispatch["E_lower_kWh"],
            mode="lines",
            name="E lower [kWh]",
            fill="tonexty",
            line=dict(width=1, color="rgba(0,0,0,0.5)"),
            fillcolor="rgba(0,0,0,0.08)",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dispatch.index,
            y=dispatch["E_kWh"],
            mode="lines",
            name="E [kWh]",
            line=dict(width=3, color="rgb(162,173,0)"),
        ),
        row=3,
        col=1,
    )

    # Row 4: Market positions
    for col in market_cols:
        mk_code = col[2:-3].upper()
        values = dispatch[col]

        colors = [_rgba(mk_code, 0.3 if v < 0 else 1.0) for v in values]
        labels = [
            f"{mk_code} Buy" if v > 0 else f"{mk_code} Sell" if v < 0 else f"{mk_code} Neutral"
            for v in values
        ]

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
                    "%{y:.1f} kW<br>%{customdata[0]}<br>Committed at: %{customdata[1]}<extra></extra>"
                ),
            ),
            row=4,
            col=1,
        )

    if has_fcr and x_fcr_dense is not None:
        fcr_commit_times = pd.Series("not committed", index=dispatch.index)
        if fcr_commit_df is not None and not fcr_commit_df.empty:
            tmp_fcr = fcr_commit_df.copy()
            if "slot_start" in tmp_fcr.columns:
                tmp_fcr["slot_start"] = pd.to_datetime(tmp_fcr["slot_start"], utc=False)
                if tmp_fcr["slot_start"].dt.tz is None:
                    tmp_fcr["slot_start"] = tmp_fcr["slot_start"].dt.tz_localize(dispatch.index.tz)
                else:
                    tmp_fcr["slot_start"] = tmp_fcr["slot_start"].dt.tz_convert(dispatch.index.tz)
            if "committed_at" in tmp_fcr.columns:
                tmp_fcr["committed_at"] = pd.to_datetime(tmp_fcr["committed_at"], utc=False)
                if tmp_fcr["committed_at"].dt.tz is None:
                    tmp_fcr["committed_at"] = tmp_fcr["committed_at"].dt.tz_localize(dispatch.index.tz)
                else:
                    tmp_fcr["committed_at"] = tmp_fcr["committed_at"].dt.tz_convert(dispatch.index.tz)
                for _, fcr_row in tmp_fcr.iterrows():
                    slot_end = fcr_row["slot_start"] + pd.Timedelta(hours=fcr_product_hours)
                    mask = (dispatch.index >= fcr_row["slot_start"]) & (dispatch.index < slot_end)
                    fcr_commit_times[mask] = fcr_row["committed_at"].strftime("%Y-%m-%d %H:%M")

        custom_fcr = list(zip(x_fcr_dense.values, fcr_commit_times.values))
        fig.add_trace(
            go.Scatter(
                x=dispatch.index,
                y=x_fcr_dense,
                mode="lines",
                name="FCR committed [kW]",
                legendgroup="fcr_committed",
                line=dict(width=2, color=FCR_GREEN),
                customdata=custom_fcr,
                hovertemplate=(
                    "FCR: ±%{customdata[0]:.0f} kW<br>Committed at: %{customdata[1]}<extra></extra>"
                ),
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dispatch.index,
                y=-x_fcr_dense,
                mode="lines",
                name="FCR committed [-kW]",
                legendgroup="fcr_committed",
                fill="tonexty",
                fillcolor=FCR_GREEN_MID,
                line=dict(width=2, color=FCR_GREEN, dash="dot"),
                customdata=custom_fcr,
                hovertemplate=(
                    "FCR: ±%{customdata[0]:.0f} kW<br>Committed at: %{customdata[1]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=4,
            col=1,
        )

    # Imbalance net position
    if has_imb:
        p_imb_net = dispatch["p_imb_pos_kw"] - dispatch["p_imb_neg_kw"]
        colors = [_rgba("IMB", 1.0 if v > 0 else 0.3) for v in p_imb_net]
        labels = ["Buy" if v > 0 else "Sell" if v < 0 else "Neutral" for v in p_imb_net]

        if has_used_flag:
            used_str = (
                dispatch["used_rebap"].astype(bool).map(lambda x: "reBAP used" if x else "no reBAP").values
            )
        else:
            used_str = ["reBAP unknown"] * len(dispatch)

        custom = pd.DataFrame(
            {
                "side": labels,
                "used": used_str,
                "pos": dispatch["p_imb_pos_kw"].values,
                "neg": dispatch["p_imb_neg_kw"].values,
            }
        ).values

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
            row=4,
            col=1,
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
                    x0=start,
                    x1=end,
                    fillcolor="rgba(120,80,160,0.12)",
                    line_width=0,
                    layer="below",
                    row=4,
                    col=1,
                )
                start = None
        if start is not None:
            fig.add_vrect(
                x0=start,
                x1=dispatch.index[-1],
                fillcolor="rgba(120,80,160,0.12)",
                line_width=0,
                layer="below",
                row="all",
                col=1,
            )

    fig.update_yaxes(title_text="Price [€/MWh]", row=1, col=1)
    fig.update_yaxes(title_text="Power [kW]", row=2, col=1)
    fig.update_yaxes(title_text="Energy [kWh]", row=3, col=1)
    fig.update_yaxes(title_text="Position [kW]", row=4, col=1)

    if has_freq_subplot and freq_resampled is not None and p_droop_kw is not None:
        # Import-positive convention: + = depot charges (downward FCR), - = depot discharges.
        bar_colors = ["rgba(31,119,180,0.75)" if v > 0 else "rgba(214,39,40,0.75)" for v in p_droop_kw]
        fig.add_trace(
            go.Bar(
                x=p_droop_kw.index,
                y=p_droop_kw.values,
                name="FCR Activation Power [kW]",
                marker_color=bar_colors,
                yaxis="y5",
                hovertemplate=("FCR power: %{y:.1f} kW<br>(+charge / -discharge)<extra></extra>"),
                opacity=0.85,
            ),
            row=5,
            col=1,
        )

        fig.update_yaxes(title_text="Power [kW]", row=5, col=1)

        fig.add_trace(
            go.Scatter(
                x=freq_resampled.index,
                y=freq_resampled.values,
                mode="lines",
                name="FCR Droop Signal",
                line=dict(width=1.5, color="rgba(80,80,80,0.7)"),
                hovertemplate="droop = %{y:.3f}<extra></extra>",
            ),
            row=6,
            col=1,
        )

        fig.add_hline(
            y=0.0,
            line=dict(width=1, color="rgba(0,0,0,0.3)", dash="dash"),
            row=6,
            col=1,
        )

        fig.update_yaxes(title_text="Droop [-1, 1]", row=6, col=1)

    plot_height = 1450 if has_freq_subplot else 1200

    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", y=0.99, yanchor="top", yref="container"),
        template="plotly_white",
        height=plot_height,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1, traceorder="normal"),
        margin=dict(l=60, r=60, t=150, b=60),
        barmode="relative",
        hovermode="x unified",
    )

    return fig
