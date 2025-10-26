# plots.py
# Interactive Plotly visualization of dispatch and prices.

import pandas as pd
import plotly.graph_objects as go


def plot_dispatch_plotly(
    dispatch: pd.DataFrame,
    prices: pd.Series | None = None,
    title: str = "Dispatch and Price Profile"
) -> go.Figure:
    """
    Create an interactive Plotly figure showing:
      - net power (discharge - charge)
      - state of charge (SoC)
      - optional market prices

    Parameters
    ----------
    dispatch : pd.DataFrame
        Must contain columns ["p_ch_kw", "p_dis_kw", "soc_kwh"].
    prices : pd.Series, optional
        Optional Series of prices (€/kWh) indexed by time.
    title : str
        Title of the figure.

    Returns
    -------
    plotly.graph_objects.Figure
        A ready-to-show interactive figure.
    """
    # Ensure time index
    if not isinstance(dispatch.index, pd.DatetimeIndex):
        raise ValueError("Dispatch index must be a DatetimeIndex")

    # Calculate net power (positive = discharging)
    net_power = dispatch["p_dis_kw"] - dispatch["p_ch_kw"]

    # Create figure with secondary y-axis
    fig = go.Figure()

    # --- Power trace ---
    fig.add_trace(
        go.Scatter(
            x=dispatch.index,
            y=net_power,
            mode="lines",
            name="Net Power [kW]",
            line=dict(color="royalblue", width=2),
            fill="tozeroy",
            fillcolor="rgba(65,105,225,0.2)",
        )
    )

    # --- SoC trace ---
    fig.add_trace(
        go.Scatter(
            x=dispatch.index,
            y=dispatch["soc_kwh"],
            mode="lines",
            name="State of Charge [kWh]",
            line=dict(color="orange", width=3),
            yaxis="y2",
        )
    )

    # --- Optional prices trace ---
    if prices is not None:
        # Convert to €/MWh for more intuitive scale
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices.values * 1000,
                mode="lines",
                name="Price [€/MWh]",
                line=dict(color="green", width=2, dash="dot"),
                yaxis="y3",
            )
        )

    # --- Layout with multiple y-axes ---
    layout = dict(
        title=title,
        xaxis=dict(title="Time"),
        yaxis=dict(title="Net Power [kW]", side="left", showgrid=False),
        yaxis2=dict(
            title="State of Charge [kWh]",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if prices is not None:
        layout["yaxis3"] = dict(
            title="Price [€/MWh]",
            overlaying="y",
            side="right",
            position=1.0,
            showgrid=False,
        )

    fig.update_layout(layout)
    return fig
