# Interactive Plotly visualization of dispatch and prices using subplots.

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_dispatch_plotly(
    dispatch: pd.DataFrame,
    prices: pd.Series | None = None,
    title: str = "Dispatch and Price Profile",
    *,
    capacity_kwh: float,
) -> go.Figure:
    """
    Create an interactive Plotly figure with two subplots:
      1. Upper subplot: Price [€/MWh] and Net Power [kW]
      2. Lower subplot: Price [€/MWh] and State of Charge [kWh]

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

    # Convert prices to €/MWh for better readability
    if prices is not None:
        price_mwh = prices * 1000
    else:
        price_mwh = pd.Series(index=dispatch.index, data=[None] * len(dispatch))

    # Convert SoC from kWh to % (relative to max)
    if "soc_kwh" not in dispatch.columns:
        raise ValueError("Dispatch must contain a 'soc_kwh' column")

    soc_kwh = dispatch["soc_kwh"]
    soc_percent = (soc_kwh / float(capacity_kwh)) * 100.0

    # --- Create subplots ---
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=("Net Power vs. Price", "State of Charge vs. Price"),
    )

    # ==== Subplot 1: Net Power + Price ====
    # Price (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=price_mwh.index,
            y=price_mwh.values,
            mode="lines",
            name="Price [€/MWh]",
            line=dict(color="green", width=2, dash="dot"),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # Net Power (right y-axis)
    fig.add_trace(
        go.Scatter(
            x=dispatch.index,
            y=net_power,
            mode="lines",
            name="Net Power [kW]",
            line=dict(color="royalblue", width=2),
            fill="tozeroy",
            fillcolor="rgba(65,105,225,0.2)",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # ==== Subplot 2: SoC + Price ====
    # Price (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=price_mwh.index,
            y=price_mwh.values,
            mode="lines",
            name="Price [€/MWh]",
            line=dict(color="green", width=2, dash="dot"),
            showlegend=False,  # already shown above
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    # SoC (right y-axis)
    fig.add_trace(
        go.Scatter(
            x=dispatch.index,
            y=soc_percent,
            mode="lines",
            name="State of Charge [kWh]",
            line=dict(color="orange", width=3),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    # --- Update axes titles ---
    fig.update_yaxes(title_text="Price [€/MWh]", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Net Power [kW]", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Price [€/MWh]", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="State of Charge [%]", row=2, col=1, secondary_y=True)

    # --- Layout ---
    fig.update_layout(
        title=title,
        xaxis=dict(title="Time"),
        template="plotly_white",
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=100, b=60),
    )

    return fig

