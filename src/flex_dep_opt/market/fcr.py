from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def droop_signal(
    freq: pd.Series,
    *,
    nominal_hz: float,
    deadband_hz: float,
    full_activation_hz: float,
) -> pd.Series:
    """
    FCR droop signal in [-1, +1]. No activation inside the deadband, then a
    linear ramp from the deadband edge up to full activation
    """
    span = full_activation_hz - deadband_hz
    if span <= 0:
        raise ValueError("full_activation_hz must be greater than deadband_hz")
    delta_f = freq - nominal_hz
    excess = (delta_f.abs() - deadband_hz).clip(lower=0.0)
    magnitude = (excess / span).clip(upper=1.0)
    return -np.sign(delta_f) * magnitude


def fcr_gate_closure_timestamp(
    slot_start: pd.Timestamp,
    *,
    hour: str = "08:00",
    closes_previous_day: bool = True,
    timezone: str = "Europe/Berlin",
) -> pd.Timestamp:
    """
    Capacity-market gate-closure timestamp for an FCR 4 h slot.
    """
    hh, mm = (int(x) for x in hour.split(":"))
    day = slot_start.normalize()
    if closes_previous_day:
        day = day - pd.Timedelta(days=1)
    return day.replace(hour=hh, minute=mm, tzinfo=ZoneInfo(timezone))


