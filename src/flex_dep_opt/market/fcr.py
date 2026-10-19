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
    """FCR droop signal in [-1, +1]: zero inside the deadband, then linear up to full activation."""
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
    """Gate closure of the FCR capacity auction for the slot starting at `slot_start`."""
    hh, mm = (int(x) for x in hour.split(":"))
    day = slot_start.tz_convert(timezone).normalize()
    if closes_previous_day:
        day = day - pd.Timedelta(days=1)
    return day.replace(hour=hh, minute=mm, tzinfo=ZoneInfo(timezone))


def symmetric_fcr_capacity(
    power_upper_kw: np.ndarray,
    power_lower_kw: np.ndarray,
    *,
    grid_limit_kw: float,
) -> float:
    """Largest symmetric FCR capacity [kW] that fits on every step of a slot."""
    upper = np.minimum(np.asarray(power_upper_kw, dtype=float), grid_limit_kw)
    lower = np.maximum(np.asarray(power_lower_kw, dtype=float), -grid_limit_kw)
    if upper.size == 0:
        return 0.0
    return max(float((0.5 * (upper - lower)).min()), 0.0)
