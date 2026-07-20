from __future__ import annotations

import pandas as pd

from flex_dep_opt.config.settings import OptimizationSettings
from flex_dep_opt.io.time import LOCAL_TIMEZONE


# =============================================================================
# Gate-closure helpers (single source of truth)
# =============================================================================
def _parse_hhmm(value: str) -> tuple[int, int]:
    """Parse a 'HH:MM' string to integers."""
    hh_str, mm_str = value.split(":")
    return int(hh_str), int(mm_str)


def gate_closure_timestamp(
    market: str,
    delivery_time: pd.Timestamp,
    optimization_cfg: OptimizationSettings,
    *,
    market_tz: str = LOCAL_TIMEZONE,
) -> pd.Timestamp:
    """
    Compute the market gate-closure timestamp for a given delivery time.

    Notes
    -----
    - This function is the *single source of truth* for gate-closure calculations.
    - It is used by:
        (i) market activity masks (tradability),
        (ii) MPC commit logging / diagnostics.

    Parameters
    ----------
    market:
        Market identifier, e.g. "DA" or "ID".
    delivery_time:
        Delivery timestamp. Internally this may be UTC; gate rules are evaluated
        in `market_tz`, and the returned timestamp is converted back to the
        timezone of `delivery_time`.
    optimization_cfg:
        Optimization configuration containing `optimization_cfg["trading"]`.

    Returns
    -------
    pd.Timestamp
        Gate-closure timestamp for the given market and delivery time.

    Raises
    ------
    ValueError
        If `market` is unknown.
    """
    trading_cfg = optimization_cfg.trading
    if delivery_time.tzinfo is None:
        delivery_local = delivery_time.tz_localize(market_tz, ambiguous="raise", nonexistent="raise")
        output_tz = market_tz
    else:
        output_tz = str(delivery_time.tz)
        delivery_local = delivery_time.tz_convert(market_tz)

    if market == "DA":
        da_cfg = trading_cfg.dayahead
        gc_hour_str = da_cfg.gate_closure_hour
        closes_prev = da_cfg.closes_previous_day

        gc_h, gc_m = _parse_hhmm(gc_hour_str)

        base_date = (
            delivery_local.normalize() - pd.Timedelta(days=1) if closes_prev else delivery_local.normalize()
        )
        # `normalize()` preserves tz-awareness; adding Timedelta keeps tz-awareness
        return (base_date + pd.Timedelta(hours=gc_h, minutes=gc_m)).tz_convert(output_tz)

    if market == "ID":
        id_cfg = trading_cfg.intraday
        offset_min = id_cfg.offset_minutes_before_delivery
        return (delivery_local - pd.Timedelta(minutes=offset_min)).tz_convert(output_tz)

    raise ValueError(f"Unknown market: {market}")


# =============================================================================
# Market selection helpers
# =============================================================================
def _enabled_markets_from_cfg(optimization_cfg: OptimizationSettings) -> list[str]:
    """
    Return enabled markets as a list of market codes ("DA", "ID", ...).
    """
    market_cfg = optimization_cfg.markets

    enabled: list[str] = []
    if market_cfg.dayahead.enabled:
        enabled.append("DA")
    if market_cfg.intraday.enabled:
        enabled.append("ID")
    return enabled


# =============================================================================
# Public API
# =============================================================================
def build_market_activity_mask_for_time(
    current_time: pd.Timestamp,
    delivery_times: pd.DatetimeIndex,
    optimization_cfg: OptimizationSettings,
) -> dict[str, pd.Series]:
    """
    Build market activity masks for ONE MPC step.

    The masks indicate whether trading is allowed *at current_time* for each
    delivery time τ in `delivery_times`. The mask is computed per enabled market.

    Trading modes
    -------------
    - mode == "none":
        All enabled markets are tradable for all delivery times (mask=True).

    - mode == "realistic":
        Enforces gate closures:

        Day-Ahead (DA):
            tradable if current_time < gate_closure_timestamp("DA", τ, cfg)

        Intraday (ID):
            tradable if current_time <= gate_closure_timestamp("ID", τ, cfg)
            (inclusive gate closure, matching previous behavior)

    Returns
    -------
    dict[str, pd.Series]
        One boolean series per enabled market, indexed by `delivery_times`.
        If no markets are enabled, returns an empty dict.
    """
    trading_cfg = optimization_cfg.trading
    mode = trading_cfg.mode

    if current_time.tzinfo is None:
        current_cmp = current_time.tz_localize(LOCAL_TIMEZONE, ambiguous="raise", nonexistent="raise")
    else:
        current_cmp = current_time

    enabled_markets = _enabled_markets_from_cfg(optimization_cfg)
    mask_by_market: dict[str, pd.Series] = {}

    # -------------------------------------------------------------------------
    # Mode "none": everything open (baseline / debugging)
    # -------------------------------------------------------------------------
    if mode == "none":
        for mk in enabled_markets:
            mask_by_market[mk] = pd.Series(True, index=delivery_times)
        return mask_by_market

    # -------------------------------------------------------------------------
    # Mode "realistic": enforce gate closures via gate_closure_timestamp()
    # -------------------------------------------------------------------------
    if mode == "realistic":
        for mk in enabled_markets:
            # Compute gate-closure timestamps for each τ
            gcs = pd.Series(
                [gate_closure_timestamp(mk, tau, optimization_cfg) for tau in delivery_times],
                index=delivery_times,
            )

            # Apply market-specific "open" condition (keep historical semantics)
            if mk == "DA":
                mask_by_market[mk] = current_cmp < gcs
            elif mk == "ID":
                mask_by_market[mk] = current_cmp <= gcs
            else:
                # If you add more markets later, enforce explicit semantics here
                raise ValueError(f"Missing activity rule for market={mk}")

        return mask_by_market

    # Unknown mode: return empty masks (explicit and safe)
    return mask_by_market
