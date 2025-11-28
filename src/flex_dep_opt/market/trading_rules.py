from __future__ import annotations
from typing import Dict, List
import pandas as pd


def build_market_activity_mask_for_time(
    current_time: pd.Timestamp,
    delivery_times: pd.DatetimeIndex,
    optimization_cfg: dict,
) -> Dict[str, pd.Series]:
    """
    Marktaktivitätsmasken für EINEN MPC-Schritt.

    Logik:
      - mode == "none":
          alle aktivierten Märkte sind für alle delivery_times handelbar.

      - mode == "realistic":
          Day-Ahead (DA):
            Für Lieferzeitpunkt τ:
              gate_closure = (τ.date - 1 Tag falls closes_previous_day)
                              + gate_closure_hour
            Handel erlaubt, wenn:
              current_time < gate_closure

          Intraday (ID):
            Für Lieferzeitpunkt τ:
              gate_closure = τ - offset_minutes_before_delivery
            Handel erlaubt, wenn:
              current_time <= gate_closure
    """

    trading_cfg = optimization_cfg.get("trading", {})
    mode = trading_cfg.get("mode", "none")

    market_cfg = optimization_cfg["markets"]

    enabled_markets = []
    if market_cfg["dayahead"]["enabled"]:
        enabled_markets.append("DA")
    if market_cfg["intraday"]["enabled"]:
        enabled_markets.append("ID")

    mask_by_market: Dict[str, pd.Series] = {}

    # --- Mode "none": alles offen -------------------------------------------
    if mode == "none":
        for mk in enabled_markets:
            mask_by_market[mk] = pd.Series(True, index=delivery_times)
        return mask_by_market

    # --- Mode "realistic": nutze deine Trading-Settings ---------------------
    if mode == "realistic":

        # Day-Ahead
        if "DA" in enabled_markets:
            da_cfg = trading_cfg.get("dayahead", {})
            gc_hour_str = da_cfg.get("gate_closure_hour", "12:00")
            closes_prev = da_cfg.get("closes_previous_day", True)

            # Stunden/Minuten aus "HH:MM" parsen
            try:
                hh_str, mm_str = gc_hour_str.split(":")
                gc_h = int(hh_str)
                gc_m = int(mm_str)
            except Exception:
                gc_h, gc_m = 12, 0  # Fallback

            da_mask_vals = []
            for tau in delivery_times:
                # Basisdatum = Lieferdatum oder Vortag
                if closes_prev:
                    gc_date = (tau.normalize() - pd.Timedelta(days=1))
                else:
                    gc_date = tau.normalize()

                # gc_date hat gleiche TZ wie τ (wenn τ tz-aware ist)
                gc_ts = gc_date + pd.Timedelta(hours=gc_h, minutes=gc_m)

                # Handel erlaubt, solange current_time vor GC liegt
                da_mask_vals.append(current_time < gc_ts)

            mask_by_market["DA"] = pd.Series(da_mask_vals, index=delivery_times)

        # Intraday
        if "ID" in enabled_markets:
            id_cfg = trading_cfg.get("intraday", {})
            offset_min = int(id_cfg.get("offset_minutes_before_delivery", 30))

            id_mask_vals = []
            for tau in delivery_times:
                gc_ts = tau - pd.Timedelta(minutes=offset_min)
                # Handel erlaubt bis inkl. Gate Closure
                id_mask_vals.append(current_time <= gc_ts)

            mask_by_market["ID"] = pd.Series(id_mask_vals, index=delivery_times)

        return mask_by_market

    return mask_by_market


def build_market_activity_mask(
    time_index: pd.DatetimeIndex,
    optimization_cfg: dict,
) -> Dict[str, pd.Series]:
    """
    Alte Version für Single-Shot-Optimierung:
    - trading.mode == "none"  -> überall True
    - trading.mode == "realistic" -> aktuell noch keine Einschränkung (nur für PF)
    """
    trading_cfg = optimization_cfg.get("trading", {})
    mode = trading_cfg.get("mode", "none")

    market_cfg = optimization_cfg["markets"]
    enabled_markets = []

    if market_cfg["dayahead"]["enabled"]:
        enabled_markets.append("DA")
    if market_cfg["intraday"]["enabled"]:
        enabled_markets.append("ID")

    mask_by_market: Dict[str, pd.Series] = {}

    # Heutiges Verhalten: alles offen
    for mk in enabled_markets:
        mask_by_market[mk] = pd.Series(True, index=time_index)

    return mask_by_market