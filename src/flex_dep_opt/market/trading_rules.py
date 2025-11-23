from __future__ import annotations

from typing import Dict
import pandas as pd


def build_market_activity_mask(
    time_index: pd.DatetimeIndex,
    optimization_cfg: dict,
) -> Dict[str, pd.Series]:
    """
    Erzeuge für jeden Markt eine boolesche Zeitreihe, die angibt,
    ob in diesem Zeitschritt gehandelt werden darf (True) oder nicht (False).

    Aktuell:
      - trading.mode == "none"  -> überall True (keine Einschränkung)
    Später:
      - trading.mode == "realistic" -> hier kommen DA-Gate-Closure,
        ID-30min-Regel etc. rein.

    Rückgabe:
      dict wie {"DA": pd.Series[bool], "ID": pd.Series[bool], ...}
      Index entspricht `time_index`.
    """

    trading_cfg = optimization_cfg.get("trading", {})
    mode = trading_cfg.get("mode", "none")

    # Mapping von settings-Schlüssel zu Marktkürzel im Modell
    market_cfg = optimization_cfg["markets"]
    enabled_markets = []

    if market_cfg["dayahead"]["enabled"]:
        enabled_markets.append("DA")
    if market_cfg["intraday"]["enabled"]:
        enabled_markets.append("ID")
    # später z.B. weitere Märkte ergänzen

    mask_by_market: Dict[str, pd.Series] = {}

    if mode == "none":
        # Heutiges Verhalten: alle Märkte jederzeit handelbar
        for mk in enabled_markets:
            mask_by_market[mk] = pd.Series(True, index=time_index)
        return mask_by_market

    # Hier können wir später weitere Modi implementieren, z.B. "realistic"
    # mit echten Gate-Closures. Bis dahin: fallback = alles True.
    for mk in enabled_markets:
        mask_by_market[mk] = pd.Series(True, index=time_index)

    return mask_by_market
