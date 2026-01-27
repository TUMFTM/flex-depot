from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import pandas as pd


# =============================================================================
# Helpers
# =============================================================================
def infer_market_position_columns(dispatch: pd.DataFrame) -> List[str]:
    """
    Return all dispatch columns that represent per-market positions p_<mk>_kw.

    Excludes physical variables (p_ch/p_dis/p_net) and imbalance variables.
    """
    return [
        c for c in dispatch.columns
        if c.startswith("p_")
        and c.endswith("_kw")
        and c not in ("p_ch_kw", "p_dis_kw", "p_net_kw", "p_imb_pos_kw", "p_imb_neg_kw")
    ]


def has_imbalance(dispatch: pd.DataFrame) -> bool:
    """True if dispatch contains imbalance columns p_imb_pos_kw and p_imb_neg_kw."""
    return ("p_imb_pos_kw" in dispatch.columns) and ("p_imb_neg_kw" in dispatch.columns)


# =============================================================================
# Core computations
# =============================================================================
def compute_cashflows_per_step(
    dispatch: pd.DataFrame,
    prices_by_market: Mapping[str, pd.Series],
    *,
    timestep_hours: float,
) -> pd.DataFrame:
    """
    Compute per-timestep cashflows for each market and total.

    Convention
    ----------
    cashflow[t] = - price[t] * p_market[t] * dt

    - p_market > 0 => buy/import => negative cashflow (cost)
    - p_market < 0 => sell/export => positive cashflow (revenue)

    Output columns
    --------------
    - "<MK> Cashflow [€/step]" for each market found in dispatch
    - "IMB Cashflow [€/step]" if imbalance columns exist and IMB prices exist
    - "Total Cashflow [€/step]"
    - "Cumulative Profit [€]"
    """
    if not isinstance(dispatch.index, pd.DatetimeIndex):
        raise ValueError("Dispatch index must be a DatetimeIndex")

    dt = float(timestep_hours)
    market_cols = infer_market_position_columns(dispatch)

    cf_df = pd.DataFrame(index=dispatch.index)

    # -------------------------------------------------------------------------
    # Market cashflows (DA, ID, ...)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Optional: imbalance cashflow (reBAP)
    # Treat as a cost component using IMB_POS and IMB_NEG prices.
    # -------------------------------------------------------------------------
    if has_imbalance(dispatch) and ("IMB_POS" in prices_by_market) and ("IMB_NEG" in prices_by_market):
        p_pos = dispatch["p_imb_pos_kw"].astype(float).copy().clip(lower=0.0)
        p_neg = dispatch["p_imb_neg_kw"].astype(float).copy().clip(lower=0.0)

        pos_price = prices_by_market["IMB_POS"]
        neg_price = prices_by_market["IMB_NEG"]

        idx = dispatch.index.intersection(pos_price.index).intersection(neg_price.index)
        if len(idx) > 0:
            p_pos = p_pos.reindex(idx).fillna(0.0)
            p_neg = p_neg.reindex(idx).fillna(0.0)
            pos_price = pos_price.reindex(idx)
            neg_price = neg_price.reindex(idx)

            cf_df["IMB Cashflow [€/step]"] = -(pos_price * p_pos + neg_price * p_neg) * dt

    if cf_df.empty:
        raise ValueError("No cashflows could be computed (check market columns and prices_by_market).")

    cf_df["Total Cashflow [€/step]"] = cf_df.sum(axis=1)
    cf_df["Cumulative Profit [€]"] = cf_df["Total Cashflow [€/step]"].cumsum()

    return cf_df


def compute_market_aggregates(
    dispatch: pd.DataFrame,
    prices_by_market: Mapping[str, pd.Series],
    *,
    timestep_hours: float,
) -> tuple[
    dict[str, tuple[float, float]],
    dict[str, tuple[float, float]],
    list[tuple[str, str, float]],
    list[tuple[str, str, float]],
]:
    """
    Compute market aggregates for KPI tables and sunburst plots.

    Returns
    -------
    energy_by_mk:
        mk -> (buy_kwh, sell_kwh), both >= 0
    cash_by_mk:
        mk -> (buy_eur, sell_eur), both >= 0 (magnitudes)
    energy_data:
        list (mk, side, value>=0) for sunburst (Buy/Sell)
    cash_data:
        list (mk, side, value>=0) for sunburst (Buy/Sell/Cost)

    Notes
    -----
    - "Buy" corresponds to p > 0, "Sell" corresponds to p < 0.
    - IMB is aggregated from IMB_POS/IMB_NEG into a synthetic market "IMB".
    """
    dt = float(timestep_hours)
    market_cols = infer_market_position_columns(dispatch)

    energy_by_mk: dict[str, tuple[float, float]] = {}
    cash_by_mk: dict[str, tuple[float, float]] = {}

    energy_data: list[tuple[str, str, float]] = []
    cash_data: list[tuple[str, str, float]] = []

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

        buy_e = float(energy_kwh[buy_mask].sum()) if buy_mask.any() else 0.0
        sell_e = float((-energy_kwh[sell_mask]).sum()) if sell_mask.any() else 0.0

        buy_c = float((-cash_eur[buy_mask]).sum()) if buy_mask.any() else 0.0
        sell_c = float((cash_eur[sell_mask]).sum()) if sell_mask.any() else 0.0

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

    # IMB aggregation
    if has_imbalance(dispatch) and ("IMB_POS" in prices_by_market) and ("IMB_NEG" in prices_by_market):
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

            p_net = p_pos - p_neg
            energy_kwh_net = p_net * dt

            buy_mask = energy_kwh_net > 0
            sell_mask = energy_kwh_net < 0

            buy_e = float(energy_kwh_net[buy_mask].sum()) if buy_mask.any() else 0.0
            sell_e = float((-energy_kwh_net[sell_mask]).sum()) if sell_mask.any() else 0.0

            imb_cost_eur = float((pos_price * p_pos + neg_price * p_neg).sum() * dt)

            energy_by_mk["IMB"] = (buy_e, sell_e)
            cash_by_mk["IMB"] = (imb_cost_eur, 0.0)

            if buy_e > 0:
                energy_data.append(("IMB", "Buy", buy_e))
            if sell_e > 0:
                energy_data.append(("IMB", "Sell", sell_e))
            if imb_cost_eur > 0:
                cash_data.append(("IMB", "Cost", imb_cost_eur))

    return energy_by_mk, cash_by_mk, energy_data, cash_data


def compute_kpis(
    cf_df: pd.DataFrame,
    energy_by_mk: Mapping[str, tuple[float, float]],
    fee_eur_per_kwh_by_market: Mapping[str, float],
    *,
    commit: Optional[pd.DataFrame] = None,
) -> dict[str, float | int]:
    """
    Compute KPIs for reporting.

    KPI definitions
    ---------------
    - gross_profit_eur:
        Total cashflow plus fee term (fees are negative).
    - trading_profit_eur:
        Profit excluding fees and imbalance cost.
    - fees_eur:
        Applied to absolute traded energy volume in DA/ID.
    - imb_cost_eur:
        Magnitude derived from "IMB Cashflow [€/step]" if present.
    - trade_steps:
        Count of committed trades based on commit_df.
    - buy/sell/net energy:
        Aggregated traded energy excluding IMB.
    """
    buy_kwh = float(sum(v[0] for mk, v in energy_by_mk.items() if mk != "IMB"))
    sell_kwh = float(sum(v[1] for mk, v in energy_by_mk.items() if mk != "IMB"))
    net_kwh = sell_kwh - buy_kwh

    def _mk_fee(mk: str) -> float:
        if mk not in energy_by_mk:
            return 0.0
        buy_e, sell_e = energy_by_mk[mk]
        return float(fee_eur_per_kwh_by_market.get(mk, 0.0)) * float(buy_e + sell_e)

    fees_eur = -(_mk_fee("DA") + _mk_fee("ID"))

    trade_steps = 0
    if commit is not None and not commit.empty:
        if ("committed_new" in commit.columns) and ("commit_now" in commit.columns):
            trade_steps = int(
                commit[(commit["committed_new"] != 0.0) & (commit["commit_now"] == True)].shape[0]
            )

    imb_cost_eur = 0.0
    if "IMB Cashflow [€/step]" in cf_df.columns:
        imb_cost_eur = float(cf_df["IMB Cashflow [€/step]"].sum())

    gross_profit_eur = float(cf_df["Total Cashflow [€/step]"].sum()) + fees_eur                             # (market CF + IMB CF (neg)) + fees (neg)
    trading_profit_eur = float(cf_df["Total Cashflow [€/step]"].sum()) - imb_cost_eur                       # (market CF + IMB CF (neg)) - IMB CF (neg) = market CF


    return {
        "gross_profit_eur": gross_profit_eur,
        "trading_profit_eur": trading_profit_eur,
        "fees_eur": float(fees_eur),
        "imb_cost_eur": float(imb_cost_eur),
        "trade_steps": int(trade_steps),
        "net_kwh": float(net_kwh),
        "sell_kwh": float(sell_kwh),
        "buy_kwh": float(buy_kwh),
    }

