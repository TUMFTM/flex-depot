"""Self-check for the FCR breakeven math (committed vs. declined orientation)."""

from flex_dep_opt.workflows.mpc_workflow import _fcr_breakeven_metrics


def _terms(objective, **deltas):
    keys = (
        "obj_energy_cashflow",
        "obj_fee_cost",
        "obj_cycling_cost",
        "obj_imb_cashflow",
        "obj_imb_vol_penalty",
        "obj_term_penalty",
        "obj_e_slack_penalty",
        "obj_reserve_penalty",
        "obj_balance_penalty",
    )
    d = {k: 0.0 for k in keys}
    d.update(deltas)
    d["objective"] = objective
    return d


def test_committed_bid_has_positive_margin():
    # Model took a 1 MW bid at 30 EUR/MW. Dropping it recovers 10 EUR of other
    # objective -> breakeven 10 EUR/MW, margin 20 EUR/MW (>0, worth taking).
    rev = 30.0  # price 30 * 1 MW * cf 1.0
    with_bid = _terms(100.0 + rev)  # obj_A includes revenue
    without_bid = _terms(100.0 + 10.0)  # obj_B: rest improves by 10
    m = _fcr_breakeven_metrics(
        with_bid, without_bid, bid_kw=1000.0, rev_eur=rev, fcr_price=30.0, covered_fraction=1.0
    )
    assert abs(m["breakeven_eur_per_mw"] - 10.0) < 1e-9
    assert abs(m["margin_eur_per_mw"] - 20.0) < 1e-9


def test_declined_bid_has_non_positive_margin():
    # Model declined a slot at price 30. Forcing 1 MW in costs 50 EUR of other
    # objective -> breakeven 50 EUR/MW, margin -20 EUR/MW (<0, not worth taking).
    rev = 30.0
    forced_in = _terms(100.0 - 50.0 + rev)  # obj_B (with bid): rest dropped 50, plus rev
    actual = _terms(100.0)  # obj_A (without bid)
    m = _fcr_breakeven_metrics(
        forced_in, actual, bid_kw=1000.0, rev_eur=rev, fcr_price=30.0, covered_fraction=1.0
    )
    assert abs(m["breakeven_eur_per_mw"] - 50.0) < 1e-9
    assert m["margin_eur_per_mw"] <= 0.0


if __name__ == "__main__":
    test_committed_bid_has_positive_margin()
    test_declined_bid_has_non_positive_margin()
    print("ok")
