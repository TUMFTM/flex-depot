from types import SimpleNamespace

import pandas as pd

from flex_dep_opt.market.trading import gate_closure_timestamp

_CFG = SimpleNamespace(
    trading=SimpleNamespace(
        dayahead=SimpleNamespace(gate_closure_hour="12:00", closes_previous_day=True),
        intraday=SimpleNamespace(offset_minutes_before_delivery=60),
    )
)


def test_da_gate_closure_d_minus_1():
    delivery = pd.Timestamp("2025-10-02 10:00", tz="Europe/Berlin")
    gc = gate_closure_timestamp("DA", delivery, _CFG)
    assert gc == pd.Timestamp("2025-10-01 12:00", tz="Europe/Berlin")


def test_id_gate_closure_offset():
    delivery = pd.Timestamp("2025-10-02 10:00", tz="Europe/Berlin")
    gc = gate_closure_timestamp("ID", delivery, _CFG)
    assert gc == pd.Timestamp("2025-10-02 09:00", tz="Europe/Berlin")


def test_gate_rules_evaluated_in_market_tz_for_utc_delivery():
    # Gate rules are Berlin-local even when the delivery time is UTC; the result
    # is returned in the delivery's own tz. 08:00 UTC == 10:00 Berlin delivery,
    # so the D-1 12:00 Berlin gate is 10:00 UTC.
    delivery = pd.Timestamp("2025-10-02 08:00", tz="UTC")
    gc = gate_closure_timestamp("DA", delivery, _CFG)
    assert gc == pd.Timestamp("2025-10-01 10:00", tz="UTC")
    assert str(gc.tz) == "UTC"


if __name__ == "__main__":
    test_da_gate_closure_d_minus_1()
    test_id_gate_closure_offset()
    test_gate_rules_evaluated_in_market_tz_for_utc_delivery()
    print("ok")
