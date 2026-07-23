"""
End-to-end regression test for the illustrative example (HiGHS solver).

Runs the full MPC + postprocessing pipeline against the bundled example data
and checks that the resulting KPIs match the golden reference values.  The
test catches silent regressions in the optimisation logic, gate-closure rules,
FCR bidding, and the postprocessing chain.

Golden values were produced with HiGHS on 2026-07-21 using the simulation
window 2026-02-06 to 2026-02-10 (4 days, 384 steps at 15-min resolution) —
the 4-day detail window of the illustrative example (S3 setup, see
examples/illustrative_example/README.md), with the realistic 5-min intraday
gate closure (offset_minutes_before_delivery = 5). The nonzero PASS2 steps
and imbalance cost are expected: the current quarter-hour's FCR activation
can no longer be netted on the intraday market, so residuals settle via
reBAP — consistent with German practice, where FCR activation energy has no
ex-post balancing-group correction (unlike aFRR/mFRR) and remains in the
provider's balancing group (see the modelling note in the example README).

Option A (FCR activation cashflow) was added 2026-07-21:
  - gross_profit_eur increased by fcr_activation_cf_eur = +51.368 EUR
  - total_potential_gross_profit_delta_eur increased by the same amount
  - trading_profit_eur, dispatch volumes, and all integer counts unchanged
"""

from pathlib import Path

import pandas as pd
import pytest

from flex_dep_opt.config.settings import Settings
from flex_dep_opt.workflows.mpc_workflow import run_mpc
from flex_dep_opt.workflows.postprocessing_workflow import postprocess_mpc_results

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_TOML = REPO_ROOT / "src/flex_dep_opt/config/settings_quickstart.toml"

# Golden KPI values — update this dict after any intentional model change.
_GOLDEN = {
    # Core economics
    "gross_profit_eur": 172.650,
    "trading_profit_eur": -55.035,
    "fees_eur": -1.221,
    "fcr_revenue_eur": 191.700,
    "imb_cost_eur": -14.162,
    "fcr_activation_cf_eur": 51.368,
    # Savings vs. uncontrolled charging
    "total_potential_gross_profit_delta_eur": 558.780,
    # Energy balance
    "net_kwh": -2985.521,
    "sell_kwh": 5693.942,
    "buy_kwh": 8679.464,
    # Integer counts — exact
    "trade_steps": 161,
    "fcr_slots_committed": 10,
    "pass2_steps": 12,
}


@pytest.mark.slow
def test_example_kpis(tmp_path, monkeypatch):
    """Full MPC + postprocessing run produces the expected KPIs."""
    monkeypatch.chdir(REPO_ROOT)

    settings = Settings.load(EXAMPLE_TOML)
    run_dir = run_mpc(settings, run_dir=tmp_path)
    postprocess_mpc_results(settings, run_dir=run_dir)

    kpis = pd.read_csv(tmp_path / "kpis.csv").iloc[0]

    # Integer KPIs: exact
    assert int(kpis["trade_steps"]) == _GOLDEN["trade_steps"]
    assert int(kpis["fcr_slots_committed"]) == _GOLDEN["fcr_slots_committed"]
    assert int(kpis["pass2_steps"]) == _GOLDEN["pass2_steps"]

    # Float KPIs: 0.1 % relative tolerance
    for key in (
        "gross_profit_eur",
        "trading_profit_eur",
        "fees_eur",
        "fcr_revenue_eur",
        "imb_cost_eur",
        "fcr_activation_cf_eur",
        "total_potential_gross_profit_delta_eur",
        "net_kwh",
        "sell_kwh",
        "buy_kwh",
    ):
        assert float(kpis[key]) == pytest.approx(_GOLDEN[key], rel=1e-3), f"KPI mismatch: {key}"
