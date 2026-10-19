"""
End-to-end regression test: the full MPC + postprocessing run of the quick-start example
(2026-02-06 to 2026-02-10, HiGHS) must reproduce the stored KPIs.

The nonzero pass-2 steps and imbalance cost are expected: FCR activation in the current
quarter-hour can no longer be netted on the intraday market and settles via reBAP (see the
modelling note in examples/illustrative_example/README.md).
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
    "gross_profit_eur": -72.853,
    "cycling_cost_eur": 297.260,
    "trading_profit_eur": -16.532,
    "fees_eur": -1.075,
    "fcr_revenue_eur": 210.020,
    "imb_cost_eur": -14.162,
    "fcr_activation_cf_eur": 46.156,
    # Savings vs. uncontrolled charging (S0 reference metered grid-side; charging
    # losses already baked into the flexband column, no extra eta division)
    "total_potential_gross_profit_delta_eur": 380.899,
    # Energy balance
    "net_kwh": -2771.777,
    "sell_kwh": 5699.440,
    "buy_kwh": 8471.217,
    # Integer counts — exact
    "trade_steps": 174,
    "fcr_slots_committed": 11,
    "pass2_steps": 12,
}


@pytest.mark.slow
def test_example_kpis(tmp_path, monkeypatch):
    """Full MPC + postprocessing run produces the expected KPIs."""
    monkeypatch.chdir(REPO_ROOT)

    settings = Settings.from_toml(EXAMPLE_TOML)
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
        "cycling_cost_eur",
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
