"""
End-to-end regression test for the illustrative example (HiGHS solver).

Runs the full MPC + postprocessing pipeline against the bundled example data
and checks that the resulting KPIs match the golden reference values.  The
test catches silent regressions in the optimisation logic, gate-closure rules,
FCR bidding, and the postprocessing chain.

Golden values were produced with HiGHS on 2026-07-17 using the simulation
window 2026-02-06 to 2026-02-10 (4 days, 384 steps at 15-min resolution) —
the 4-day detail window of the illustrative example (S3 setup, see
examples/illustrative_example/README.md).
"""

from pathlib import Path

import pandas as pd
import pytest

from flex_dep_opt.config.settings import Settings
from flex_dep_opt.workflows.mpc_workflow import run_mpc
from flex_dep_opt.workflows.postprocessing_workflow import postprocess_mpc_results

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_TOML = REPO_ROOT / "src/flex_dep_opt/config/settings_example.toml"

# Golden KPI values — update this dict after any intentional model change.
_GOLDEN = {
    # Core economics
    "gross_profit_eur": 126.011,
    "trading_profit_eur": -64.454,
    "fees_eur": -1.235,
    "fcr_revenue_eur": 191.700,
    # Savings vs. uncontrolled charging
    "total_potential_gross_profit_delta_eur": 512.141,
    # Energy balance
    "net_kwh": -3061.612,
    "sell_kwh": 5727.062,
    "buy_kwh": 8788.674,
    # Integer counts — exact
    "trade_steps": 220,
    "fcr_slots_committed": 10,
    "pass2_steps": 0,
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
        "total_potential_gross_profit_delta_eur",
        "net_kwh",
        "sell_kwh",
        "buy_kwh",
    ):
        assert float(kpis[key]) == pytest.approx(_GOLDEN[key], rel=1e-3), f"KPI mismatch: {key}"

    assert float(kpis["imb_cost_eur"]) == pytest.approx(0.0, abs=1e-6)
