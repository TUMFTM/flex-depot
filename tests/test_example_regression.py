"""
End-to-end regression test for the illustrative example (HiGHS solver).

Runs the full MPC + postprocessing pipeline against the bundled example data
and checks that the resulting KPIs match the golden reference values.  The
test catches silent regressions in the optimisation logic, gate-closure rules,
FCR bidding, and the postprocessing chain.

Golden values were produced with HiGHS on 2026-07-14 using the simulation
window 2026-01-08 to 2026-01-12 (4 days, 384 steps at 15-min resolution).
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
    "gross_profit_eur": 906.104,
    "trading_profit_eur": 393.023,
    "fees_eur": -0.907,
    "fcr_revenue_eur": 513.988,
    # Savings vs. uncontrolled charging (reviewer R1.4)
    "total_potential_gross_profit_delta_eur": 1486.608,
    # Energy balance
    "net_kwh": -288.871,
    "sell_kwh": 5442.456,
    "buy_kwh": 5731.327,
    # Integer counts — exact
    "trade_steps": 310,
    "fcr_slots_committed": 15,
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
