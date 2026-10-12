"""In-memory simulation of a two-hour slice of the quick-start example."""

from pathlib import Path

import pandas as pd
import pytest

from flex_dep_opt.config.settings import Settings
from flex_dep_opt.errors import SimulationError
from flex_dep_opt.io.inputs import load_inputs
from flex_dep_opt.workflows import mpc_workflow
from flex_dep_opt.workflows.mpc_workflow import run_mpc, simulate

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_TOML = REPO_ROOT / "src/flex_dep_opt/config/settings_quickstart.toml"


@pytest.fixture
def settings(monkeypatch) -> Settings:
    # data paths in the settings are relative to the repository root
    monkeypatch.chdir(REPO_ROOT)
    settings = Settings.from_toml(EXAMPLE_TOML)
    settings.simulation.end = settings.simulation.start + pd.Timedelta(hours=2)
    settings.simulation.warmup_hours = 0.0
    return settings


def test_simulate_returns_one_dispatch_row_per_step(settings):
    inputs = load_inputs(settings)

    result = simulate(settings, inputs)

    assert len(result.dispatch) == len(inputs.index)
    assert result.dispatch.index.equals(inputs.index)
    assert not result.commits.empty
    assert result.settings is settings


def _make_third_step_infeasible(monkeypatch):
    calls = {"n": 0}
    solve_model = mpc_workflow.solve_model

    def fail_on_third_step(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("no feasible solution")
        return solve_model(*args, **kwargs)

    monkeypatch.setattr(mpc_workflow, "solve_model", fail_on_third_step)


def test_infeasible_step_raises_with_the_partial_result(settings, monkeypatch):
    settings.optimization.imbalance.enabled = False
    inputs = load_inputs(settings)
    _make_third_step_infeasible(monkeypatch)

    with pytest.raises(SimulationError, match="No feasible solution at 2026-02-06 00:30") as excinfo:
        simulate(settings, inputs)

    assert excinfo.value.partial_result.dispatch.index.equals(inputs.index[:2])


def test_run_mpc_saves_the_partial_result_before_raising(settings, tmp_path, monkeypatch):
    settings.optimization.imbalance.enabled = False
    _make_third_step_infeasible(monkeypatch)

    with pytest.raises(SimulationError):
        run_mpc(settings, run_dir=tmp_path)

    assert len(pd.read_csv(tmp_path / "dispatch.csv")) == 2
