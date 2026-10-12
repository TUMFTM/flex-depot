"""Saving a SimulationResult to a run directory (no MPC runs)."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from flex_dep_opt.config.settings import Settings
from flex_dep_opt.io.results import SimulationResult

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_TOML = REPO_ROOT / "src/flex_dep_opt/config/settings_quickstart.toml"
TZ = ZoneInfo("Europe/Berlin")


def _result(**tables) -> SimulationResult:
    empty = pd.DataFrame()
    return SimulationResult(
        settings=Settings.from_toml(EXAMPLE_TOML),
        dispatch=tables.get("dispatch", empty),
        commits=tables.get("commits", empty),
        fcr_commits=tables.get("fcr_commits", empty),
        started_at=datetime(2026, 2, 6, 12, 0, tzinfo=TZ),
        finished_at=datetime(2026, 2, 6, 12, 5, tzinfo=TZ),
    )


def test_save_writes_tables_settings_and_run_info(tmp_path):
    time = pd.Timestamp("2026-02-06 00:00", tz="UTC")
    result = _result(
        dispatch=pd.DataFrame({"time": [time], "p_net_kw": [100.0]}),
        commits=pd.DataFrame({"delivery_time": [time], "current_time": [time], "market": ["DA"]}),
    )

    run_dir = result.save(tmp_path / "run")

    assert sorted(p.name for p in run_dir.iterdir()) == [
        "commit.csv",
        "dispatch.csv",
        "run_info.txt",
        "settings.toml",
    ]
    assert pd.read_csv(run_dir / "dispatch.csv")["p_net_kw"].tolist() == [100.0]
    assert Settings.from_toml(run_dir / "settings.toml").model_dump() == result.settings.model_dump()
    assert "duration_seconds: 300.00" in (run_dir / "run_info.txt").read_text()


def test_save_skips_empty_tables(tmp_path):
    run_dir = _result().save(tmp_path / "run")

    assert sorted(p.name for p in run_dir.iterdir()) == ["run_info.txt", "settings.toml"]
