"""Batch of two-hour slices of the quick-start example."""

from pathlib import Path

import pandas as pd
import pytest

from flex_dep_opt.batch import run_batch

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_TOML = REPO_ROOT / "src/flex_dep_opt/config/settings_quickstart.toml"


def test_manifest_has_one_row_per_config_and_records_failures(tmp_path, monkeypatch):
    # data paths in the settings are relative to the repository root
    monkeypatch.chdir(REPO_ROOT)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "default.toml").write_text(EXAMPLE_TOML.read_text(encoding="utf-8"), encoding="utf-8")
    short = '[simulation]\nend = "2026-02-06 02:00"\nwarmup_hours = 0\n'
    (configs / "good.toml").write_text(short)
    (configs / "bad.toml").write_text(short + '[optimization.markets.dayahead]\nsource = "missing.csv"\n')
    monkeypatch.setattr("flex_dep_opt.batch._make_batch_results_root", lambda _: tmp_path / "results")

    manifest = run_batch(configs, jobs=1, manifest=tmp_path / "manifest.csv")

    assert manifest.set_index(manifest["config"].map(lambda p: Path(p).name))["status"].to_dict() == {
        "bad.toml": "failed",
        "good.toml": "ok",
    }
    assert manifest["gross_profit_eur"].notna().tolist() == [False, True]
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "manifest.csv"), manifest, check_dtype=False)


def test_directory_without_configs_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="No \\*.toml configs found"):
        run_batch(tmp_path)
