from __future__ import annotations

import logging
import multiprocessing as mp
import os
import tomllib
import traceback
from functools import partial
from pathlib import Path

import pandas as pd

from flex_dep_opt._logging import configure_logging
from flex_dep_opt.config.settings import Settings
from flex_dep_opt.io.results import make_run_dir
from flex_dep_opt.workflows.mpc_workflow import run_mpc
from flex_dep_opt.workflows.postprocessing_workflow import postprocess_mpc_results

logger = logging.getLogger(__name__)

DEFAULT_TOML_NAME = "default.toml"


def _make_batch_results_root(configs_dir: Path) -> Path:
    batches_root = Path("batches").resolve()
    try:
        batch_name = configs_dir.resolve().relative_to(batches_root)
    except ValueError:
        batch_name = Path(configs_dir.name)
    batch_label = batch_name.as_posix().replace("/", "_")
    return make_run_dir(Path("results") / "batches", batch_label)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_merged(config_path: str, default_path: str | None) -> Settings:
    if default_path is None:
        return Settings.from_toml(config_path)
    with open(default_path, "rb") as f:
        base = tomllib.load(f)
    with open(config_path, "rb") as f:
        merged = _deep_merge(base, tomllib.load(f))
    return Settings.from_dict(merged, source_path=config_path)


def _run_one(config_path: str, default_path: str | None = None, results_root: str | None = None) -> dict:
    configure_logging()  # batch workers are spawned processes and start without handlers
    run_dir: Path | None = Path(results_root) / Path(config_path).stem if results_root else None
    # sim_start/sim_end make the manifest self-contained for annualization,
    # so plots work from the manifest.csv alone (run dirs are often too big
    # to copy off the compute machine)
    sim_times: dict[str, str] = {}
    try:
        settings = _load_merged(config_path, default_path)
        sim_times = {
            "sim_start": settings.simulation.start.isoformat(),
            "sim_end": settings.simulation.end.isoformat(),
        }
        run_dir = run_mpc(settings, run_dir=run_dir)
        postprocess_mpc_results(settings, run_dir=run_dir)

        row = {"config": config_path, "status": "ok", "run_dir": str(run_dir), **sim_times}
        kpis_csv = run_dir / "kpis.csv"
        if kpis_csv.exists():
            row.update(pd.read_csv(kpis_csv).iloc[0].to_dict())
        return row

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Run failed for %s:\n%s", config_path, tb)
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "FAILED.txt").write_text(
                f"Config: {config_path}\nError: {e}\n\n{tb}", encoding="utf-8"
            )
        return {
            "config": config_path,
            "status": "failed",
            "run_dir": str(run_dir) if run_dir else "",
            **sim_times,
            "error": str(e),
        }


def run_batch(
    configs_dir: str | Path, jobs: int | None = None, manifest: str | Path | None = None
) -> pd.DataFrame:
    """
    Simulate and postprocess every TOML config in `configs_dir` and return the manifest.

    A `default.toml` in the directory is the base that each config overrides. The manifest has
    one row per config with its status, run directory and KPIs and is also written to `manifest`
    (default: results/batches/<batch>__<timestamp>/manifest.csv). `jobs` defaults to all CPU cores.
    """
    configs_dir = Path(configs_dir)
    jobs = jobs or os.cpu_count() or 1
    default_file = configs_dir / DEFAULT_TOML_NAME
    default_path = str(default_file) if default_file.is_file() else None

    configs = sorted(str(p) for p in configs_dir.glob("*.toml") if p.name != DEFAULT_TOML_NAME)
    if not configs:
        raise FileNotFoundError(f"No *.toml configs found in {configs_dir}")

    results_root = str(_make_batch_results_root(configs_dir))
    run = partial(_run_one, default_path=default_path, results_root=results_root)
    if jobs > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(jobs) as pool:
            rows = pool.map(run, configs)
    else:
        rows = [run(c) for c in configs]

    manifest = Path(manifest) if manifest else Path(results_root) / "manifest.csv"
    table = pd.DataFrame(rows)
    table.to_csv(manifest, index=False)

    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_failed = sum(1 for r in rows if r.get("status") == "failed")
    status_str = f"{n_ok} ok, {n_failed} failed" if n_failed else f"{n_ok} ok"
    logger.info(f"Batch done: {len(rows)} runs ({status_str}) -> {manifest}")
    return table
