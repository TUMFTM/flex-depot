import argparse
import logging
import multiprocessing as mp
import os
import tempfile
import traceback
from functools import partial
from pathlib import Path

import pandas as pd
import toml
import tomllib

from flex_dep_opt.config.settings import Settings
from flex_dep_opt.io.results_io import make_run_dir
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
        return Settings.load(config_path)
    with open(default_path, "rb") as f:
        base = tomllib.load(f)
    with open(config_path, "rb") as f:
        merged = _deep_merge(base, tomllib.load(f))
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tf:
        toml.dump(merged, tf)
        tmp = tf.name
    try:
        return Settings.load(tmp)
    finally:
        os.unlink(tmp)


def _run_one(config_path: str, default_path: str | None = None, results_root: str | None = None) -> dict:
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


def main():
    parser = argparse.ArgumentParser(prog="flex-dep-opt")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_mpc = sub.add_parser("run-sim")
    p_mpc.add_argument(
        "--config",
        default=None,
        help="Path to a TOML config file (default: bundled settings_example.toml).",
    )
    p_mpc.add_argument(
        "--run-dir",
        default=None,
        help="Explicit output directory for this run (default: results/<name>__<timestamp>).",
    )

    p_plot_mpc = sub.add_parser("run-post")
    p_plot_mpc.add_argument(
        "--config",
        default=None,
        help="Path to a TOML config file (default: the settings.toml saved in the latest run directory).",
    )
    p_plot_mpc.add_argument(
        "--run-dir",
        default=None,
        help="Run directory to postprocess (default: the latest run from results/LATEST.txt).",
    )

    p_batch = sub.add_parser(
        "run-batch", help="Run sim+post for every TOML in a directory, write a KPI manifest."
    )
    p_batch.add_argument("configs_dir", help="Directory containing one TOML config per run.")
    p_batch.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel processes (default: all CPU cores).",
    )
    p_batch.add_argument(
        "--manifest",
        default=None,
        help="Manifest CSV path (default: results/batches/<batch>__<timestamp>/manifest.csv).",
    )

    args = parser.parse_args()

    if args.cmd == "run-sim":
        run_mpc(Settings.load(args.config), run_dir=Path(args.run_dir) if args.run_dir else None)
        return

    if args.cmd == "run-post":
        settings = Settings.load(args.config) if args.config else None
        postprocess_mpc_results(settings, run_dir=Path(args.run_dir) if args.run_dir else None)
        return

    if args.cmd == "run-batch":
        configs_dir = Path(args.configs_dir)
        default_file = configs_dir / DEFAULT_TOML_NAME
        default_path = str(default_file) if default_file.is_file() else None

        configs = sorted(str(p) for p in configs_dir.glob("*.toml") if p.name != DEFAULT_TOML_NAME)
        if not configs:
            raise SystemExit(f"No *.toml configs found in {args.configs_dir}")

        results_root = str(_make_batch_results_root(configs_dir))
        run = partial(_run_one, default_path=default_path, results_root=results_root)
        if args.jobs > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.jobs) as pool:
                rows = pool.map(run, configs)
        else:
            rows = [run(c) for c in configs]

        manifest = Path(args.manifest) if args.manifest else Path(results_root) / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)

        n_ok = sum(1 for r in rows if r.get("status") == "ok")
        n_failed = sum(1 for r in rows if r.get("status") == "failed")
        status_str = f"{n_ok} ok, {n_failed} failed" if n_failed else f"{n_ok} ok"
        print(f"Batch done: {len(rows)} runs ({status_str}) -> {manifest}")
        return
