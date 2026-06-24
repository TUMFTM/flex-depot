import argparse
import multiprocessing as mp
import os
import tempfile
import tomllib
from functools import partial
from pathlib import Path

import pandas as pd
import toml

from flex_dep_opt.config.settings import Settings
from flex_dep_opt.workflows.mpc_workflow import run_mpc
from flex_dep_opt.workflows.postprocessing_workflow import postprocess_mpc_results

DEFAULT_TOML_NAME = "default.toml"


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


def _run_one(
    config_path: str, default_path: str | None = None, results_root: str | None = None
) -> dict:
    settings = _load_merged(config_path, default_path)
    run_dir = Path(results_root) / Path(config_path).stem if results_root else None
    run_dir = run_mpc(settings, run_dir=run_dir)
    postprocess_mpc_results(settings, run_dir=run_dir)

    row = {"config": config_path, "run_dir": str(run_dir)}
    kpis_csv = run_dir / "kpis.csv"
    if kpis_csv.exists():
        row.update(pd.read_csv(kpis_csv).iloc[0].to_dict())
    return row


def main():
    parser = argparse.ArgumentParser(prog="flex-dep-opt")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_mpc = sub.add_parser("run-sim")
    p_mpc.add_argument(
        "--config", default=None,
        help="Path to a TOML config file (default: bundled settings_example.toml).",
    )

    p_plot_mpc = sub.add_parser("run-post")
    p_plot_mpc.add_argument(
        "--config", default=None,
        help="Path to a TOML config file (default: the settings.toml saved in the latest run directory).",
    )

    p_batch = sub.add_parser(
        "run-batch", help="Run sim+post for every TOML in a directory, write a KPI manifest."
    )
    p_batch.add_argument("configs_dir", help="Directory containing one TOML config per run.")
    p_batch.add_argument("--jobs", type=int, default=1, help="Parallel processes (default: 1).")
    p_batch.add_argument(
        "--manifest", default=None,
        help="Manifest CSV path (default: <configs_dir>/manifest.csv).",
    )

    args = parser.parse_args()

    if args.cmd == "run-sim":
        run_mpc(Settings.load(args.config))
        return

    if args.cmd == "run-post":
        settings = Settings.load(args.config) if args.config else None
        postprocess_mpc_results(settings)
        return

    if args.cmd == "run-batch":
        configs_dir = Path(args.configs_dir)
        default_file = configs_dir / DEFAULT_TOML_NAME
        default_path = str(default_file) if default_file.is_file() else None

        configs = sorted(
            str(p) for p in configs_dir.glob("*.toml") if p.name != DEFAULT_TOML_NAME
        )
        if not configs:
            raise SystemExit(f"No *.toml configs found in {args.configs_dir}")

        results_root = str(configs_dir / "results")
        run = partial(_run_one, default_path=default_path, results_root=results_root)
        if args.jobs > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.jobs) as pool:
                rows = pool.map(run, configs)
        else:
            rows = [run(c) for c in configs]

        manifest = Path(args.manifest) if args.manifest else Path(args.configs_dir) / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        print(f"Batch done: {len(rows)} runs → {manifest}")
        return
