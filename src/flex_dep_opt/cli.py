import argparse
import os
from pathlib import Path

from flex_dep_opt._logging import configure_logging
from flex_dep_opt.batch import run_batch
from flex_dep_opt.config.settings import Settings
from flex_dep_opt.errors import SimulationError
from flex_dep_opt.workflows.mpc_workflow import run_mpc
from flex_dep_opt.workflows.postprocessing_workflow import postprocess_mpc_results


def main():
    parser = argparse.ArgumentParser(prog="flex-depot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_mpc = sub.add_parser("run-sim")
    p_mpc.add_argument(
        "--config",
        default=None,
        help="Path to a TOML config file (default: bundled settings_quickstart.toml).",
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
    configure_logging()

    if args.cmd == "run-sim":
        try:
            run_mpc(Settings.from_toml(args.config), run_dir=Path(args.run_dir) if args.run_dir else None)
        except SimulationError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc
        return

    if args.cmd == "run-post":
        settings = Settings.from_toml(args.config) if args.config else None
        postprocess_mpc_results(settings, run_dir=Path(args.run_dir) if args.run_dir else None)
        return

    if args.cmd == "run-batch":
        try:
            run_batch(args.configs_dir, jobs=args.jobs, manifest=args.manifest)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        return
