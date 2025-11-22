import argparse
import yaml
from pathlib import Path

from flex_dep_opt.workflows.optimize_workflow import run_optimize
from flex_dep_opt.workflows.plot_workflow import run_plot


def load_settings(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(prog="flex-dep-opt")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_opt = sub.add_parser("optimize")
    p_opt.add_argument("--config", default="settings.yaml")

    p_plot = sub.add_parser("plot-results")
    p_plot.add_argument("--config", default="settings.yaml")

    args = parser.parse_args()

    cfg = load_settings(args.config)

    if args.cmd == "optimize":
        run_optimize(cfg)

    elif args.cmd == "plot-results":
        run_plot(cfg)