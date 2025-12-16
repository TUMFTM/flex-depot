import argparse
import yaml

from flex_dep_opt.workflows.mpc_workflow import run_mpc
from flex_dep_opt.workflows.optimize_workflow import run_optimize
from flex_dep_opt.workflows.plot_workflow import run_plot_mpc


def load_settings(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(prog="flex-dep-opt")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_opt = sub.add_parser("optimize")
    p_opt.add_argument("--config", default="settings.yaml")

    p_mpc = sub.add_parser("mpc")
    p_mpc.add_argument("--config", default="settings.yaml")

    p_plot = sub.add_parser("plot-results")
    p_plot.add_argument("--config", default="settings.yaml")

    p_plot_mpc = sub.add_parser("plot-results-mpc")
    p_plot_mpc.add_argument("--config", default="settings.yaml")

    args = parser.parse_args()

    # run optimize
    if args.cmd == "optimize":
        cfg = load_settings(args.config)
        run_optimize(cfg)
        return

    # run mpc (Rolling Horizon)
    if args.cmd == "mpc":
        cfg = load_settings(args.config)
        run_mpc(cfg)
        return

    # run plot
    if args.cmd == "plot-results-mpc":
        cfg = load_settings(args.config)
        run_plot_mpc(cfg)
        return
