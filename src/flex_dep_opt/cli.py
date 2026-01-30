import argparse
import yaml

from flex_dep_opt.workflows.mpc_workflow import run_mpc
from flex_dep_opt.workflows.postprocessing_workflow import postprocess_mpc_results
from flex_dep_opt.workflows.postprocessing_workflow_paper import postprocess_mpc_results_paper


def load_settings(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(prog="flex-dep-opt")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_mpc = sub.add_parser("run-sim")
    p_mpc.add_argument("--config", default="settings_example.yaml")

    p_plot_mpc = sub.add_parser("run-post")
    p_plot_mpc.add_argument("--config", default="settings_example.yaml")

    p_plot_mpc = sub.add_parser("plot-results-mpc-paper")
    p_plot_mpc.add_argument("--config", default="settings_example.yaml")

    args = parser.parse_args()

    # run mpc (Rolling Horizon)
    if args.cmd == "run-sim":
        cfg = load_settings(args.config)
        run_mpc(cfg)
        return

    # run plot
    if args.cmd == "run-post":
        cfg = load_settings(args.config)
        postprocess_mpc_results(cfg)
        return

    # run paper plot
    if args.cmd == "plot-results-mpc-paper":
        cfg = load_settings(args.config)
        postprocess_mpc_results_paper(cfg)
        return