import argparse

from flex_dep_opt.config.settings import Settings

from flex_dep_opt.workflows.mpc_workflow import run_mpc
from flex_dep_opt.workflows.postprocessing_workflow import postprocess_mpc_results

def main():
    parser = argparse.ArgumentParser(prog="flex-dep-opt")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_mpc = sub.add_parser("run-sim")
    p_mpc.add_argument("--config", default="settings_example.yaml")

    p_plot_mpc = sub.add_parser("run-post")
    p_plot_mpc.add_argument("--config", default="settings_example.yaml")

    args = parser.parse_args()

    settings = Settings()

    if args.cmd == "run-sim":
        run_mpc(settings)
        return

    if args.cmd == "run-post":
        postprocess_mpc_results(settings)
        return