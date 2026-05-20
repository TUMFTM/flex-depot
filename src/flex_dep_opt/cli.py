import argparse

from flex_dep_opt.config.settings import Settings

from flex_dep_opt.workflows.mpc_workflow import run_mpc
from flex_dep_opt.workflows.postprocessing_workflow import postprocess_mpc_results

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

    args = parser.parse_args()

    if args.cmd == "run-sim":
        run_mpc(Settings.load(args.config))
        return

    if args.cmd == "run-post":
        settings = Settings.load(args.config) if args.config else None
        postprocess_mpc_results(settings)
        return
