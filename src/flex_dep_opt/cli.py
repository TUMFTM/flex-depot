import argparse
import yaml

from flex_dep_opt.workflows.mpc_workflow import run_mpc
from flex_dep_opt.workflows.optimize_workflow import run_optimize
from flex_dep_opt.workflows.plot_workflow import run_plot, run_plot_mpc, run_plot_mpc_onepager
from flex_dep_opt.workflows.price_generation_workflow import (run_generate_prices_DA,run_import_epex_DA,run_generate_prices_ID, run_import_reBAP)


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

    p_gen = sub.add_parser("generate-prices-DA")
    p_gen.add_argument("--out", default="data/example_prices_DA.csv")

    p_gen = sub.add_parser("generate-prices-ID")
    p_gen.add_argument("--out", default="data/example_prices_ID.csv")

    p_epex = sub.add_parser("import-epex-DA")
    p_epex.add_argument("src", help="Path to raw EPEX CSV file")
    p_epex.add_argument("--out", default="data/epex_prices_DA.csv")

    p_reBAP = sub.add_parser("import-reBAP")
    p_reBAP.add_argument("src", help="Path to raw reBAP CSV file")

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
    if args.cmd == "plot-results":
        cfg = load_settings(args.config)
        run_plot(cfg)
        return

    if args.cmd == "plot-results-mpc":
        cfg = load_settings(args.config)
        run_plot_mpc(cfg)
        return

    # generate dummy DA prices
    if args.cmd == "generate-prices-DA":
        run_generate_prices_DA(args.out)
        return

    # import DA EPEX data
    if args.cmd == "import-epex-DA":
        run_import_epex_DA(args.src, args.out)
        return

    # generate dummy ID prices
    if args.cmd == "generate-prices-ID":
        run_generate_prices_ID(args.out)
        return

    # import reBAP data
    if args.cmd == "import-reBAP":
        run_import_reBAP(args.src)
        return