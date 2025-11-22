import argparse
import yaml
from pathlib import Path

from flex_dep_opt.workflows.optimize_workflow import run_optimize
from flex_dep_opt.workflows.plot_workflow import run_plot
from flex_dep_opt.workflows.price_generation_workflow import (run_generate_prices_DA,run_import_epex_DA,run_generate_prices_ID)


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

    p_gen = sub.add_parser("generate-prices-DA")
    p_gen.add_argument("--out", default="data/example_prices_DA.csv")

    p_gen = sub.add_parser("generate-prices-ID")
    p_gen.add_argument("--out", default="data/example_prices_ID.csv")

    p_epex = sub.add_parser("import-epex-DA")
    p_epex.add_argument("src", help="Path to raw EPEX CSV file")
    p_epex.add_argument("--out", default="data/epex_prices_DA.csv")

    args = parser.parse_args()

    # run optimize
    if args.cmd == "optimize":
        cfg = load_settings(args.config)
        run_optimize(cfg)
        return

    # run plot
    if args.cmd == "plot-results":
        cfg = load_settings(args.config)
        run_plot(cfg)
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