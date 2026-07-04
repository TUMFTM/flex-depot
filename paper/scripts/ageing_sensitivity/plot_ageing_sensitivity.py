"""Placeholder: paper plots for the ageing sensitivity batch.

Run configs are named f<fleet>_c<idx> (e.g. f1_c05), varying the cycle
regularization cost (cost_eur_per_kwh_throughput). Will be implemented once
the batch results are available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paper-ready ageing sensitivity plots from a batch manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the batch manifest.csv.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("paper/figures/ageing_sensitivity/ageing_sensitivity"),
        help="Output path without extension. PDF, SVG, PNG and CSV are written.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG export DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.exit(
        f"Not implemented yet: ageing sensitivity plotting (manifest: {args.manifest}). "
        "Waiting for batch results."
    )


if __name__ == "__main__":
    main()
