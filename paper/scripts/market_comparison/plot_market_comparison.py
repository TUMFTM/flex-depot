from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    FLEET_SIZES,
    FULL_WIDTH_IN,
    MM_TO_INCH,
    SMALL_FONT_PT,
    TUM_COLORS,
    apply_paper_style,
    manifest_row_annualization,
)

FIG_WIDTH_IN = FULL_WIDTH_IN
FIG_HEIGHT_IN = 115.0 * MM_TO_INCH
BAR_LABEL_FONT_PT = SMALL_FONT_PT

apply_paper_style()

SETUP_ORDER = ["DA", "DA+ID", "DA+ID+FCR"]
SETUP_FROM_SUFFIX = {
    "da": "DA",
    "da_id": "DA+ID",
    "da_id_fcr": "DA+ID+FCR",
}
# TUM palette, market colors consistent with the predecessor paper
# (DA = blue, ID = orange); FCR = green is new here.
MARKET_COLORS = {
    "DA": TUM_COLORS["TUMBlue"],
    "ID": TUM_COLORS["Orange"],
    "FCR": TUM_COLORS["Green"],
}
# One fill color per setup (market colors consistent with the predecessor
# paper); black hatching kept as redundant encoding for grayscale print and
# CVD readers.
FILL_COLORS = {
    "DA": MARKET_COLORS["DA"],
    "DA+ID": MARKET_COLORS["ID"],
    "DA+ID+FCR": MARKET_COLORS["FCR"],
}
HATCHES = {
    "DA": "",
    "DA+ID": "///",
    "DA+ID+FCR": "xxx",
}


def _parse_run_name(row: pd.Series) -> tuple[str, str]:
    candidates = [
        Path(str(row["config"])).stem if pd.notna(row.get("config")) else "",
        Path(str(row["run_dir"])).name if pd.notna(row.get("run_dir")) else "",
    ]
    for name in candidates:
        match = re.fullmatch(r"f(\d+)_(da(?:_id)?(?:_fcr)?)", name)
        if match:
            fleet = f"F{int(match.group(1))}"
            setup = SETUP_FROM_SUFFIX[match.group(2)]
            return fleet, setup
    raise ValueError(f"Could not parse fleet/setup from manifest row: {row.to_dict()}")


def load_plot_data(manifest: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    required = {"config", "run_dir", metric}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    parsed = df.apply(_parse_run_name, axis=1, result_type="expand")
    parsed.columns = ["fleet", "setup"]
    plot_df = pd.concat([parsed, df[[metric]].astype(float)], axis=1)
    unknown_fleets = sorted(set(plot_df["fleet"]) - set(FLEET_SIZES))
    if unknown_fleets:
        raise ValueError(f"No fleet size defined in common.FLEET_SIZES for: {unknown_fleets}")
    annualization = df.apply(manifest_row_annualization, axis=1, manifest_dir=manifest.parent)
    plot_df["n_vehicles"] = plot_df["fleet"].map(FLEET_SIZES)
    plot_df[f"{metric}_per_year"] = plot_df[metric] * annualization
    plot_df[f"{metric}_per_year_per_bet"] = plot_df[f"{metric}_per_year"] / plot_df["n_vehicles"]

    duplicate_mask = plot_df.duplicated(["fleet", "setup"], keep=False)
    if duplicate_mask.any():
        duplicates = plot_df.loc[duplicate_mask, ["fleet", "setup"]]
        raise ValueError(f"Duplicate fleet/setup combinations in manifest:\n{duplicates}")

    fleet_order = sorted(plot_df["fleet"].unique(), key=lambda value: int(value[1:]))
    index = pd.MultiIndex.from_product([fleet_order, SETUP_ORDER], names=["fleet", "setup"])
    plot_df = plot_df.set_index(["fleet", "setup"]).reindex(index).reset_index()
    if plot_df[metric].isna().any():
        missing_rows = plot_df.loc[plot_df[metric].isna(), ["fleet", "setup"]]
        raise ValueError(f"Missing market comparison rows:\n{missing_rows}")

    return plot_df


def _bar_label(value_keur: float) -> str:
    """Compact delta label in k€; one decimal below 10 k€ for precision."""
    return f"+{value_keur:.1f}" if value_keur < 10.0 else f"+{value_keur:.0f}"


def _draw_panel(ax: plt.Axes, plot_df: pd.DataFrame, value_col: str, ylabel: str) -> None:
    fleets = list(dict.fromkeys(plot_df["fleet"]))
    x = np.arange(len(fleets), dtype=float)
    width = 0.24
    offsets = {
        "DA": -width,
        "DA+ID": 0.0,
        "DA+ID+FCR": width,
    }

    for setup in SETUP_ORDER:
        subset = plot_df[plot_df["setup"] == setup].set_index("fleet").loc[fleets]
        values_keur = subset[value_col].to_numpy() / 1e3
        container = ax.bar(
            x + offsets[setup],
            values_keur,
            width=width,
            label=setup,
            color=FILL_COLORS[setup],
            edgecolor="black",
            linewidth=0.6,
            hatch=HATCHES[setup],
        )
        ax.bar_label(
            container,
            labels=[_bar_label(v) for v in values_keur],
            fontsize=BAR_LABEL_FONT_PT,
            padding=2,
        )

    ax.set_ylabel(ylabel)
    ax.set_xticks(x, fleets)
    ax.margins(y=0.12)  # headroom so bar labels stay inside the axes
    ax.grid(axis="y", color="#bfbfbf", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(plot_df: pd.DataFrame, metric: str, output_base: Path) -> Path:
    fig, (ax_a, ax_b) = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        constrained_layout=True,
    )

    _draw_panel(
        ax_a,
        plot_df,
        f"{metric}_per_year",
        "Annual gross profit vs.\nstatic-price charging (k€/a)",
    )
    _draw_panel(
        ax_b,
        plot_df,
        f"{metric}_per_year_per_bet",
        "Annual gross profit per BET vs.\nstatic-price charging (k€/a)",
    )
    ax_b.set_xlabel("Fleet")

    for ax, panel in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax.text(
            0.0,
            1.02,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
        )

    # shared legend above the top panel, outside the axes to avoid any
    # collision with bars or labels
    ax_a.legend(
        ncols=3,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        borderaxespad=0.0,
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the paper-ready market comparison figure from a batch manifest."
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
        default=Path("paper/figures/market_comparison/market_comparison_profit_delta"),
        help="Output path without extension. PDF and CSV are written.",
    )
    parser.add_argument(
        "--metric",
        default="total_potential_gross_profit_delta_eur",
        help="Manifest column to plot on the y-axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_df = load_plot_data(args.manifest, args.metric)
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(args.output_base.with_suffix(".csv"), index=False)
    print(f"Wrote {args.output_base.with_suffix('.csv')}")

    pdf_path = make_figure(plot_df, args.metric, args.output_base)
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
