"""Paper plot for the efficiency sensitivity batch: annual gross profit per BET
vs. charging efficiency, one line per fleet.

Run configs are named f<fleet>_eta<pct> (e.g. f1_eta95); all runs use the full
DA+ID+FCR setup.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    COLUMN_WIDTH_IN,
    FLEET_SIZES,
    MM_TO_INCH,
    TUM_COLORS,
    apply_paper_style,
    manifest_row_annualization,
)

FIG_WIDTH_IN = COLUMN_WIDTH_IN
FIG_HEIGHT_IN = 78.0 * MM_TO_INCH

apply_paper_style()

# Lines encode fleets (all runs share the same market setup), so the
# DA/ID/FCR market-color semantics do not apply here: distinct TUM colors
# plus one marker per fleet as redundant encoding for grayscale print.
FLEET_STYLE = {
    "F1": (TUM_COLORS["TUMBlue"], "o"),
    "F2": (TUM_COLORS["Orange"], "s"),
    "F3": (TUM_COLORS["Green"], "^"),
    "F4": (TUM_COLORS["tum-pink"], "D"),
    "F5": (TUM_COLORS["tum-blue-bright"], "v"),
    "F6": (TUM_COLORS["Gray"], "X"),
}

RUN_NAME_RE = re.compile(r"f(\d+)_eta(\d+(?:\.\d+)?)")


def _parse_run_name(row: pd.Series) -> tuple[str, float]:
    candidates = [
        Path(str(row["config"])).stem if pd.notna(row.get("config")) else "",
        Path(str(row["run_dir"])).name if pd.notna(row.get("run_dir")) else "",
    ]
    for name in candidates:
        match = RUN_NAME_RE.fullmatch(name)
        if match:
            return f"F{int(match.group(1))}", float(match.group(2))
    raise ValueError(f"Could not parse fleet/efficiency from manifest row: {row.to_dict()}")


def load_plot_data(manifest: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    required = {"config", "run_dir", metric}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    if "status" in df.columns:
        failed = df[df["status"] != "ok"]
        if not failed.empty:
            print(
                "WARNING: skipping non-ok manifest rows (their line points will "
                f"be missing):\n{failed[['config', 'status']].to_string(index=False)}"
            )
            df = df[df["status"] == "ok"].reset_index(drop=True)

    parsed = df.apply(_parse_run_name, axis=1, result_type="expand")
    parsed.columns = ["fleet", "efficiency_pct"]
    plot_df = pd.concat([parsed, df[[metric]].astype(float)], axis=1)
    unknown_fleets = sorted(set(plot_df["fleet"]) - set(FLEET_SIZES))
    if unknown_fleets:
        raise ValueError(f"No fleet size defined in common.FLEET_SIZES for: {unknown_fleets}")
    annualization = df.apply(manifest_row_annualization, axis=1, manifest_dir=manifest.parent)
    plot_df["n_vehicles"] = plot_df["fleet"].map(FLEET_SIZES)
    plot_df[f"{metric}_per_year"] = plot_df[metric] * annualization
    plot_df[f"{metric}_per_year_per_bet"] = plot_df[f"{metric}_per_year"] / plot_df["n_vehicles"]

    duplicate_mask = plot_df.duplicated(["fleet", "efficiency_pct"], keep=False)
    if duplicate_mask.any():
        duplicates = plot_df.loc[duplicate_mask, ["fleet", "efficiency_pct"]]
        raise ValueError(f"Duplicate fleet/efficiency combinations in manifest:\n{duplicates}")

    return plot_df.sort_values(["fleet", "efficiency_pct"]).reset_index(drop=True)


def make_figure(plot_df: pd.DataFrame, metric: str, output_base: Path) -> Path:
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), constrained_layout=True)

    value_col = f"{metric}_per_year_per_bet"
    fleets = sorted(plot_df["fleet"].unique(), key=lambda value: int(value[1:]))
    for fleet in fleets:
        subset = plot_df[plot_df["fleet"] == fleet]
        color, marker = FLEET_STYLE.get(fleet, ("#666666", "o"))
        ax.plot(
            subset["efficiency_pct"],
            subset[value_col] / 1e3,
            label=fleet,
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.2,
            markeredgecolor="black",
            markeredgewidth=0.4,
        )

    ax.set_xlabel("(Dis-)Charging efficiency (%)")
    ax.set_ylabel("Annualized gross profit per BET vs.\nstatic-price charging (k€/a)")
    ax.set_xticks(sorted(plot_df["efficiency_pct"].unique()))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#bfbfbf", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        ncols=3,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        borderaxespad=0.0,
        columnspacing=1.2,
        handlelength=1.8,
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the paper-ready efficiency sensitivity figure from a batch manifest."
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
        default=Path("paper/figures/efficiency_sensitivity/efficiency_sensitivity"),
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