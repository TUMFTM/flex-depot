from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
# Thin gray hatching so the black unidirectional benchmark markers stand out
# against the hatched bars.
HATCH_COLOR = "0.45"
plt.rcParams["hatch.linewidth"] = 0.4


def _apply_hatch_color(patch) -> None:
    """Gray hatch on a black-edged patch. Matplotlib couples the hatch color
    to an explicitly set edgecolor, so override the private attribute
    (matplotlib >= 3.11 exposes this as the hatchcolor parameter)."""
    patch._hatch_color = mcolors.to_rgba(HATCH_COLOR)


def _parse_run_name(row: pd.Series) -> tuple[str, str, str]:
    candidates = [
        Path(str(row["config"])).stem if pd.notna(row.get("config")) else "",
        Path(str(row["run_dir"])).name if pd.notna(row.get("run_dir")) else "",
    ]
    for name in candidates:
        match = re.fullmatch(r"f(\d+)_(da(?:_id)?(?:_fcr)?)(_uni)?", name)
        if match:
            fleet = f"F{int(match.group(1))}"
            setup = SETUP_FROM_SUFFIX[match.group(2)]
            direction = "uni" if match.group(3) else "bidi"
            return fleet, setup, direction
    raise ValueError(f"Could not parse fleet/setup from manifest row: {row.to_dict()}")


def _load_manifest(manifest: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    required = {"config", "run_dir", metric}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    parsed = df.apply(_parse_run_name, axis=1, result_type="expand")
    parsed.columns = ["fleet", "setup", "direction"]
    out = pd.concat([parsed, df[[metric]].astype(float)], axis=1)
    unknown_fleets = sorted(set(out["fleet"]) - set(FLEET_SIZES))
    if unknown_fleets:
        raise ValueError(f"No fleet size defined in common.FLEET_SIZES for: {unknown_fleets}")
    annualization = df.apply(manifest_row_annualization, axis=1, manifest_dir=manifest.parent)
    out["n_vehicles"] = out["fleet"].map(FLEET_SIZES)
    out[f"{metric}_per_year"] = out[metric] * annualization
    out[f"{metric}_per_year_per_bet"] = out[f"{metric}_per_year"] / out["n_vehicles"]
    return out


def _pivot_direction(df: pd.DataFrame, metric: str, direction: str) -> pd.DataFrame:
    """Reindex one direction's rows onto the full fleet x setup grid and fail
    on gaps, so missing runs surface as errors instead of silent holes."""
    subset = df[df["direction"] == direction].drop(columns=["direction"])
    fleet_order = sorted(df["fleet"].unique(), key=lambda value: int(value[1:]))
    index = pd.MultiIndex.from_product([fleet_order, SETUP_ORDER], names=["fleet", "setup"])
    subset = subset.set_index(["fleet", "setup"]).reindex(index).reset_index()
    if subset[metric].isna().any():
        missing_rows = subset.loc[subset[metric].isna(), ["fleet", "setup"]]
        raise ValueError(f"Missing {direction} market comparison rows:\n{missing_rows}")
    return subset


def load_plot_data(manifest: Path, metric: str, manifest_uni: Path | None = None) -> pd.DataFrame:
    """Wide plot table: one row per fleet/setup with bidirectional values and,
    when unidirectional runs exist, matching *_uni columns.

    Unidirectional runs are recognized by their _uni name suffix within a
    combined manifest; a separate uni-only manifest (whose run names carry no
    suffix) can be supplied via manifest_uni instead.
    """
    df = _load_manifest(manifest, metric)
    if manifest_uni is not None:
        uni_df = _load_manifest(manifest_uni, metric)
        uni_df["direction"] = "uni"
        df = pd.concat([df[df["direction"] == "bidi"], uni_df], ignore_index=True)

    duplicate_mask = df.duplicated(["fleet", "setup", "direction"], keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, ["fleet", "setup", "direction"]]
        raise ValueError(f"Duplicate fleet/setup/direction combinations:\n{duplicates}")

    plot_df = _pivot_direction(df, metric, "bidi")
    if (df["direction"] == "uni").any():
        value_cols = [metric, f"{metric}_per_year", f"{metric}_per_year_per_bet"]
        uni = _pivot_direction(df, metric, "uni")
        uni = uni[["fleet", "setup", *value_cols]].rename(
            columns={col: f"{col}_uni" for col in value_cols}
        )
        plot_df = plot_df.merge(uni, on=["fleet", "setup"], validate="one_to_one")

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

    uni_col = f"{value_col}_uni"
    has_uni = uni_col in plot_df.columns

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
        for patch in container.patches:
            _apply_hatch_color(patch)
        ax.bar_label(
            container,
            labels=[_bar_label(v) for v in values_keur],
            fontsize=BAR_LABEL_FONT_PT,
            padding=2,
        )
        if has_uni:
            # Unidirectional level as a benchmark marker on the bidirectional
            # bar; exactly bar-wide so markers of neighboring bars at similar
            # heights do not merge into one line.
            uni_keur = subset[uni_col].to_numpy() / 1e3
            centers = x + offsets[setup]
            ax.hlines(
                uni_keur,
                centers - 0.5 * width,
                centers + 0.5 * width,
                color="black",
                linewidth=1.4,
                zorder=3,
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
        "Annualized gross profit vs.\nstatic-price charging (k€/a)",
    )
    _draw_panel(
        ax_b,
        plot_df,
        f"{metric}_per_year_per_bet",
        "Annualized gross profit per BET vs.\nstatic-price charging (k€/a)",
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
    handles = [
        Patch(
            facecolor=FILL_COLORS[setup],
            hatch=HATCHES[setup],
            edgecolor="black",
            linewidth=0.6,
            label=setup,
        )
        for setup in SETUP_ORDER
    ]
    for handle in handles:
        _apply_hatch_color(handle)
    if f"{metric}_per_year_uni" in plot_df.columns:
        handles.append(
            Line2D([0], [0], color="black", linewidth=1.4, label="Unidirectional")
        )
    ax_a.legend(
        handles=handles,
        ncols=len(handles),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        borderaxespad=0.0,
        columnspacing=1.2,
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(pdf_path)
    fig.savefig(output_base.with_suffix(".svg"))
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
        help=(
            "Path to the batch manifest.csv. Runs whose names end in _uni are "
            "drawn as unidirectional benchmark markers on the bidirectional bars."
        ),
    )
    parser.add_argument(
        "--manifest-uni",
        type=Path,
        default=None,
        help=(
            "Optional separate manifest.csv holding the unidirectional runs "
            "(legacy two-batch layout where uni run names carry no _uni suffix)."
        ),
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
    plot_df = load_plot_data(args.manifest, args.metric, args.manifest_uni)
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(args.output_base.with_suffix(".csv"), index=False)
    print(f"Wrote {args.output_base.with_suffix('.csv')}")

    pdf_path = make_figure(plot_df, args.metric, args.output_base)
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
