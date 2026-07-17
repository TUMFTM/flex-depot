"""
Two-panel comparison figure of the illustrative-example scenarios S1-S4
(190 mm double column).

Panel (a): gross profit relative to uncontrolled charging at the static
reference price (S0), i.e. the `total_potential_gross_profit_delta_eur` KPI,
with a secondary axis in percent of the S0 cost and "+X € / +Y %" annotations
on the bars. Bars use a neutral gray fill — the TUM market colors are
reserved for per-market quantities (panel (b) and the detail figure); S4 is
hatched to mark imperfect price foresight (redundant encoding for grayscale
print). Panel (b): stacked cashflow composition per scenario (DA, ID, FCR
capacity revenue, fees & imbalance) with the net cashflow marker and the S0
reference line, y-axis symmetric around zero.

Style follows the predecessor-paper plots (Times serif, 9 pt base / 8 pt bar
labels, TUM palette: DA = blue, ID = orange, FCR = green).

Requires the optional plotting dependency:  pip install -e .[paper]

Usage:
    python examples/illustrative_example/plot_comparison.py [comparison.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError:
    sys.exit("matplotlib is required for the figure scripts: pip install -e .[paper]")

DEFAULT_CSV = Path("results/illustrative_example/comparison.csv")

BASE_FONT_PT = 9.0
SMALL_FONT_PT = 8.0
MM_TO_INCH = 1.0 / 25.4

# TUM palette (paper/style/TUM.gpl of the predecessor paper)
MARKET_COLORS = {
    "DA": "#0065BD",  # TUMBlue
    "ID": "#E37222",  # Orange
    "FCR": "#A2AD00",  # Green
    "Fees & imbalance": "#999999",  # Gray
}
# Neutral gray tones per market setup for the scenario-level bars in panel (a);
# the market colors are reserved for per-market quantities (panel (b), detail
# figure). Light to dark with increasing market access (TUM grays).
SETUP_GRAYS = {
    "DA": "#DAD7CB",  # LightGray
    "DA+ID": "#999999",  # Gray
    "DA+ID+FCR": "#6A757E",  # tum-grey-4
}
HATCH_COLOR = "#F0F0F0"  # near-white hatch, visible on the dark setup gray
GRID_KW = {"axis": "y", "color": "#bfbfbf", "linewidth": 0.5, "alpha": 0.8}
FORECAST_HATCH = "///"


def apply_paper_style() -> None:
    """Match paper/scripts/common.py of the predecessor paper."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": BASE_FONT_PT,
            "axes.labelsize": BASE_FONT_PT,
            "xtick.labelsize": BASE_FONT_PT,
            "ytick.labelsize": BASE_FONT_PT,
            "legend.fontsize": SMALL_FONT_PT,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # keep SVG text as text (not paths) so figures stay editable
            "svg.fonttype": "none",
        }
    )


def _panel_advantage(ax, df: pd.DataFrame) -> None:
    """(a) Cost advantage vs. S0 in EUR (left axis) and percent (right axis)."""
    advantage = df["cost_advantage_eur"]

    x = range(len(df))
    bars = ax.bar(
        x,
        advantage,
        width=0.6,
        color=[SETUP_GRAYS[mk] for mk in df["markets"]],
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, fc in zip(bars, df["price_foresight"]):
        if fc == "forecast":
            bar.set_hatch(FORECAST_HATCH)
            # Matplotlib couples the hatch color to the edgecolor; override the
            # private attribute so the black edge keeps a light hatch.
            bar._hatch_color = mcolors.to_rgba(HATCH_COLOR)

    for xi, (adv, pct) in enumerate(zip(advantage, df["cost_advantage_pct"])):
        ax.annotate(
            f"{adv:+.0f} €\n{pct:+.0f} %",
            (xi, adv),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            va="bottom",
            fontsize=SMALL_FONT_PT,
        )

    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"{sc}\n{mk}" for sc, mk in zip(df["scenario"], df["markets"])], fontsize=SMALL_FONT_PT
    )
    ax.set_ylabel("Gross profit vs. static price charging\nfor one-month period (€)")
    ax.margins(y=0.18)
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)

    forecast_patch = Patch(
        facecolor=SETUP_GRAYS["DA+ID+FCR"],
        edgecolor="black",
        hatch=FORECAST_HATCH,
        label="Imperfect foresight",
    )
    forecast_patch._hatch_color = mcolors.to_rgba(HATCH_COLOR)
    ax.legend(handles=[forecast_patch], frameon=False, loc="upper left")


def _panel_composition(ax, df: pd.DataFrame) -> None:
    """(b) Stacked cashflow composition per scenario + net cashflow + S0 line."""
    components = {
        "DA": df["da_cashflow_eur"],
        "ID": df["id_cashflow_eur"],
        "FCR": df["fcr_revenue_eur"],
        "Fees & imbalance": df["fees_eur"] + df["imb_cost_eur"],
    }
    net = -df["total_energy_cost_eur"]  # gross profit (negative = net cost)
    ref = -float(df["ref_cost_s0_eur"].iloc[0])  # S0 gross profit

    x = range(len(df))
    pos_base = [0.0] * len(df)
    neg_base = [0.0] * len(df)
    for label, values in components.items():
        bottoms = [pb if v >= 0 else nb for v, pb, nb in zip(values, pos_base, neg_base)]
        ax.bar(
            x,
            values,
            bottom=bottoms,
            width=0.6,
            label=label,
            color=MARKET_COLORS[label],
            edgecolor="black",
            linewidth=0.4,
        )
        pos_base = [pb + max(v, 0.0) for v, pb in zip(values, pos_base)]
        neg_base = [nb + min(v, 0.0) for v, nb in zip(values, neg_base)]

    ax.scatter(x, net, marker="D", s=18, color="black", zorder=5, label="Net cashflow")
    ax.axhline(ref, color="black", lw=1.4, ls="--", label="S0 (uncontrolled)")
    ax.axhline(0.0, color="black", lw=0.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"{sc}\n{mk}" for sc, mk in zip(df["scenario"], df["markets"])], fontsize=SMALL_FONT_PT
    )
    ax.set_ylabel("Cashflow for one-month period (€)")
    # Symmetric y-axis: zero line in the middle (S0 line and stack tops included)
    extremes = [float(min(neg_base)), float(max(pos_base)), ref, float(net.min()), float(net.max())]
    limit = 1.25 * max(abs(v) for v in extremes)
    ax.set_ylim(-limit, limit)
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)
    # Column-wise legend order: [Net, S0, Fees | DA, ID, FCR] so the market
    # patches share one column.
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = ["Net cashflow", "S0 (uncontrolled)", "Fees & imbalance", "DA", "ID", "FCR"]
    ax.legend(
        [by_label[lb] for lb in order],
        order,
        frameon=False,
        ncol=2,
        loc="upper left",
        columnspacing=1.2,
        handletextpad=0.5,
    )


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        print(f"ERROR: comparison table not found: {csv_path} (run run_all first)", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)

    apply_paper_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(190 * MM_TO_INCH, 75 * MM_TO_INCH))

    _panel_advantage(ax_a, df)
    _panel_composition(ax_b, df)
    for ax, label in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax.set_title(label, fontsize=BASE_FONT_PT, loc="left")

    fig.tight_layout(w_pad=2.0)

    fig_dir = csv_path.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for ext in ("pdf", "svg"):
        out = fig_dir / f"comparison.{ext}"
        try:
            fig.savefig(out)
            print(f"saved -> {out}")
        except PermissionError:
            print(f"ERROR: cannot write {out} (file open in another program?)", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
