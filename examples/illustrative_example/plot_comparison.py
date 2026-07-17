"""
Bar chart comparing the illustrative-example scenarios S1-S4.

Reads results/illustrative_example/comparison.csv (written by
aggregate_results.py) and plots, per scenario, the stacked cashflow
composition (DA, ID, FCR capacity revenue, fees & imbalance) together with
the uncontrolled-charging reference S0 as a horizontal line. The gap between
a scenario's net cashflow marker and the S0 line is its cost advantage.

Requires the optional plotting dependency:  pip install -e .[paper]

Usage:
    python examples/illustrative_example/plot_comparison.py [comparison.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required for the figure scripts: pip install -e .[paper]")

DEFAULT_CSV = Path("results/illustrative_example/comparison.csv")

# Market colors (TUM palette, consistent with the paper plots)
COLORS = {
    "DA": "#0065BD",
    "ID": "#E37222",
    "FCR": "#A2AD00",
    "Fees & imbalance": "#999999",
}


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        print(f"ERROR: comparison table not found: {csv_path} (run run_all first)", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)

    components = {
        "DA": df["da_cashflow_eur"],
        "ID": df["id_cashflow_eur"],
        "FCR": df["fcr_revenue_eur"],
        "Fees & imbalance": df["fees_eur"] + df["imb_cost_eur"],
    }
    net = -df["total_energy_cost_eur"]  # gross profit (negative = net cost)
    ref = -df["ref_cost_s0_eur"].iloc[0]  # S0 gross profit (identical across scenarios)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(90 / 25.4, 65 / 25.4))  # 90 mm single column

    x = range(len(df))
    pos_base = [0.0] * len(df)
    neg_base = [0.0] * len(df)
    for label, values in components.items():
        bottoms = [pb if v >= 0 else nb for v, pb, nb in zip(values, pos_base, neg_base)]
        ax.bar(x, values, bottom=bottoms, width=0.6, label=label, color=COLORS[label])
        pos_base = [pb + max(v, 0.0) for v, pb in zip(values, pos_base)]
        neg_base = [nb + min(v, 0.0) for v, nb in zip(values, neg_base)]

    ax.scatter(x, net, marker="D", s=18, color="black", zorder=5, label="Net cashflow")
    ax.axhline(ref, color="black", lw=0.8, ls="--", label="S0 (uncontrolled)")
    ax.axhline(0.0, color="black", lw=0.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{sc}\n{mk}" for sc, mk in zip(df["scenario"], df["markets"])], fontsize=7)
    ax.set_ylabel("Cashflow over one month [EUR]")
    ax.margins(y=0.15)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()

    fig_dir = csv_path.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        out = fig_dir / f"comparison.{ext}"
        fig.savefig(out)
        print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
