"""
Shared style for the illustrative-example paper figures.

Mirrors paper/scripts/common.py of the predecessor paper: Times serif, 9 pt
base / 8 pt small, thin axes, TUM palette (DA = blue, ID = orange,
FCR = green; market colors are reserved for per-market quantities).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

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
# Neutral gray tones per market setup (scenario-level encodings, light to dark
# with increasing market access); market colors stay reserved for markets.
SETUP_GRAYS = {
    "DA": "#DAD7CB",  # LightGray
    "DA+ID": "#999999",  # Gray
    "DA+ID+FCR": "#6A757E",  # tum-grey-4
}
BAND_GRAY = "#DAD7CB"  # flexibility-band fill (TUM LightGray)
GRID_KW = {"axis": "y", "color": "#bfbfbf", "linewidth": 0.5, "alpha": 0.8}


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
