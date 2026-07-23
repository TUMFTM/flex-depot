"""
plot_workflow.py  --  FLEX-DEPOT control-workflow figure (rolling-horizon MPC).

Rebuild of the PowerPoint-generated flex-depot-workflow.svg as a pure
matplotlib script, style analogous to plot_architecture.py.

  Energy Data    -> [Aggregation] -> [Foresight Horizons] -> [ Model ] -> [Real-World]
  Market Data    -------------------->        |                ^  ||           |
  Grid Frequency ------------------->         +-- Prices/Droop-+  vv           v
                                                    [Market Execution]     VB state

  Aggregation (Park et al.) happens once, upstream, on the full fleet data;
  the Foresight Horizons box is the per-step window slicing inside
  FLEX-DEPOT, so all three Model inputs (bounds, prices, droop) pass it.
  Model -> Market Execution: DA/ID positions + FCR capacity bids, with
  gate-closure annotations (DA D-1 12:00, ID t-0 min, FCR D-1 08:00).
  Decision vs. settlement prices: the Foresight Horizons -> Model price
  arrow is annotated "decision: forecast or realized" (optional per-market
  forecast_source; default = realized = perfect foresight), Market
  Execution carries "settlement: realized prices".
  Feedback loops: Virtual Battery State (outer), Committed Positions &
  FCR Capacity (inner); Imbalance Prices in / Imbalance Positions out.
  Prediction horizons are the settings_quickstart.toml values.

Marking: white = physical modeling, light gray = market modeling (same
gray as the flexibility bands in the paper overview figure; colors are
reserved for the per-market plots); diagonally split boxes are both.

Elsevier double column: 190 x 89 mm, Times serif, 7 pt body / 8 pt bold
titles, fonttype 42.

Run:  python images/plot_workflow.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent

mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "font.size":        7,
    "mathtext.fontset": "stix",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "svg.fonttype":     "none",
})

# ── Canvas: original drawing space 643 x 302 px, scaled to 190 mm width ──────
MM = 1.0 / 25.4
PX_W, PX_H = 643.0, 320.0
FIG_W = 190 * MM
FIG_H = FIG_W * PX_H / PX_W          # ≈ 89 mm, same aspect as the original
fig = plt.figure(figsize=(FIG_W, FIG_H))

BLACK = "#1A1A1A"
GRAY  = "#777777"
MARK  = "#E0E0E0"                    # market modeling (flexibility-band gray
                                     # from the paper overview figure);
                                     # physical modeling stays white


def X(px):
    return px / PX_W


def Y(px):
    return 1.0 - px / PX_H


# ── Primitives ────────────────────────────────────────────────────────────────
def rbox(x0, y0, x1, y1, *, fc="white", lw=0.75, z=2):
    """Rectangle in original px coords (y down)."""
    fig.add_artist(Rectangle(
        (X(x0), Y(y1)), X(x1) - X(x0), Y(y0) - Y(y1),
        transform=fig.transFigure,
        fc=fc, ec=BLACK, lw=lw, zorder=z,
    ))


def split_box(x0, y0, x1, y1):
    """Box split along the top-right -> bottom-left diagonal:
    physical (white) upper-left half, market (gray) lower-right half,
    black border on top."""
    ul = [(X(x0), Y(y0)), (X(x1), Y(y0)), (X(x0), Y(y1))]
    lr = [(X(x1), Y(y1)), (X(x0), Y(y1)), (X(x1), Y(y0))]
    fig.add_artist(Polygon(ul, transform=fig.transFigure,
                           fc="white", ec="none", zorder=2))
    fig.add_artist(Polygon(lr, transform=fig.transFigure,
                           fc=MARK, ec="none", zorder=2))
    rbox(x0, y0, x1, y1, fc="none", z=2.5)


def ft(x, y, s, *, fs=7, fw="normal", fst="normal", c=BLACK,
       ha="center", va="center", z=5):
    fig.text(X(x), Y(y), s, fontsize=fs, fontweight=fw, fontstyle=fst,
             color=c, ha=ha, va=va, zorder=z)


def farrow(x0, y0, x1, y1, *, ms=6, sA=1, sB=1, z=4):
    fig.add_artist(FancyArrowPatch(
        (X(x0), Y(y0)), (X(x1), Y(y1)), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=ms,
        color=GRAY, lw=0.8, shrinkA=sA, shrinkB=sB, zorder=z,
    ))


def fline(x0, y0, x1, y1, *, z=3):
    fig.add_artist(mlines.Line2D(
        [X(x0), X(x1)], [Y(y0), Y(y1)],
        transform=fig.transFigure,
        color=GRAY, lw=0.8, zorder=z,
    ))


def dot(x, y, *, r=2.0, z=4.5):
    """Junction dot: a line tap (connected crossing)."""
    fig.add_artist(Ellipse(
        (X(x), Y(y)), 2 * r / PX_W, 2 * r / PX_H,
        transform=fig.transFigure,
        fc=GRAY, ec="none", zorder=z,
    ))


def hop(x, y, *, r=5.0, z=3.5):
    """White gap in a vertical line where a horizontal line crosses
    without connection (drawn above verticals, below horizontals)."""
    fig.add_artist(Ellipse(
        (X(x), Y(y)), 2 * r / PX_W, 2 * r / PX_H,
        transform=fig.transFigure,
        fc="white", ec="none", zorder=z,
    ))


# ── Boxes ─────────────────────────────────────────────────────────────────────
rbox(45.5, 19.5, 112.5, 68.4)                # Aggregation (physical)
split_box(135.5, 19.5, 219.5, 118.3)         # Foresight Horizons (physical + market)
split_box(310.5, 19.5, 384.5, 220.1)         # Model       (physical + market)
rbox(457.5, 20.5, 524.5, 66.4)               # Real-World  (physical)
rbox(458.5, 167.2, 524.5, 269.0, fc=MARK)    # Market Execution (market)

# ── Box titles (above the boxes) ──────────────────────────────────────────────
ft(79.0,  16.5, "Aggregation", fs=8, fw="bold", va="bottom")
ft(177.5, 16.5, "Foresight horizons", fs=8, fw="bold", va="bottom")
ft(347.5, 16.5, "Model",       fs=8, fw="bold", va="bottom")
ft(491.0, 17.5, "Depot",  fs=8, fw="bold", va="bottom")
ft(491.5, 148.0, "Market",     fs=8, fw="bold")
ft(491.5, 161.0, "execution",  fs=8, fw="bold")

# ── Box body text ─────────────────────────────────────────────────────────────
# Aggregation
ft(79.0, 40.0, "Park et al.")
ft(79.0, 53.0, "[26]†")

# Foresight Horizons (values from settings_quickstart.toml; grid frequency
# is only known for the current step -> 0 min horizon)
ft(177.5,  35.0, "Flexibility: 48 h*")
ft(177.5,  52.0, "DA-market: 48 h*")
ft(177.5,  69.0, "ID-market: 48 h*")
ft(177.5,  86.0, "FCR-market: 48 h*")
ft(177.5, 103.0, "Frequency: 0 min*")

# Model
ft(347.5,  88.0, "Mixed Integer")
ft(347.5, 101.0, "Linear")
ft(347.5, 114.0, "Programming")
ft(347.5, 127.0, "(MILP)")
ft(347.5, 152.0, "Virt. arb.: off*")
ft(347.5, 165.0, "Cycle reg.: on*")

# Real-World
ft(491.0, 39.0, "Behavior:")
ft(491.0, 52.0, "ideal", fst="italic")

# Market Execution
ft(491.5, 179.0, "Trade")
ft(491.5, 192.0, "acceptance:")
ft(491.5, 205.0, "ideal", fst="italic")
ft(491.5, 224.0, "Settlement:", fs=5.5, c=GRAY)
ft(491.5, 233.0, "realized prices", fs=5.5, c=GRAY)
ft(491.5, 250.0, "Imbalance")
ft(491.5, 263.0, "accounting")

# ── Feedback-loop lines (verticals + bottom returns, drawn below arrows) ──────
# Outer loop: Real-World "Virtual Battery State" back into Model
fline(629.5, 45.4, 629.5, 293.9)
fline(124.5, 293.9, 629.5, 293.9)
fline(124.5, 146.2, 124.5, 293.9)
# Inner loop: Market Execution "Committed Market Positions" back into Model
fline(599.5, 201.5, 599.5, 281.0)
fline(156.5, 281.0, 599.5, 281.0)
fline(156.5, 198.8, 156.5, 281.0)

# Hops: horizontals cross the loop verticals without connection
hop(124.5, 198.8)
hop(124.5, 257.0)
hop(156.5, 257.0)
hop(599.5, 257.0)
hop(629.5, 201.5)
hop(629.5, 257.0)

# ── Arrows (horizontal flow, drawn above loop verticals) ──────────────────────
# External inputs (left edge; Market Data / Grid Frequency pass beneath the
# short Aggregation box straight into the Foresight Horizons box)
farrow(0.0,  44.0,  45.5,  44.0)    # Energy Data      -> Aggregation
farrow(0.0,  78.0, 135.5,  78.0)    # Market Data      -> Foresight Horizons
farrow(0.0, 112.0, 135.5, 112.0)    # Grid Frequency   -> Foresight Horizons
farrow(0.0, 257.0, 458.5, 257.0)    # Imbalance Prices -> Market Execution

# Forward path
farrow(112.5,  44.5, 135.5,  44.5)  # Aggregation -> Foresight Horizons (VB bounds)
farrow(219.5,  44.5, 310.5,  44.5)  # Horizons    -> Model (power/energy bounds)
farrow(219.5,  78.0, 310.5,  78.0)  # Horizons    -> Model (market prices)
farrow(219.5, 112.0, 310.5, 112.0)  # Horizons    -> Model (FCR droop signal)
farrow(384.5,  44.5, 457.5,  44.5)  # Model       -> Real-World (VB power)
farrow(384.5, 175.0, 458.5, 175.0)  # Model       -> Market Execution (DA/ID)
farrow(384.5, 213.0, 458.5, 213.0)  # Model       -> Market Execution (FCR bids)

# Feedback into Model (tap junctions on the loop verticals)
farrow(0.0, 146.2, 310.5, 146.2)    # Virtual Battery State      -> Model
farrow(0.0, 198.8, 310.5, 198.8)    # Committed Market Positions -> Model

# reBAP prices also enter the Model (PASS-1: FCR activation cashflow pricing;
# PASS-2: imbalance slack pricing). Tap from the Imbalance Prices horizontal
# up into the bottom edge of the Model box.
dot(347.5, 257.0)
farrow(347.5, 257.0, 347.5, 220.1, sA=3)

# Outputs (right edge)
farrow(524.5,  45.4, 643.0,  45.4)  # Real-World -> Virtual Battery State
farrow(524.5, 201.5, 643.0, 201.5)  # Market Execution -> Committed Positions
farrow(524.5, 257.0, 643.0, 257.0)  # Market Execution -> Imbalance Positions

# Junction dots (loop taps)
dot(629.5,  45.4)
dot(124.5, 146.2)
dot(599.5, 201.5)
dot(156.5, 198.8)

# ── Arrow labels ──────────────────────────────────────────────────────────────
ft(22.5,  27.0, "Energy")
ft(22.5,  39.0, "data")
ft(22.5,  61.0, "Market")
ft(22.5,  73.0, "data")
ft(22.5,  95.0, "Grid")
ft(22.5, 107.0, "frequency")
ft(265.0, 38.0, "Power bounds")
ft(265.0, 55.0, "Energy bounds")
ft(265.0, 72.0, "Market prices")
ft(265.0, 86.0, "Decision:", fs=5.5, c=GRAY)
ft(265.0, 94.5, "Forecast or realized", fs=5.5, c=GRAY)
ft(265.0, 106.0, "Droop signal")
ft(262.0, 140.0, "Virtual battery state")
ft(240.0, 191.0, "Committed positions & FCR capacity")
ft(421.0,  38.0, "VB power")
ft(577.0,  38.0, "Virtual battery state")
ft(421.0, 166.0, "DA / ID positions")
ft(421.0, 184.0, "Gate DA: D-1 12:00*", fs=5.5, c=GRAY)
ft(421.0, 193.0, "Gate ID: t-5 min*", fs=5.5, c=GRAY)
ft(421.0, 204.0, "FCR capacity bids")
ft(421.0, 223.0, "Gate FCR: D-1 08:00*", fs=5.5, c=GRAY)
ft(575.0, 180.0, "Committed positions")
ft(575.0, 193.0, "& FCR capacity")
ft(414.0, 249.0, "Imbalance prices")
ft(560.0, 238.0, "Imbalance")
ft(560.0, 251.0, "positions")

# ── Legend ────────────────────────────────────────────────────────────────────
rbox(6.5, 265.0, 25.5, 280.0)
rbox(6.5, 282.0, 25.5, 296.5, fc=MARK)
ft(31.0, 272.5, "Physical modeling", ha="left")
ft(31.0, 289.5, "Market modeling",   ha="left")

# ── Footnote (style as in paper overview figure) ──────────────────────────────
ft(641.0, 308.0, "* values as in the illustrative example; freely configurable", fs=6, c="#4C4C4C",
   ha="right", va="baseline")
ft(641.0, 316.0, "† authors' prior work", fs=6, c="#4C4C4C",
   ha="right", va="baseline")

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ("svg", "pdf"):
    path = HERE / f"flex-depot-workflow.{ext}"
    fig.savefig(path, facecolor="white", bbox_inches=None)
    print(f"saved -> {path}")
