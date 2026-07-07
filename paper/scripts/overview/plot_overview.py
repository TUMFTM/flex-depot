"""
plot_overview.py -- Methodology overview figure (Fig. 1) for the Applied
Energy results paper.

Pipeline story:
  (a) real fleet data [DT Cargo]  ->  (b) synthetic BETs [DT Cargo]
  ->  (c) flexibility-band aggregation [Park et al.]
  ->  (d) FLEX-DEPOT multi-market MPC [SoftwareX; FCR: this study]
  ->  (e) results of this paper (highlighted)

All plotted data in panels (c), (d), (e) is SYNTHETIC placeholder data.
Replace the functions in the "SYNTHETIC DATA" section with loaders for
real VBbounds CSVs / price series / batch results.

Conventions follow paper/scripts/common.py: 190 mm double column,
serif/Times, 9 pt base / 8 pt small (mini-plots use 6-7 pt), fonttype 42,
TUM market colors (DA = blue, ID = orange, FCR = green).
"""

import sys
from pathlib import Path as FilePath

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon, Patch
from matplotlib.path import Path

# ----------------------------------------------------------------------
# REPO INTEGRATION
# Place this file at:  paper/scripts/overview/plot_overview.py
# common.py is imported from paper/scripts/; the getattr fallbacks below
# let the script also run standalone outside the repo. Replace the
# attribute names ("COL_DA", ...) with the actual constant names used in
# your common.py once integrated.
# ----------------------------------------------------------------------
HERE = FilePath(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))            # -> paper/scripts/
try:
    import common                                # paper/scripts/common.py
except ImportError:
    common = None

FLEET_SIZES = getattr(common, "FLEET_SIZES",
                      {"F1": 18, "F2": 5, "F3": 12, "F4": 18, "F5": 13, "F6": 84})

if (HERE.parent / "common.py").exists():         # running inside the repo
    FIG_DIR = HERE.parent.parent / "figures" / "overview"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
else:                                            # standalone fallback
    FIG_DIR = FilePath(".")

# ----------------------------------------------------------------------
# CONFIG -- adapt to paper/scripts/common.py
# ----------------------------------------------------------------------
MM = 1 / 25.4
FIG_W, FIG_H = 190 * MM, 110 * MM

# TUM palette (paper/style/TUM.gpl) -- taken from common.py if defined there
TUM_BLUE = getattr(common, "COL_DA", "#0065BD")      # market: day-ahead
TUM_ORANGE = getattr(common, "COL_ID", "#E37222")    # market: intraday
TUM_GREEN = getattr(common, "COL_FCR", "#A2AD00")    # market: FCR
TUM_DARKBLUE = "#003359"
GRAY_EDGE = "#8C8C8C"
GRAY_FILL = "#EFEFEF"
HIGHLIGHT = "#B55CA5"     # tum-pink: "this study" accent
HIGHLIGHT_BG = "#F6EAF4"  # tum-pink-4: transparent-looking pink fill

def tint(hexcol, f=0.82):
    """Mix a color with white (f = white fraction)."""
    r, g, b = mpl.colors.to_rgb(hexcol)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)

MARKET = {"DA": TUM_BLUE, "ID": TUM_ORANGE, "FCR": TUM_GREEN}

# Mirrors the paper style from common.py -- remove this block (or parts of
# it) if your common.py already applies the rc style on import.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})

FS_TITLE, FS_TEXT, FS_SMALL = 8, 7, 6

# ----------------------------------------------------------------------
# SYNTHETIC DATA -- replace with real loaders
# ----------------------------------------------------------------------
def smooth(x, k=9):
    xp = np.pad(x, k, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="same")[k:-k]

def synth_flex_bands():
    """Placeholder for a VBbounds CSV (one depot, one day, 15 min)."""
    t = np.arange(0, 24.25, 0.25)
    conn = smooth(np.interp(t, [0, 5, 8.5, 15, 20, 24], [1, 1, 0.25, 0.25, 1, 1]))
    p_max = 18 * 150 / 1000                       # 18 BETs x 150 kW  [MW]
    p_up, p_lo = conn * p_max, -conn * p_max      # bidirectional
    cap = 18 * 572 / 1000                         # [MWh]
    e_up = smooth(np.interp(t, [0, 5, 9, 15, 20, 24], [1, 1, .32, .32, .95, 1])) * cap
    e_lo = smooth(np.interp(t, [0, 6, 15, 21, 24], [.18, .15, .08, .70, .88])) * cap
    return t, p_lo, p_up, e_lo, e_up

def synth_prices():
    """Placeholder for a 15 min DA price series over 1.5 delivery days."""
    rng = np.random.default_rng(4)
    h = np.arange(0, 36.25, 0.25)
    p = 80 + 28 * np.sin((h - 7) / 24 * 2 * np.pi) + 14 * np.sin(h / 3.1) \
        + rng.normal(0, 3.5, h.size)
    return h, p

def synth_results():
    """Placeholder for annualized per-BET profits by market setup:
    (bidirectional total, unidirectional benchmark) in k€ / BET / a."""
    return {"DA": (2.0, 1.2), "+ID": (3.0, 1.6), "+FCR": (4.6, 2.6)}

# ----------------------------------------------------------------------
# FIGURE-LEVEL HELPERS
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(FIG_W, FIG_H))

def fbox(x0, y0, x1, y1, ec, lw=0.7, ls="-", fc="none", z=1):
    fig.add_artist(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
        boxstyle="round,pad=0,rounding_size=0.012", mutation_aspect=FIG_W / FIG_H,
        fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

def ftext(x, y, s, fs=FS_TEXT, w="normal", c="0.15", ha="center", st="normal", z=5):
    fig.text(x, y, s, fontsize=fs, fontweight=w, color=c,
             ha=ha, va="center", style=st, zorder=z)

def badge(x, y, s, fc, ec, tc, fs=FS_SMALL):
    fig.text(x, y, s, fontsize=fs, color=tc, ha="center", va="center", zorder=6,
             bbox=dict(boxstyle="round,pad=0.32", fc=fc, ec=ec, lw=0.6))

def farrow(p0, p1, elbow=None):
    if elbow is None:
        art = FancyArrowPatch(p0, p1, transform=fig.transFigure,
                              arrowstyle="-|>", mutation_scale=8,
                              color="0.35", lw=0.8, shrinkA=0, shrinkB=0)
    else:
        path = Path([p0] + elbow + [p1])
        art = FancyArrowPatch(path=path, transform=fig.transFigure,
                              arrowstyle="-|>", mutation_scale=8,
                              color="0.35", lw=0.8)
    fig.add_artist(art)

def draw_truck(ax, x0=1.0, y0=1.6, battery=False):
    """Simple schematic truck; ax must have equal aspect."""
    ax.add_patch(Rectangle((x0, y0), 5.0, 1.9, fc="0.88", ec="0.25", lw=0.7))
    ax.add_patch(Rectangle((x0 + 5.2, y0), 1.5, 1.5, fc="0.80", ec="0.25", lw=0.7))
    ax.add_patch(Rectangle((x0 + 5.9, y0 + 0.75), 0.8, 0.6, fc="white",
                           ec="0.25", lw=0.5))
    for wx in (x0 + 1.0, x0 + 2.2, x0 + 3.6, x0 + 5.9):
        ax.add_patch(Circle((wx, y0 - 0.1), 0.42, fc="0.35", ec="0.15", lw=0.5))
        ax.add_patch(Circle((wx, y0 - 0.1), 0.18, fc="0.75", ec="none"))
    if battery:
        # small battery with lightning bolt marking the truck as electric;
        # gray like the truck so the schematic adds no extra color;
        # centered in the 5.0 x 1.9 cargo body (incl. the 0.16 terminal nub)
        bx, by, bw, bh = x0 + 1.42, y0 + 0.48, 2.0, 0.95
        ax.add_patch(Rectangle((bx, by), bw, bh, fc="0.45", ec="0.25", lw=0.6))
        ax.add_patch(Rectangle((bx + bw, by + 0.28), 0.16, 0.4,
                               fc="0.45", ec="0.25", lw=0.5))
        bolt = np.array([(0.2, 1.0), (-0.4, -0.05), (0.0, -0.05),
                         (-0.2, -1.0), (0.5, 0.1), (0.1, 0.1)])
        ax.add_patch(Polygon(bolt * (0.3, 0.42) + (bx + 0.5 * bw, by + 0.5 * bh),
                             closed=True, fc="white", ec="none"))

# ----------------------------------------------------------------------
# LAYOUT (figure fractions)
# ----------------------------------------------------------------------
ROW1_Y0, ROW1_Y1 = 0.615, 0.895
A = dict(x0=0.030, x1=0.240)
B = dict(x0=0.330, x1=0.545)
C = dict(x0=0.625, x1=0.968)
D = dict(x0=0.030, x1=0.455, y0=0.055, y1=0.545)
E = dict(x0=0.530, x1=0.968, y0=0.055, y1=0.545)

# outer "this study" frame + label
fbox(0.010, 0.020, 0.988, 0.975, ec=HIGHLIGHT, lw=0.9, ls=(0, (5, 3)))
ftext(0.028, 0.945, "This study: end-to-end assessment of depot-charging "
      "flexibility marketing with real fleet data",
      fs=FS_TITLE, w="bold", c=HIGHLIGHT, ha="left")

# ---------------------------------------------------------------- (a) --
fbox(A["x0"], ROW1_Y0, A["x1"], ROW1_Y1, ec=GRAY_EDGE)
ftext((A["x0"] + A["x1"]) / 2, 0.868, "(a) Fleet operation data", w="bold")
ax_a = fig.add_axes([A["x0"] + 0.012, 0.700, A["x1"] - A["x0"] - 0.024, 0.145])
ax_a.set(xlim=(0, 10), ylim=(0, 5.4)); ax_a.set_aspect("equal"); ax_a.axis("off")
draw_truck(ax_a, x0=1.65, y0=1.3)   # truck spans 6.7 units: center in xlim 0..10
ax_a.text(5, 4.7, f"{len(FLEET_SIZES)} fleets  |  {sum(FLEET_SIZES.values())} trucks | 2.15 mio. km",
          fontsize=5.4, ha="center", va="center", color="0.25")
ftext((A["x0"] + A["x1"]) / 2, 0.678, "Real telemetry data,\n6 home depots",
      fs=FS_SMALL, c="0.30")
badge((A["x0"] + A["x1"]) / 2, 0.636, "Paper et al. [1]†", GRAY_FILL, GRAY_EDGE, "0.25")

# ---------------------------------------------------------------- (b) --
fbox(B["x0"], ROW1_Y0, B["x1"], ROW1_Y1, ec=GRAY_EDGE)
ftext((B["x0"] + B["x1"]) / 2, 0.868, "(b) Synthetic electrification", w="bold")
ax_b = fig.add_axes([B["x0"] + 0.012, 0.700, B["x1"] - B["x0"] - 0.024, 0.145])
ax_b.set(xlim=(0, 10), ylim=(0, 5.4)); ax_b.set_aspect("equal"); ax_b.axis("off")
draw_truck(ax_b, x0=1.65, y0=1.3, battery=True)
ax_b.text(5, 4.7, "Diesel \u2192 BET",
          fontsize=5.4, ha="center", va="center", color="0.25")
ftext((B["x0"] + B["x1"]) / 2, 0.678, "Depot arrival & departure times,\nCharging demand per BET",
      fs=FS_SMALL, c="0.30")
badge((B["x0"] + B["x1"]) / 2, 0.636, "Paper et al. [1]†", GRAY_FILL, GRAY_EDGE, "0.25")

# ---------------------------------------------------------------- (c) --
fbox(C["x0"], ROW1_Y0, C["x1"], ROW1_Y1, ec=GRAY_EDGE)
ftext((C["x0"] + C["x1"]) / 2, 0.868, "(c) Flexibility aggregation", w="bold")
t, p_lo, p_up, e_lo, e_up = synth_flex_bands()
band_c, band_f = "0.25", "0.88"      # truck grays: keep the color count low

ax_c1 = fig.add_axes([C["x0"] + 0.042, 0.792, C["x1"] - C["x0"] - 0.070, 0.070])
ax_c1.fill_between(t, p_lo, p_up, fc=band_f, ec=band_c, lw=0.7)
ax_c1.axhline(0, color="0.4", lw=0.4)
ax_c1.set(xlim=(0, 24), xticks=[], yticks=[])
ax_c1.set_ylabel("Power", fontsize=FS_SMALL, labelpad=1)
ax_c2 = fig.add_axes([C["x0"] + 0.042, 0.708, C["x1"] - C["x0"] - 0.070, 0.070])
ax_c2.fill_between(t, e_lo, e_up, fc=band_f, ec=band_c, lw=0.7)
ax_c2.set(xlim=(0, 24), xticks=[], yticks=[])
ax_c2.set_xlabel("time", fontsize=FS_SMALL, labelpad=2)
ax_c2.set_ylabel("Energy", fontsize=FS_SMALL, labelpad=1)
for ax in (ax_c1, ax_c2):
    ax.tick_params(length=1.5, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
ftext((C["x0"] + C["x1"]) / 2, 0.662, "Virtual battery per depot \u00b7 "
      "P, \u03b7, uni/bidi variations", fs=FS_SMALL, c="0.30")
badge((C["x0"] + C["x1"]) / 2, 0.636, "Park et al. [2]†", GRAY_FILL, GRAY_EDGE, "0.25")

# ---------------------------------------------------------------- (d) --
fbox(D["x0"], D["y0"], D["x1"], D["y1"], ec=GRAY_EDGE)
ftext((D["x0"] + D["x1"]) / 2, 0.518,
      "(d) FLEX-DEPOT: rolling-horizon multi-market MPC", w="bold")
ax_d = fig.add_axes([D["x0"] + 0.030, 0.262, D["x1"] - D["x0"] - 0.048, 0.228])
h, price = synth_prices()

def price_norm(p):
    return 0.10 + 0.80 * (p - price.min()) / np.ptp(price)

ax_d.step(h, price_norm(price), where="post", color=TUM_BLUE, lw=0.9)
# same 15 min resolution as DA, but more volatile, inside the ID window
rng_id = np.random.default_rng(7)
h_id = np.arange(24, 36.25, 0.25)
price_id = np.interp(h_id, h, price) + rng_id.normal(0, 8, h_id.size)
ax_d.step(h_id, np.clip(price_norm(price_id), 0.02, 1.0), where="post",
          color=MARKET["ID"], lw=0.6)
ax_d.add_patch(Rectangle((24, 0.04), 12, 0.96, fc=MARKET["ID"], alpha=0.13, ec="none"))
ax_d.text(30, 0.985, "ID trading", fontsize=5.2, color=MARKET["ID"],
          ha="center", va="top")
for gx, gc, lbl, halign, dx in ((8, MARKET["FCR"], "FCR gc", "right", -0.4),
                                 (12, MARKET["DA"], "DA gc", "left", 0.4)):
    ax_d.vlines(gx, 0.04, 1.0, color=gc, lw=0.8, ls=(0, (3, 2)))
    ax_d.text(gx + dx, 0.985, lbl, fontsize=5.2, color=gc, ha=halign, va="top")
ax_d.set(xlim=(0, 36), ylim=(0, 1.02), yticks=[],
         xticks=[0, 12, 24, 36])
ax_d.tick_params(length=1.5, pad=1, labelbottom=False)
for s in ("top", "right", "left"):
    ax_d.spines[s].set_visible(False)

# FCR capacity-price strip: same time axis, but its own thin mini axis so
# the EUR/MW capacity prices stay separate from the EUR/MWh energy prices
# above. Shows the 4 h product blocks and carries the shared x tick labels.
ax_f = fig.add_axes([D["x0"] + 0.030, 0.205, D["x1"] - D["x0"] - 0.048, 0.045])
fcr_edges = np.arange(0, 37, 4)
fcr_vals = [0.35, 0.28, 0.42, 0.55, 0.45, 0.62, 0.38, 0.52, 0.70]
ax_f.stairs(fcr_vals, fcr_edges, fill=True, fc=tint(MARKET["FCR"], 0.6), ec="none")
ax_f.stairs(fcr_vals, fcr_edges, fill=False, ec=MARKET["FCR"], lw=0.8)
ax_f.vlines(8, 0, 1, color=MARKET["FCR"], lw=0.8, ls=(0, (3, 2)))
ax_f.text(0.6, 0.95, "FCR capacity (4 h)", fontsize=5.2,
          color=MARKET["FCR"], ha="left", va="top")
ax_f.set(xlim=(0, 36), ylim=(0, 1.0), yticks=[], xticks=[0, 12, 24, 36])
ax_f.set_xticklabels(["D\u22121, 0 h", "12 h", "D, 0 h", "delivery"])
ax_f.tick_params(length=1.5, pad=1)
for s in ("top", "right", "left"):
    ax_f.spines[s].set_visible(False)

# Market badges grouped by attribution: {DA, ID} from the framework paper,
# "+ FCR" added on top by this study. Proximity and the leading "+" carry
# the grouping (two clusters, not an even spread tied to the time axis);
# both rows sit low in the panel, clear of the price plot.
x_da, x_id, x_fcr = D["x0"] + 0.078, D["x0"] + 0.198, D["x1"] - 0.082
badge(x_da, 0.130, "Day-ahead", tint(MARKET["DA"]), MARKET["DA"], "0.1")
badge(x_id, 0.130, "Intraday", tint(MARKET["ID"]), MARKET["ID"], "0.1")
badge(x_fcr, 0.130, "+ FCR", tint(MARKET["FCR"]), MARKET["FCR"], "0.1")
badge((x_da + x_id) / 2, 0.082, "Brödel et al. [3]†", GRAY_FILL, GRAY_EDGE, "0.25")
badge(x_fcr, 0.082, "This study", HIGHLIGHT_BG, HIGHLIGHT, HIGHLIGHT)

# ---------------------------------------------------------------- (e) --
fbox(E["x0"], E["y0"], E["x1"], E["y1"], ec=HIGHLIGHT, lw=1.1, fc=HIGHLIGHT_BG, z=0)
ftext((E["x0"] + E["x1"]) / 2, 0.518, "(e) Results of this study",
      w="bold", c=HIGHLIGHT)
ax_e = fig.add_axes([E["x0"] + 0.055, 0.135, 0.155, 0.345])
ax_e.set_facecolor("none")
res = synth_results()
# One fill color + hatch per setup, exactly as in the market comparison
# results figure (fill = color of the added market; gray hatch as redundant
# encoding for grayscale print and CVD readers).
SETUP_STYLE = {"DA": (MARKET["DA"], ""), "+ID": (MARKET["ID"], "///"),
               "+FCR": (MARKET["FCR"], "xxx")}
mpl.rcParams["hatch.linewidth"] = 0.4
for i, (scen, (total, uni)) in enumerate(res.items()):
    fill, hatch = SETUP_STYLE[scen]
    bars = ax_e.bar(i, total, 0.62, fc=fill, ec="black", lw=0.5, hatch=hatch)
    for p in bars.patches:
        p._hatch_color = mpl.colors.to_rgba("0.45")
    # unidirectional benchmark as a bar-wide black marker, as in the
    # market comparison results figure
    ax_e.hlines(uni, i - 0.31, i + 0.31, color="black", lw=1.2, zorder=3)
# neutral gray box stands in for the (differently colored) bidi bars
ax_e.legend(handles=[Patch(fc="0.85", ec="black", lw=0.5, label="Bidirectional"),
                     Line2D([0], [0], color="black", lw=1.2, label="Unidirectional")],
            frameon=False, fontsize=5.2, loc="upper left",
            handlelength=1.2, labelspacing=0.3, borderaxespad=0.1)
ax_e.set(xticks=range(len(res)), xticklabels=list(res), yticks=[])
ax_e.set_ylabel("Annualized profit", fontsize=FS_SMALL, labelpad=1)
ax_e.tick_params(length=1.5, pad=1)
for s in ("top", "right"):
    ax_e.spines[s].set_visible(False)

dx = E["x0"] + 0.250
ftext(dx, 0.435, "Scenario dimensions", fs=FS_SMALL, c="0.35", ha="left", st="italic")
for j, line in enumerate(["Markets: DA / +ID / +FCR",
                          "Uni- vs. bidirectional (V2G)",
                          "(Dis-)Charging power & efficiency,",
                          "Ageing, ID horizon"]):
    ftext(dx, 0.385 - 0.052 * j, line, fs=FS_TEXT, c="0.15", ha="left")
ftext(dx, 0.150, "6 fleets, 150 BETs\nJan\u2013Aug 2026, 15 min\nannualized per BET",
      fs=FS_SMALL, c="0.35", ha="left")

# ---------------------------------------------------------- arrows -----
ymid = (ROW1_Y0 + ROW1_Y1) / 2
farrow((A["x1"] + 0.006, ymid), (B["x0"] - 0.006, ymid))
farrow((B["x1"] + 0.006, ymid), (C["x0"] - 0.006, ymid))
xc = (C["x0"] + C["x1"]) / 2
xd = (D["x0"] + D["x1"]) / 2
farrow((xc, ROW1_Y0 - 0.006), (xd, D["y1"] + 0.006),
       elbow=[(xc, 0.582), (xd, 0.582)])
farrow((D["x1"] + 0.006, 0.30), (E["x0"] - 0.006, 0.30))
#ftext((D["x1"] + E["x0"]) / 2, 0.335, "simulate\nall scenarios", fs=5.2, c="0.35")

# dagger footnote for the reference badges
ftext(0.966, 0.036, "† authors' prior work", fs=FS_SMALL, c="0.30", ha="right")

# ---------------------------------------------------------- export -----
for ext in ("png", "svg", "pdf"):
    fig.savefig(FIG_DIR / f"fig1_overview.{ext}", dpi=300 if ext == "png" else None,
                facecolor="white", bbox_inches=None)
print(f"figures written to {FIG_DIR.resolve()}")