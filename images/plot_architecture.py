"""
plot_architecture.py  --  FLEX-DEPOT software-architecture figure.

Layout: horizontal execution flow (Option B), monochrome, with database icons.

  [ data DB ]   [       Configuration Layer       ]   [ results DB ]
                               |
  [ I/O Layer ] <-> [ Workflow Layer ] -> [ Postprocessing Layer ]
                               |
            __________________|__________________
            |                 |                 |
     [ Optimization ]      [ Domain ]        [ Market ]

Elsevier double-column: 190 x 78 mm, Times serif, 7 pt, fonttype 42.

Y layout (78 mm, all outer boxes 18 mm):
  4 | Config 18 | 8 | Mid-row 18 | 8 | Sub-row 18 | 4  (mm)

Run:  python images/plot_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent

mpl.rcParams.update({
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "font.size":    7,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
    "svg.fonttype": "none",
})
_MONO = FontProperties(family="monospace", size=5.5)

MM = 1.0 / 25.4
FIG_W, FIG_H = 190 * MM, 78 * MM
fig = plt.figure(figsize=(FIG_W, FIG_H))

BLACK = "#1A1A1A"
DARK  = "#444444"
GRAY  = "#777777"

# ── Y layout (figure fractions, 0=bottom, 1=top) ─────────────────────────────
# 4 | DB 15 | 2 | Config 18 | 3 | Mid-row 18 | 8 | Sub-row 18 | 4  (mm, total=90)
_H = 78.0
YSB = 4 / _H
YST = (4 + 18) / _H
YMB = (4 + 18 + 8) / _H
YMT = (4 + 18 + 8 + 18) / _H
YCB = (4 + 18 + 8 + 18 + 8) / _H
YCT = (4 + 18 + 8 + 18 + 8 + 18) / _H

# ── X layout: three equal columns, same for mid-row AND sub-row ───────────────
XL  = 0.022; XR = 0.978
_gap = 0.020
_bw  = (XR - XL - 2 * _gap) / 3     # ≈ 0.3053 → 58 mm each

X1L = XL;            X1R = X1L + _bw
X2L = X1R + _gap;    X2R = X2L + _bw
X3L = X2R + _gap;    X3R = XR

XIL,  XIR  = X1L, X1R   # I/O
XWL,  XWR  = X2L, X2R   # Workflow
XPL,  XPR  = X3L, X3R   # Postprocessing
XS1L, XS1R = X1L, X1R   # Optimization
XS2L, XS2R = X2L, X2R   # Domain
XS3L, XS3R = X3L, X3R   # Market

cx1    = (X1L + X1R) / 2
cx2    = (X2L + X2R) / 2
cx3    = (X3L + X3R) / 2
XCFG_L = XIR - 0.070
XCFG_R = XWR
cfg_cx = (XCFG_L + XCFG_R) / 2
cfg_io_x = XCFG_L + 0.025

# ── Primitives ────────────────────────────────────────────────────────────────
def rbox(x0, y0, x1, y1, *, lw=0.75, ec=BLACK, fc="white", ls="-", z=2):
    fig.add_artist(Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        transform=fig.transFigure,
        fc=fc, ec=ec, lw=lw, ls=ls, zorder=z,
    ))


def ft(x, y, s, *, fs=7, fw="normal", c=BLACK, ha="center", va="center",
       fp=None, z=4):
    if fp is not None:
        fig.text(x, y, s, fontproperties=fp, color=c, ha=ha, va=va, zorder=z)
    else:
        fig.text(x, y, s, fontsize=fs, fontweight=fw, color=c,
                 ha=ha, va=va, zorder=z)


def farrow(p0, p1, *, ms=6, sA=1, sB=1, col=GRAY):
    fig.add_artist(FancyArrowPatch(
        p0, p1, transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=ms,
        color=col, lw=0.8, shrinkA=sA, shrinkB=sB,
    ))


def fbiarrow(p0, p1, *, ms=6, sA=1, sB=1, col=GRAY):
    fig.add_artist(FancyArrowPatch(
        p0, p1, transform=fig.transFigure,
        arrowstyle="<->", mutation_scale=ms,
        color=col, lw=0.8, shrinkA=sA, shrinkB=sB,
    ))


def fline(x0, y0, x1, y1, *, z=3, col=GRAY):
    fig.add_artist(mlines.Line2D(
        [x0, x1], [y0, y1],
        transform=fig.transFigure,
        color=col, lw=0.8, zorder=z,
    ))


def _ell_arc(cx, cy, a, b, t1_deg, t2_deg, *, n=40, lw=0.6, col=BLACK, z=5):
    """Parametric arc as Line2D in figure-fraction space (transform-safe)."""
    t  = np.linspace(np.radians(t1_deg), np.radians(t2_deg), n)
    xs = (cx + a * np.cos(t)).tolist()
    ys = (cy + b * np.sin(t)).tolist()
    fig.add_artist(mlines.Line2D(xs, ys,
        transform=fig.transFigure,
        color=col, lw=lw, solid_capstyle="round", zorder=z,
    ))


def db_icon(cx, y_top, label, *, label_y):
    """Cylinder database icon + label (figure fractions).  y_top: top of icon."""
    W  = 0.058
    H  = 0.105
    ER = 0.014

    y_bot      = y_top - H
    cy_top_cap = y_top - ER
    cy_bot_cap = y_bot + ER

    fline(cx - W / 2, cy_top_cap, cx - W / 2, cy_bot_cap, z=4, col=BLACK)
    fline(cx + W / 2, cy_top_cap, cx + W / 2, cy_bot_cap, z=4, col=BLACK)
    fig.add_artist(Ellipse(
        (cx, cy_top_cap), W, 2 * ER,
        transform=fig.transFigure,
        fc="white", ec=BLACK, lw=0.6, zorder=5,
    ))
    for frac in (1 / 3, 2 / 3):
        _ell_arc(cx, y_bot + H * frac, W / 2, ER, 180, 360)
    _ell_arc(cx, cy_bot_cap, W / 2, ER, 180, 360)
    ft(cx, label_y, label, fs=6.5, c=BLACK, z=5)


def module_row(x0, y0, x1, y1, mods, *, gap=0.008, margin=0.010):
    """Row of dashed module-file boxes sized to content and centred."""
    n = len(mods)
    if n == 0:
        return
    inner_w = x1 - x0 - 2 * margin
    gap_tot = (n - 1) * gap
    _cw = 3.30 / 538.6   # fig-fraction char width at 5.5 pt mono
    _pad = 0.008
    pref = [max(len(m) * _cw + 2 * _pad, 0.060) for m in mods]
    total_pref = sum(pref) + gap_tot
    if total_pref <= inner_w:
        widths = pref
        x_start = x0 + margin + (inner_w - total_pref) / 2
    else:
        widths = [(inner_w - gap_tot) / n] * n
        x_start = x0 + margin
    cy = (y0 + y1) / 2
    x = x_start
    for m, w in zip(mods, widths):
        rbox(x, y0, x + w, y1, lw=0.45, ec=DARK, ls=(0, (5, 2.5)), z=3)
        ft(x + w / 2, cy, m, fp=_MONO, c=DARK)
        x += w + gap


# ── Sub-box placement (physical dimensions preserved from 75 mm original) ─────
_bh  = 0.075   # sub-box height  (6.75 mm at 90 mm ≡ same as original)
_off = 0.036   # offset from outer-box bottom to sub-box bottom  (3.24 mm)

CFG_NY  = YCT - 0.050;   CFG_SY0 = YCB + _off;   CFG_SY1 = CFG_SY0 + _bh
MID_NY  = YMT - 0.050;   MID_SY0 = YMB + _off;   MID_SY1 = MID_SY0 + _bh
SUB_NY  = YST - 0.050;   SUB_SY0 = YSB + _off;   SUB_SY1 = SUB_SY0 + _bh

# ── Outer layer boxes ─────────────────────────────────────────────────────────
rbox(XCFG_L, YCB, XCFG_R, YCT) # Configuration
rbox(XIL,  YMB, XIR,  YMT)    # I/O
rbox(XWL,  YMB, XWR,  YMT)    # Workflow
rbox(XPL,  YMB, XPR,  YMT)    # Postprocessing
rbox(XS1L, YSB, XS1R, YST)    # Optimization
rbox(XS2L, YSB, XS2R, YST)    # Domain
rbox(XS3L, YSB, XS3R, YST)    # Market

# ── Layer name labels ─────────────────────────────────────────────────────────
ft(cfg_cx, CFG_NY, "Configuration Layer",  fs=8, fw="bold")
ft(cx1,    MID_NY, "Input / Output Layer", fs=8, fw="bold")
ft(cx2,    MID_NY, "Workflow Layer",       fs=8, fw="bold")
ft(cx3,    MID_NY, "Postprocessing Layer", fs=8, fw="bold")
ft(cx1,    SUB_NY, "Optimization Layer",   fs=8, fw="bold")
ft(cx2,    SUB_NY, "Domain Layer",         fs=8, fw="bold")
ft(cx3,    SUB_NY, "Market Layer",         fs=8, fw="bold")

# ── Dashed module-file sub-boxes ──────────────────────────────────────────────
module_row(XCFG_L, CFG_SY0, XCFG_R, CFG_SY1,
           ["settings.py", "settings.toml"])
module_row(XIL,  MID_SY0, XIR,  MID_SY1,
           ["prices_io.py", "results_io.py", "time_utils.py"])
module_row(XWL,  MID_SY0, XWR,  MID_SY1,
           ["mpc_workflow.py", "postproc_workflow.py"])
module_row(XPL,  MID_SY0, XPR,  MID_SY1,
           ["metrics.py", "plots.py"])
module_row(XS1L, SUB_SY0, XS1R, SUB_SY1,
           ["model.py", "solve.py"])
module_row(XS2L, SUB_SY0, XS2R, SUB_SY1,
           ["depot.py"])
module_row(XS3L, SUB_SY0, XS3R, SUB_SY1,
           ["trading.py", "fcr.py"])

# ── Database icons ────────────────────────────────────────────────────────────
# Place database icons in the left-side space opened by the shorter Configuration layer.
_DB_ICON_H = 0.105
_DB_YTOP   = (YCB + YCT) / 2 + _DB_ICON_H / 2
_db_ybot   = _DB_YTOP - _DB_ICON_H
_db_arrow_y = YCB
_db_label_y = (_db_ybot + _db_arrow_y) / 2
_db_data_x = XIL + 0.050
_db_res_x  = XIL + 0.150

db_icon(_db_data_x, _DB_YTOP, "data", label_y=_db_label_y)
db_icon(_db_res_x,  _DB_YTOP, "results", label_y=_db_label_y)

# ── Arrows ────────────────────────────────────────────────────────────────────
mid_cy = (YMB + YMT) / 2

# Config -> Workflow and Config -> I/O
farrow((cx2, YCB), (cx2, YMT))
farrow((cfg_io_x, YCB), (cfg_io_x, YMT), ms=5, sA=2, sB=2)

# I/O <-> Workflow
fbiarrow((XIR, mid_cy), (XWL, mid_cy), ms=6, sA=1, sB=1)

# Workflow -> Postprocessing
farrow((XWR, mid_cy), (XPL, mid_cy))

# Fork: Workflow -> Optimization, Domain, Market
gap_y = (YMB + YST) / 2
fline(cx2, YMB, cx2, gap_y)
fline(cx1, gap_y, cx3, gap_y)
farrow((cx1, gap_y), (cx1, YST), ms=5, sA=0, sB=1)
farrow((cx2, gap_y), (cx2, YST), ms=5, sA=0, sB=1)
farrow((cx3, gap_y), (cx3, YST), ms=5, sA=0, sB=1)

# DB icon arrows (data -> I/O and I/O -> results)
farrow((_db_data_x, _db_arrow_y), (_db_data_x, YMT), ms=5, sA=2, sB=2)
farrow((_db_res_x, YMT), (_db_res_x, _db_arrow_y), ms=5, sA=2, sB=2)

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ("svg", "pdf"):
    path = HERE / f"flex-depot-architecture.{ext}"
    fig.savefig(path, facecolor="white", bbox_inches=None)
    print(f"saved -> {path}")
