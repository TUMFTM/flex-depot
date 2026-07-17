"""
plot_icon.py  --  FLEX-DEPOT README icon.

Battery-electric truck whose trailer carries the depot's virtual battery:
a battery outline holding the aggregated flexibility band (gray, same tone
as the paper figures) with the dispatch trajectory in TUM blue.

Transparent background so the icon sits directly next to the README title.

Run:  python images/plot_icon.py
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent

DARK = "#1A1A1A"
BLUE = "#0065BD"                     # TUM blue: dispatch trajectory
BAND = "#E0E0E0"                     # flexibility-band gray (paper figures)

MM = 1.0 / 25.4
fig = plt.figure(figsize=(26 * MM, 18 * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-0.1, 12.5)
ax.set_ylim(-0.2, 8.5)
ax.set_aspect("equal")
ax.axis("off")

# ── Truck body ────────────────────────────────────────────────────────────────
# Trailer
ax.add_patch(Rectangle((3.3, 1.7), 8.6, 5.6, fc="none", ec=DARK, lw=2.0))
# Cab
ax.add_patch(Polygon(
    [(0.5, 1.7), (0.5, 5.4), (1.35, 6.5), (2.9, 6.5), (2.9, 1.7)],
    closed=True, fc="none", ec=DARK, lw=2.0, joinstyle="round",
))
# Wheels (drawn after the body: white fill integrates them into the outline)
for cx in (1.7, 4.4, 10.2, 11.6):
    ax.add_patch(Circle((cx, 1.0), 0.85, fc="white", ec=DARK, lw=1.6))

# ── Virtual battery with flexibility band (inside the trailer) ────────────────
ax.add_patch(FancyBboxPatch(
    (4.15, 2.55), 6.0, 3.8,
    boxstyle="round,pad=0.12,rounding_size=0.3",
    fc="white", ec=DARK, lw=1.8,
))
ax.add_patch(Rectangle((10.32, 3.85), 0.55, 1.2, fc=DARK, ec="none"))  # Pol

t = np.linspace(0.0, 1.0, 80)
xs = 4.55 + t * 5.25
up = 5.55 + 0.32 * np.sin(t * 5.0 + 0.6)
lo = 3.35 + 0.32 * np.sin(t * 5.0 - 1.8)
ax.fill_between(xs, lo, up, fc=BAND, ec="none")
ax.plot(xs, up, color=DARK, lw=1.1, solid_capstyle="round")
ax.plot(xs, lo, color=DARK, lw=1.1, solid_capstyle="round")
ax.plot(xs, (up + lo) / 2 + 0.45 * np.sin(t * 8.5), color=BLUE, lw=1.5,
        solid_capstyle="round")

# ── Save (transparent background) ─────────────────────────────────────────────
for ext in ("svg", "pdf"):
    path = HERE / f"flex-depot-icon.{ext}"
    fig.savefig(path, transparent=True)
    print(f"saved -> {path}")
