"""Shared constants and helpers for paper plotting scripts."""

from __future__ import annotations

import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HOURS_PER_YEAR = 8760.0
LOCAL_TIMEZONE = "Europe/Berlin"

# Elsevier artwork sizing: full page width (double column) is 190 mm, single
# column is 90 mm. Fonts: serif to match the running text, 9 pt base / 8 pt
# small (Elsevier minimum is 7 pt), embedded as TrueType (fonttype 42) so the
# PDF passes submission checks.
MM_TO_INCH = 1.0 / 25.4
FULL_WIDTH_IN = 190.0 * MM_TO_INCH
COLUMN_WIDTH_IN = 90.0 * MM_TO_INCH
BASE_FONT_PT = 9.0
SMALL_FONT_PT = 8.0

_STYLE_DIR = Path(__file__).resolve().parents[1] / "style"


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": BASE_FONT_PT,
            "axes.labelsize": BASE_FONT_PT,
            "xtick.labelsize": BASE_FONT_PT,
            "ytick.labelsize": BASE_FONT_PT,
            "legend.fontsize": BASE_FONT_PT,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # keep SVG text as text (not paths) so figures stay editable
            "svg.fonttype": "none",
        }
    )


def load_gpl_palette(path: Path) -> dict[str, str]:
    """Parse a GIMP .gpl palette into {color name: hex string}."""
    colors: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" in line or line == "GIMP Palette":
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        r, g, b = (int(v) for v in parts[:3])
        colors[parts[3].strip()] = f"#{r:02X}{g:02X}{b:02X}"
    return colors


TUM_COLORS = load_gpl_palette(_STYLE_DIR / "TUM.gpl")

# Total fleet size (number of BETs) per fleet, set manually. Used to normalize
# annual profits to per-BET values; note this is the whole fleet, not the
# maximum number of simultaneously connected vehicles.
FLEET_SIZES = {
    "F1": 18,
    "F2": 5,
    "F3": 12,
    "F4": 18,
    "F5": 13,
    "F6": 84,
}


def resolve_run_dir(run_dir: str, manifest_dir: Path) -> Path:
    """Resolve a manifest run_dir, falling back to the manifest's own folder
    so renamed batch directories keep working."""
    path = Path(run_dir)
    if (path / "settings.toml").exists():
        return path
    fallback = manifest_dir / path.name
    if (fallback / "settings.toml").exists():
        return fallback
    raise FileNotFoundError(f"Run directory not found: {run_dir} (also tried {fallback})")


def _annualization_from_simulation(sim: dict, source: object) -> float:
    start = pd.Timestamp(sim["start"]).tz_localize(LOCAL_TIMEZONE)
    end = pd.Timestamp(sim["end"]).tz_localize(LOCAL_TIMEZONE)
    sim_hours = (end - start) / pd.Timedelta(hours=1)
    if sim_hours <= 0:
        raise ValueError(f"Non-positive simulation horizon in {source}")
    return HOURS_PER_YEAR / sim_hours


def annualization_factor(run_dir: Path) -> float:
    """Annualization factor derived from the settings.toml snapshot saved in
    the run directory."""
    with open(run_dir / "settings.toml", "rb") as f:
        settings = tomllib.load(f)
    return _annualization_from_simulation(settings["simulation"], run_dir / "settings.toml")


def annualization_factor_from_config(config_path: Path) -> float:
    """Annualization factor derived from a batch run config, falling back to
    the sibling default.toml for keys the run config does not override. Used
    when the run directories (and their settings.toml snapshots) are gone."""
    with open(config_path, "rb") as f:
        sim = tomllib.load(f).get("simulation", {})
    if "start" not in sim or "end" not in sim:
        default_path = config_path.parent / "default.toml"
        with open(default_path, "rb") as f:
            sim = {**tomllib.load(f).get("simulation", {}), **sim}
    return _annualization_from_simulation(sim, config_path)


def manifest_row_annualization(row: pd.Series, manifest_dir: Path) -> float:
    """Annualization factor for one manifest row.

    Prefers the sim_start/sim_end columns written by run-batch, which make a
    copied manifest.csv self-contained. Falls back to the settings.toml
    snapshot in the run directory, then to the batch config tracked in the
    repo (relevant when only the manifest was copied off the compute
    machine)."""
    start, end = row.get("sim_start"), row.get("sim_end")
    if pd.notna(start) and pd.notna(end):
        return _annualization_from_simulation(
            {"start": start, "end": end}, f"manifest row for {row.get('config')}"
        )
    try:
        return annualization_factor(resolve_run_dir(str(row["run_dir"]), manifest_dir))
    except FileNotFoundError:
        return annualization_factor_from_config(Path(str(row["config"])))