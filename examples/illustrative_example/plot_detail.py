"""
Time-series detail figure of the illustrative example (4-day window).

Reads dispatch.csv and cashflow.csv from a run directory (default: the 4-day
quick-start run in results/illustrative_example/detail_4day, i.e. the bundled
settings_example.toml — scenario S3 setup over Fri 2026-02-06 to Tue
2026-02-10) and plots three stacked panels over time:

  1. market positions (DA / ID / FCR droop) and net depot power
  2. depot energy state within the flexibility band
  3. cumulative profit

Requires the optional plotting dependency:  pip install -e .[paper]

Usage:
    python examples/illustrative_example/plot_detail.py [run_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required for the figure scripts: pip install -e .[paper]")

DEFAULT_RUN_DIR = Path("results/illustrative_example/detail_4day")
LOCAL_TZ = "Europe/Berlin"

POSITION_COLS = {  # dispatch column -> (label, color)
    "p_da_kw": ("DA", "#0065BD"),
    "p_id_kw": ("ID", "#E37222"),
    "p_droop_kw": ("FCR activation", "#A2AD00"),
}


def _read_timeseries(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    idx = pd.to_datetime(df["time"], utc=True).dt.tz_convert(LOCAL_TZ)
    return df.drop(columns=["time"]).set_index(idx)


def main() -> int:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN_DIR
    dispatch_csv = run_dir / "dispatch.csv"
    cashflow_csv = run_dir / "cashflow.csv"
    if not dispatch_csv.exists() or not cashflow_csv.exists():
        print(
            f"ERROR: {run_dir} must contain dispatch.csv and cashflow.csv "
            "(run the quick-start example first, see README)",
            file=sys.stderr,
        )
        return 1

    dispatch = _read_timeseries(dispatch_csv)
    cashflow = _read_timeseries(cashflow_csv)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(190 / 25.4, 110 / 25.4))
    ax_p, ax_e, ax_cf = axes

    # Panel 1: market positions + net depot power
    for col, (label, color) in POSITION_COLS.items():
        if col in dispatch.columns and dispatch[col].abs().max() > 0:
            ax_p.step(dispatch.index, dispatch[col], where="post", lw=0.8, label=label, color=color)
    ax_p.step(
        dispatch.index, dispatch["p_net_kw"], where="post", lw=0.8, ls="--", color="black", label="Net"
    )
    ax_p.axhline(0.0, color="black", lw=0.5)
    ax_p.set_ylabel("Power [kW]")
    ax_p.legend(fontsize=6.5, frameon=False, ncol=4, loc="upper left")

    # Panel 2: energy state within the flexibility band
    ax_e.fill_between(
        dispatch.index,
        dispatch["E_lower_kWh"],
        dispatch["E_upper_kWh"],
        color="#E0E0E0",
        label="Flexibility band",
    )
    ax_e.plot(dispatch.index, dispatch["E_kWh"], lw=0.9, color="#0065BD", label="Energy state")
    ax_e.set_ylabel("Energy [kWh]")
    ax_e.legend(fontsize=6.5, frameon=False, loc="upper left")

    # Panel 3: cumulative profit
    ax_cf.plot(cashflow.index, cashflow["Cumulative Profit [€]"], lw=0.9, color="black")
    ax_cf.axhline(0.0, color="black", lw=0.5)
    ax_cf.set_ylabel("Cum. profit [EUR]")
    ax_cf.set_xlabel(f"Time ({LOCAL_TZ})")
    ax_cf.xaxis.set_major_locator(mdates.DayLocator(tz=cashflow.index.tz))
    ax_cf.xaxis.set_major_formatter(mdates.DateFormatter("%a %d.%m.", tz=cashflow.index.tz))

    fig.align_ylabels(axes)
    fig.tight_layout()

    fig_dir = Path("results/illustrative_example/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        out = fig_dir / f"detail_4day.{ext}"
        fig.savefig(out)
        print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
