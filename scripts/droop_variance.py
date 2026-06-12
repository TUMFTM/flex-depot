"""
Examine how faithfully the 15-min mean droop represents the real 1-second signal.

Usage:
    python scripts/droop_variance.py <start> <end> [options]

    start, end            ISO timestamps bounding the window (naive = Europe/Berlin).
    --raw PATH            Raw "DATE;TIME;FREQUENCY_[HZ]" file. Default: the
                          Frequenz_*.csv (not *_15min) under data/prices/.
    --nominal HZ          Nominal frequency (default 50.0).
    --deadband HZ         Droop deadband (default 0.010).
    --full HZ             Full-activation deviation (default 0.200).
    --out-dir DIR         Where to write outputs (default ".").

Example:
    python scripts/droop_variance.py 2025-01-15 2025-01-16
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

TZ = "Europe/Berlin"

# todo use settings values
DEFAULT_NOMINAL_HZ = 50.0
DEFAULT_DEADBAND_HZ = 0.010
DEFAULT_FULL_ACTIVATION_HZ = 0.200

MAX_OVERLAY_POINTS = 250_000


def default_raw_path() -> pathlib.Path:
    """Find a raw Frequenz_*.csv (not *_15min) under data/prices/."""
    prices = pathlib.Path("data/prices")
    candidates = sorted(
        p for p in prices.glob("Frequenz_*.csv") if "_15min" not in p.name
    )
    if not candidates:
        raise FileNotFoundError(
            "No raw Frequenz_*.csv found under data/prices/; pass --raw explicitly."
        )
    return candidates[0].resolve()


def load_raw_frequency(path: pathlib.Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Load the raw 'DATE;TIME;FREQUENCY_[HZ]' file, filtered to [start, end]."""
    days = pd.date_range(start.normalize(), (end + pd.Timedelta(days=1)).normalize(), freq="D")
    target_dates = {d.strftime("%d.%m.%Y") for d in days}

    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig", chunksize=2_000_000)
    for chunk in reader:
        chunk.columns = [c.strip().upper() for c in chunk.columns]
        date_col = next(c for c in chunk.columns if c == "DATE")
        time_col = next(c for c in chunk.columns if c == "TIME")
        freq_col = next(c for c in chunk.columns if "FREQ" in c.replace(" ", "_"))
        mask = chunk[date_col].str.strip().isin(target_dates)
        if mask.any():
            parts.append(chunk.loc[mask, [date_col, time_col, freq_col]])

    if not parts:
        raise ValueError(f"No frequency rows found in window {start} .. {end} in {path}")

    df = pd.concat(parts, ignore_index=True)
    date_col, time_col, freq_col = df.columns
    dt = pd.to_datetime(
        df[date_col].str.strip() + " " + df[time_col].str.strip(),
        format="%d.%m.%Y %H:%M:%S",
    )
    freq = df[freq_col].str.strip().str.replace(",", ".", regex=False).astype(float)
    s = pd.Series(freq.values, index=pd.DatetimeIndex(dt))
    s = s.tz_localize(TZ, ambiguous="infer", nonexistent="shift_forward").sort_index()
    s = s[~s.index.duplicated(keep="first")]
    # end is exclusive so a whole-day window yields exactly its full 15-min slots
    # (an inclusive end would leak one sample into a spurious trailing slot).
    return s[(s.index >= start) & (s.index < end)]


def droop_per_second(freq: pd.Series, nominal: float, deadband: float, full: float) -> pd.Series:
    delta_f = freq - nominal
    droop = (-delta_f / full).clip(-1.0, 1.0)
    return droop.where(delta_f.abs() >= deadband, 0.0)


def _count_sign_changes(values: np.ndarray) -> int:
    """Polarity flips within a slot, ignoring deadband (zero) samples."""
    sign = np.sign(values)
    sign = sign[sign != 0]
    if sign.size < 2:
        return 0
    return int(np.sum(sign[1:] != sign[:-1]))


def compute_slot_factors(droop: pd.Series) -> pd.DataFrame:
    """One row per 15-min slot with dispersion / rectification factors."""
    slot = droop.index.floor("15min")
    g = droop.groupby(slot)
    abs_g = droop.abs().groupby(slot)

    mean = g.mean()
    abs_mean = abs_g.mean()
    abs_max = abs_g.max()
    abs_mean_gap = abs_mean - mean.abs()
    peak_to_mean = abs_max / mean.abs().replace(0.0, np.nan)
    sign_changes = g.apply(lambda s: _count_sign_changes(s.to_numpy(dtype=float)))

    out = pd.DataFrame(
        {
            "samples": g.count(),
            "droop_mean": mean,
            "droop_std": g.std(),
            "droop_var": g.var(),
            "droop_abs_mean": abs_mean,
            "abs_mean_gap": abs_mean_gap,   # mean(|d|) - |mean(d)| : the rectification gap
            "droop_abs_max": abs_max,
            "peak_to_mean": peak_to_mean,
            "sign_changes": sign_changes.astype(int),
        }
    )
    out.index.name = "slot_start"
    return out


def summary_text(fac: pd.DataFrame, start, end, raw_path) -> str:
    n = len(fac)
    sum_abs_mean = float(fac["droop_abs_mean"].sum())
    sum_abs_of_mean = float(fac["droop_mean"].abs().sum())
    hidden_pct = (
        100.0 * (sum_abs_mean / sum_abs_of_mean - 1.0) if sum_abs_of_mean > 0 else float("nan")
    )
    ptm = fac["peak_to_mean"].replace([np.inf, -np.inf], np.nan)

    lines = [
        "=" * 64,
        "DROOP VARIANCE  --  1-second vs 15-min mean",
        "=" * 64,
        f"Raw file        : {raw_path}",
        f"Window          : {start} .. {end}",
        f"15-min slots    : {n}",
        f"1-s samples     : {int(fac['samples'].sum()):,}",
        "",
        "-- Dispersion of 1-s droop around the slot mean --------------",
        f"Mean   slot std : {fac['droop_std'].mean():.4f}",
        f"Median slot std : {fac['droop_std'].median():.4f}",
        f"Max    slot std : {fac['droop_std'].max():.4f}",
        "",
        "-- Rectification gap  mean(|d|) - |mean(d)| ------------------",
        f"Mean per slot   : {fac['abs_mean_gap'].mean():.4f}",
        f"Max  per slot   : {fac['abs_mean_gap'].max():.4f}",
        f"Hidden cycling  : {hidden_pct:+.1f}%   "
        "(Sigma mean(|d|) / Sigma |mean(d)| - 1, droop-signal level)",
        "",
        "-- Spike magnitude / oscillation -----------------------------",
        f"Max peak-to-mean: {ptm.max():.1f}x   (max|d| / |mean(d)| over slots)",
        f"Sign changes    : {int(fac['sign_changes'].sum()):,} total, "
        f"{fac['sign_changes'].mean():.1f} avg/slot, {int(fac['sign_changes'].max())} max/slot",
        "=" * 64,
    ]
    return "\n".join(lines)


def _step_xy(values: pd.Series, slot: pd.Timedelta) -> tuple[list, list]:
    """Build flat-per-slot step arrays: each slot value held start->end."""
    xs: list = []
    ys: list = []
    for ts, v in values.items():
        xs += [ts, ts + slot]
        ys += [v, v]
    return xs, ys


def build_overlay(droop: pd.Series, fac: pd.DataFrame) -> "object":
    import plotly.graph_objects as go

    slot = pd.Timedelta(minutes=15)

    # Auto-guard: stride-downsample the raw 1-s line if it is too dense to plot.
    d_plot = droop
    note = ""
    if len(droop) > MAX_OVERLAY_POINTS:
        stride = int(np.ceil(len(droop) / MAX_OVERLAY_POINTS))
        d_plot = droop.iloc[::stride]
        note = f" (1-s line downsampled 1:{stride})"
        print(
            f"  overlay: {len(droop):,} 1-s points > {MAX_OVERLAY_POINTS:,}; "
            f"downsampling raw line 1:{stride}"
        )

    fig = go.Figure()

    # Raw 1-second droop (thin, faint)
    fig.add_trace(
        go.Scatter(
            x=d_plot.index, y=d_plot.values, mode="lines",
            name="droop (1s)",
            line=dict(width=0.7, color="rgba(80,80,80,0.55)"),
            hovertemplate="droop = %{y:.3f}<extra></extra>",
        )
    )

    # +/-1 sigma band around the slot mean (upper first, lower fills to it)
    up_x, up_y = _step_xy(fac["droop_mean"] + fac["droop_std"].fillna(0.0), slot)
    lo_x, lo_y = _step_xy(fac["droop_mean"] - fac["droop_std"].fillna(0.0), slot)
    fig.add_trace(
        go.Scatter(
            x=up_x, y=up_y, mode="lines", name="mean +1sigma",
            line=dict(width=0, color="rgba(0,101,189,0.0)"),
            showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lo_x, y=lo_y, mode="lines", name="+/-1sigma band",
            line=dict(width=0, color="rgba(0,101,189,0.0)"),
            fill="tonexty", fillcolor="rgba(0,101,189,0.15)",
            hoverinfo="skip",
        )
    )

    # Slot-mean step line (the value the model commits to)
    m_x, m_y = _step_xy(fac["droop_mean"], slot)
    fig.add_trace(
        go.Scatter(
            x=m_x, y=m_y, mode="lines", name="slot mean (15min)",
            line=dict(width=2, color="rgb(0,101,189)"),
            hovertemplate="mean = %{y:.3f}<extra></extra>",
        )
    )

    # mean(|d|) step line -- gap to |mean| is the hidden cycling
    am_x, am_y = _step_xy(fac["droop_abs_mean"], slot)
    fig.add_trace(
        go.Scatter(
            x=am_x, y=am_y, mode="lines", name="mean(|d|) (15min)",
            line=dict(width=2, color="rgb(227,114,34)", dash="dot"),
            hovertemplate="mean|d| = %{y:.3f}<extra></extra>",
        )
    )

    fig.add_hline(y=0.0, line=dict(width=1, color="rgba(0,0,0,0.3)", dash="dash"))
    fig.update_layout(
        title=f"1-second droop vs 15-min mean{note}",
        template="plotly_white",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=80, b=60),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="droop [-1, 1]")
    return fig


def parse_window_ts(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(TZ)
    else:
        ts = ts.tz_convert(TZ)
    return ts


def main() -> None:
    ap = argparse.ArgumentParser(description="1-second droop variance vs 15-min mean.")
    ap.add_argument("start", help="window start (ISO; naive = Europe/Berlin)")
    ap.add_argument("end", help="window end (ISO; naive = Europe/Berlin)")
    ap.add_argument("--raw", default=None, help="raw Frequenz_*.csv path")
    ap.add_argument("--nominal", type=float, default=DEFAULT_NOMINAL_HZ)
    ap.add_argument("--deadband", type=float, default=DEFAULT_DEADBAND_HZ)
    ap.add_argument("--full", type=float, default=DEFAULT_FULL_ACTIVATION_HZ)
    ap.add_argument("--out-dir", default=".", help="output directory")
    args = ap.parse_args()

    start = parse_window_ts(args.start)
    end = parse_window_ts(args.end)
    if end <= start:
        print("Error: end must be after start.")
        sys.exit(1)

    raw_path = pathlib.Path(args.raw).expanduser().resolve() if args.raw else default_raw_path()
    if not raw_path.exists():
        print(f"Error: raw file not found: {raw_path}")
        sys.exit(1)

    print(f"Raw file : {raw_path}")
    print(f"Window   : {start} .. {end}")
    freq = load_raw_frequency(raw_path, start, end)
    print(f"Loaded   : {len(freq):,} 1-second samples")

    droop = droop_per_second(freq, args.nominal, args.deadband, args.full).dropna()
    fac = compute_slot_factors(droop)

    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"

    csv_path = out_dir / f"droop_variance_{tag}.csv"
    fac.to_csv(csv_path)

    summary = summary_text(fac, start, end, raw_path)
    print("\n" + summary)
    (out_dir / f"droop_variance_{tag}.txt").write_text(summary + "\n")

    fig = build_overlay(droop, fac)
    html_path = out_dir / f"droop_variance_{tag}.html"
    fig.write_html(str(html_path))

    print(f"\nPer-slot factors : {csv_path}")
    print(f"Overlay          : {html_path}")


if __name__ == "__main__":
    main()
