from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import tomllib

from flex_dep_opt.market.fcr import droop_signal

TZ = "Europe/Berlin"


def resolve_run_dir(arg: str | None) -> pathlib.Path:
    if arg:
        return pathlib.Path(arg).expanduser().resolve()
    pointer = pathlib.Path("results/LATEST.txt")
    if not pointer.exists():
        raise FileNotFoundError("No run_dir given and results/LATEST.txt not found.")
    return pathlib.Path(pointer.read_text().strip()).expanduser().resolve()


def load_settings(run_dir: pathlib.Path) -> dict:
    path = run_dir / "settings.toml"
    if not path.exists():
        raise FileNotFoundError(f"settings.toml not found in {run_dir}")
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def raw_freq_path(settings: dict, override: str | None) -> pathlib.Path:
    if override:
        return pathlib.Path(override).expanduser().resolve()
    src = settings["optimization"]["trading"]["fcr"]["frequency_source"]
    p = pathlib.Path(src)
    # "Frequenz_..._15min.csv" -> "Frequenz_....csv"
    raw = p.with_name(p.name.replace("_15min", ""))
    return raw.expanduser().resolve()


def load_raw_frequency(path: pathlib.Path, start, end) -> pd.Series:
    """Load the raw 'DATE;TIME;FREQUENCY_[HZ]' file, filtered to [start, end]."""
    # Set of "dd.mm.yyyy" date strings spanning the window (inclusive, +1 day).
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
    return s.loc[start:end]


def replay(run_dir: pathlib.Path, raw_freq_override: str | None) -> None:
    settings = load_settings(run_dir)
    sim = settings["simulation"]
    depot = settings["optimization"]["depot"]
    fcr = settings["optimization"]["trading"]["fcr"]
    cyc = settings["optimization"].get("flexibility", {}).get("cycle_regularization", {})

    dt_h = float(sim["timestep_hours"])
    eta_c = float(depot.get("eta_grid2depot", 1.0))
    eta_d = float(depot.get("eta_depot2grid", 1.0))
    nominal = float(fcr.get("frequency_nominal_hz", 50.0))
    deadband = float(fcr.get("deadband_hz", 0.01))
    full = float(fcr.get("full_activation_hz", 0.2))
    reserve_min = float(fcr.get("energy_reserve_minutes", 0.0))
    r_kwh_per_kw = reserve_min / 60.0
    c_deg = float(cyc.get("cost_eur_per_kwh_throughput", 0.0)) if cyc.get("enabled") else 0.0

    # --- dispatch ---
    disp = pd.read_csv(run_dir / "dispatch.csv")
    disp.index = pd.to_datetime(disp["time"], utc=True).dt.tz_convert(TZ)
    disp = disp.drop(columns=["time"])

    fcr_steps = disp[disp["x_fcr_kw"] > 0.0]
    if fcr_steps.empty:
        print("No FCR-active steps in this run — nothing to replay.")
        return

    start = fcr_steps.index.min()
    end = fcr_steps.index.max() + pd.Timedelta(hours=dt_h)

    print(f"Run dir   : {run_dir}")
    print(f"FCR steps : {len(fcr_steps)}  ({start} .. {end})")

    # --- raw frequency -> per-second droop ---
    raw_path = raw_freq_path(settings, raw_freq_override)
    print(f"Raw freq  : {raw_path}")
    freq = load_raw_frequency(raw_path, start, end)
    droop = droop_signal(freq, nominal_hz=nominal, deadband_hz=deadband, full_activation_hz=full)
    print(f"Loaded    : {len(freq):,} 1-second samples\n")

    step = pd.Timedelta(hours=dt_h)
    dt_s = 1.0 / 3600.0  # hours per second

    rows = []
    for ts, row in fcr_steps.iterrows():
        C = float(row["x_fcr_kw"])
        p_net = float(row["p_net_kw"])
        p_droop_mean = float(row["p_droop_kw"])  # = -droop_mean * C
        p_sched = p_net - p_droop_mean  # markets+imbalance (const over slot)
        model_droop = float(row["fcr_droop"])  # slot droop the MODEL committed to

        d = droop.loc[ts : ts + step - pd.Timedelta(seconds=1)].to_numpy(dtype=float)
        n = d.size
        if n == 0:
            continue

        # The 15-min FREQ_DROOP_MEAN the model consumed is not exactly the true per-second slot mean (an alignment artefact of the model's nearest-bin lookup). Report that gap separately, and anchor the per-second signal to the model's committed mean for the SoC diagnostic so the intra-slot *shape* is isolated from that mean error (otherwise net SoC drifts).
        true_mean = float(np.mean(d))
        mean_gap_kwh = abs(model_droop - true_mean) * C * dt_h
        d_anchored = d + (model_droop - true_mean)

        # instantaneous battery power (grid side, import-positive)
        # throughput uses the raw physical signal; SoC uses the anchored signal.
        p_true = p_sched - d * C
        p = p_sched - d_anchored * C

        # SoC trajectory from the model's planned start SoC
        e0 = float(row["E_kWh"])
        de = dt_s * np.where(p > 0.0, eta_c * p, p / eta_d)
        e = e0 + np.cumsum(de)

        # bounds, linearly interpolated across the slot
        frac = np.arange(1, n + 1) / n
        lo = float(row["E_lower_kWh"]) + (float(row["E_lower_next_kWh"]) - float(row["E_lower_kWh"])) * frac
        up = float(row["E_upper_kWh"]) + (float(row["E_upper_next_kWh"]) - float(row["E_upper_kWh"])) * frac

        over = np.maximum(e - up, 0.0)  # breach above upper bound
        under = np.maximum(lo - e, 0.0)  # breach below lower bound
        headroom = np.minimum(up - e, e - lo)  # min distance to a hard bound (<0 = breach)

        # throughput (raw physical signal)
        modeled_thru = (float(row["p_ch_kw"]) + float(row["p_dis_kw"])) * dt_h
        true_thru = float(np.sum(np.abs(p_true)) * dt_s)

        rows.append(
            {
                "time": ts,
                "x_fcr_kw": C,
                "droop_mean": true_mean,
                "model_droop": model_droop,
                "mean_gap_kwh": mean_gap_kwh,
                "droop_abs_mean": float(np.mean(np.abs(d))),
                "droop_abs_max": float(np.max(np.abs(d))),
                "modeled_throughput_kwh": modeled_thru,
                "true_throughput_kwh": true_thru,
                "hidden_throughput_kwh": true_thru - modeled_thru,
                "E_end_true": float(e[-1]),
                "E_end_model": float(row["E_next_kWh"]),
                "peak_over_upper_kwh": float(np.max(over)),
                "peak_under_lower_kwh": float(np.max(under)),
                "min_headroom_kwh": float(np.min(headroom)),
                "reserve_kwh": r_kwh_per_kw * C,
                "samples": n,
            }
        )

    rep = pd.DataFrame(rows).set_index("time")

    out_csv = run_dir / "fcr_replay_steps.csv"
    rep.to_csv(out_csv)

    n_steps = len(rep)
    modeled_total = rep["modeled_throughput_kwh"].sum()
    true_total = rep["true_throughput_kwh"].sum()
    hidden_total = true_total - modeled_total
    hidden_pct = 100.0 * hidden_total / modeled_total if modeled_total > 0 else float("nan")
    extra_cycling_eur = hidden_total * c_deg

    breach_steps = rep[(rep["peak_over_upper_kwh"] > 1e-6) | (rep["peak_under_lower_kwh"] > 1e-6)]
    n_breach = len(breach_steps)
    breach_energy = (breach_steps["peak_over_upper_kwh"] + breach_steps["peak_under_lower_kwh"]).sum()

    # reserve utilization: how far into the reserve buffer the true SoC went.
    # fraction = (reserve - min_headroom) / reserve, clipped to [0, inf).
    with np.errstate(divide="ignore", invalid="ignore"):
        util = np.where(
            rep["reserve_kwh"] > 0,
            (rep["reserve_kwh"] - rep["min_headroom_kwh"]) / rep["reserve_kwh"],
            np.nan,
        )
    worst_util = float(np.nanmax(util)) if np.isfinite(util).any() else float("nan")

    soc_drift = (rep["E_end_true"] - rep["E_end_model"]).abs().max()
    peak_to_mean = (rep["droop_abs_max"] / rep["droop_abs_mean"].replace(0, np.nan)).max()
    mean_gap_max = rep["mean_gap_kwh"].max()
    mean_gap_std = (rep["model_droop"] - rep["droop_mean"]).std()

    # The model's committed slot mean should equal the true per-second slot mean.
    # A nonzero gap means the run's droop was computed with a different definition or a misaligned bin lookup (e.g. a run saved by older code).
    if mean_gap_max > 0.5:  # kWh
        align_lines = [
            "-- WARNING: 15-min mean misaligned ---------------------------",
            "Model's committed slot mean != true per-second slot mean:",
            f"  std {mean_gap_std:.4f} droop  ->  up to {mean_gap_max:.1f} kWh per-slot SoC mis-tracking",
            "  (run's droop definition differs from the raw trace — likely a",
            "   stale run saved by older code, or a bin-lookup misalignment)",
            "",
        ]
    else:
        align_lines = [
            "-- Mean-alignment check --------------------------------------",
            f"Model slot mean == true per-second mean (max gap {mean_gap_max:.2f} kWh) ✓",
            "",
        ]

    lines = [
        "=" * 64,
        "FCR POST-HOC REPLAY",
        "=" * 64,
        f"FCR-active 15-min steps replayed : {n_steps}",
        f"Net SoC consistency (max drift)  : {soc_drift:.4f} kWh   (sanity ~0 after anchoring)",
        "",
        *align_lines,
        "-- Throughput / cycling --------------------------------------",
        f"Modeled (net) throughput         : {modeled_total:,.1f} kWh",
        f"True (1-s) throughput            : {true_total:,.1f} kWh",
        f"Hidden throughput                : {hidden_total:,.1f} kWh  ({hidden_pct:+.1f}%)",
        f"Understated cycling cost @ {c_deg:g} €/kWh : {extra_cycling_eur:,.2f} €",
        "",
        "-- Hidden SoC excursions / imbalance -------------------------",
        f"Steps breaching hard energy band : {n_breach} / {n_steps}",
        f"Total peak breach energy         : {breach_energy:,.2f} kWh",
        f"Worst reserve-buffer utilization : {worst_util:.0%}  (reserve = {reserve_min:g} min/kW)",
        "",
        "-- Spike magnitude -------------------------------------------",
        f"Max(|droop|_peak / |droop|_mean) : {peak_to_mean:.1f}x   "
        "(how much a 1-s spike exceeds the slot mean)",
        "=" * 64,
        f"Per-step detail written to: {out_csv}",
    ]
    summary = "\n".join(lines)
    print(summary)
    (run_dir / "fcr_replay_summary.txt").write_text(summary + "\n")


def main() -> None:
    run_dir = resolve_run_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    raw_override = sys.argv[2] if len(sys.argv) > 2 else None
    replay(run_dir, raw_override)


if __name__ == "__main__":
    main()
