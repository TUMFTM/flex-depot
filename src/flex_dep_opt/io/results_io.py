from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


def _as_path(path: str | Path) -> Path:
    """Normalize path input and ensure parent directory exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_dispatch_to_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    time_col: str = "time",
) -> str:
    """
    Save a dispatch-like DataFrame to CSV with the timestamp as a dedicated
    column named `time_col` (not an unnamed index column).

    The DataFrame must have a DatetimeIndex or already carry a `time_col` column.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    out = _as_path(path)

    out_df = df
    if isinstance(df.index, pd.DatetimeIndex):
        out_df = df.reset_index().rename(columns={"index": time_col})
    elif time_col not in df.columns:
        raise ValueError(f"df has no DatetimeIndex and no '{time_col}' column.")

    out_df.to_csv(out, index=False)
    return str(out.resolve())


def save_table_to_csv(df: pd.DataFrame, path: str | Path) -> str:
    """
    Save any table-like DataFrame to CSV exactly as provided (index not written).

    Use this for tables where the index has no semantic meaning (e.g. commit logs).
    """
    out = _as_path(path)
    df.to_csv(out, index=False)
    return str(out.resolve())


def save_summary_to_csv(summary: Mapping[str, Any], path: str | Path) -> str:
    """
    Save a summary/metrics dict (KPIs) to a single-row CSV file.

    Parameters
    ----------
    summary:
        Mapping of KPI name -> value.
    path:
        Output path.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    out = _as_path(path)
    pd.DataFrame([dict(summary)]).to_csv(out, index=False)
    return str(out.resolve())


def _slugify(name: str) -> str:
    """Make a filesystem-friendly run name."""
    s = name.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s or "run"


def make_run_dir(base_dir: str | Path, run_name: str, *, tz: str = "Europe/Berlin") -> Path:
    """
    Create a unique run directory: <base>/<run_name>__<YYYY-MM-DD_HH-MM-SS>.

    Returns the created directory path.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(ZoneInfo(tz)).strftime("%Y-%m-%d_%H-%M-%S")
    folder = base / f"{_slugify(run_name)}__{ts}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def write_latest_run_pointer(run_dir: str | Path, results_root: str | Path = "results") -> str:
    """
    Write a pointer file results/LATEST.txt containing the absolute path to the latest run directory.
    """
    run_dir = Path(run_dir).resolve()

    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)

    latest_path = root / "LATEST.txt"
    latest_path.write_text(str(run_dir), encoding="utf-8")
    return str(latest_path.resolve())


def read_latest_run_pointer(results_root: str | Path = "results") -> Path:
    """
    Read results/LATEST.txt and return the referenced run directory.
    """
    root = Path(results_root)
    latest_path = root / "LATEST.txt"

    if not latest_path.exists():
        raise FileNotFoundError(
            f"No latest run pointer found at {latest_path.resolve()}. "
            "Run `run-sim` first."
        )

    run_dir = Path(latest_path.read_text(encoding="utf-8").strip())

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Latest run directory from LATEST.txt does not exist: {run_dir}"
        )

    return run_dir


def save_run_info_txt(
    *,
    run_dir: str | Path,
    simulation_name: str,
    config_path: str | Path | None,
    solver_name: str,
    start_time: datetime,
    end_time: datetime,
    tz: str = "Europe/Berlin",
) -> str:
    """
    Save run metadata into run_info.txt.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tzinfo = ZoneInfo(tz)

    # Ensure timezone-aware timestamps
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=tzinfo)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=tzinfo)

    duration_s = (end_time - start_time).total_seconds()

    lines = [
        f"simulation.name: {simulation_name}",
        f"solver: {solver_name}",
        f"config_path: {Path(config_path).resolve() if config_path else '(not provided)'}",
        f"started_at: {start_time.astimezone(tzinfo).isoformat()}",
        f"finished_at: {end_time.astimezone(tzinfo).isoformat()}",
        f"duration_seconds: {duration_s:.2f}",
    ]

    out_path = run_dir / "run_info.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out_path.resolve())
