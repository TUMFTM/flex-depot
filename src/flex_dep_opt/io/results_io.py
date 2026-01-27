from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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
    include_time_column: bool = True,
    time_col: str = "time",
) -> str:
    """
    Save a dispatch-like DataFrame to CSV.

    This helper standardizes the common convention in the project:
    - A dedicated timestamp column named `time` (default), not an unnamed index column.

    Parameters
    ----------
    df:
        DataFrame to write. Index may be a DatetimeIndex (recommended).
    path:
        Output path (string or Path).
    include_time_column:
        If True and df has a DatetimeIndex, the index is written as a column `time_col`.
        If False, the DataFrame is written as-is (index is not written).
    time_col:
        Name of the timestamp column to create when `include_time_column=True`.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    out = _as_path(path)

    out_df = df
    if include_time_column:
        if isinstance(df.index, pd.DatetimeIndex):
            out_df = df.copy()
            out_df = out_df.reset_index().rename(columns={"index": time_col})
        elif time_col not in df.columns:
            raise ValueError(
                f"include_time_column=True, but df has no DatetimeIndex and no '{time_col}' column."
            )

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
