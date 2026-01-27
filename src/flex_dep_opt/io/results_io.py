from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Union

import pandas as pd

PathLike = Union[str, Path]


# =============================================================================
# Result I/O utilities
# =============================================================================
def save_dispatch_to_csv(dispatch_df: pd.DataFrame, path: PathLike) -> str:
    """
    Save a dispatch DataFrame to a CSV file.

    Parameters
    ----------
    dispatch_df:
        Time-indexed dispatch DataFrame as returned by `extract_dispatch(...)`.
        The index is expected to represent decision timestamps and is written
        to CSV.
    path:
        Target file path. Parent directories are created automatically.

    Returns
    -------
    str
        Absolute path to the written CSV file.

    Notes
    -----
    - The DataFrame index is preserved (index=True).
    - No validation of column names or units is performed here; this function
      is purely responsible for serialization.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dispatch_df.to_csv(output_path, index=True)

    return str(output_path.resolve())


def save_summary_to_csv(summary: Dict[str, Any], path: PathLike) -> str:
    """
    Save a summary dictionary (e.g., KPIs or aggregated metrics) to a CSV file.

    Parameters
    ----------
    summary:
        Dictionary of scalar values (e.g., floats, ints, strings).
        Each key becomes a column; the CSV contains a single row.
    path:
        Target file path. Parent directories are created automatically.

    Returns
    -------
    str
        Absolute path to the written CSV file.

    Notes
    -----
    - This function assumes a *flat* dictionary structure.
    - Nested dictionaries should be flattened upstream.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([summary]).to_csv(output_path, index=False)

    return str(output_path.resolve())
