# results_io.py
# Functions for saving optimization results (dispatch, SoC, etc.) to CSV files.

import pandas as pd
from pathlib import Path

def save_dispatch_to_csv(dispatch_df: pd.DataFrame, path: str) -> str:
    """Save a dispatch DataFrame (e.g., p_ch, p_dis, soc) to a CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dispatch_df.to_csv(output_path, index=True)
    return str(output_path.resolve())

def save_summary_to_csv(summary: dict, path: str) -> str:
    """Save a summary dictionary (e.g., KPIs) to a single-row CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_path, index=False)
    return str(output_path.resolve())