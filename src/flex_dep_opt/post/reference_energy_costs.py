from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_REFERENCE_ENERGY_COLUMN = "Ref_driving_energy_kWh"


def compute_reference_driving_energy_costs(
    flexibility_csv: str | Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    static_price_eur_per_kwh: float,
    charging_efficiency: float = 0.98,
    energy_column: str = DEFAULT_REFERENCE_ENERGY_COLUMN,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Compute static-price reference costs for driving energy.

    The reference energy column holds the *grid-side* driving-energy demand in
    kWh per timestep (it is not multiplied by timestep duration). The charging
    losses are already baked in upstream by the flexband aggregation, which
    stores ``required_battery_energy / charging_efficiency``. The reference cost
    is therefore charged on this grid-side energy directly — no further
    efficiency correction is applied here, as that would double-count the
    charging losses. In the efficiency sweep the embedded efficiency is the
    run's own charging efficiency, so the reference tracks the swept value
    rather than a fixed benchmark.

    ``charging_efficiency`` is retained for provenance / backward compatibility
    only; it no longer enters the grid-side energy or cost calculation.
    """
    price = float(static_price_eur_per_kwh)
    if price < 0.0:
        raise ValueError("static_price_eur_per_kwh must be non-negative.")

    eta = float(charging_efficiency)
    if not (0.0 < eta <= 1.0):
        raise ValueError("charging_efficiency must be in (0, 1].")

    path = Path(flexibility_csv)
    df = pd.read_csv(path)

    if "time" not in df.columns:
        raise ValueError(f"{path} must contain a 'time' column.")
    if energy_column not in df.columns:
        raise ValueError(
            f"{path} must contain reference energy column '{energy_column}' "
            "when reference_driving_energy_costs.enabled is true."
        )

    ts = pd.to_datetime(df["time"], errors="coerce", utc=True)
    if ts.isna().any():
        bad_n = int(ts.isna().sum())
        raise ValueError(f"Found {bad_n} unparsable timestamps in {path}")

    energy = pd.to_numeric(df[energy_column], errors="coerce")
    if energy.isna().any():
        bad_rows = energy[energy.isna()].index[:5].tolist()
        raise ValueError(f"Found non-numeric values in {energy_column} in {path} (rows): {bad_rows}")
    if (energy < 0.0).any():
        bad_rows = energy[energy < 0.0].index[:5].tolist()
        raise ValueError(f"Found negative values in {energy_column} in {path} (rows): {bad_rows}")

    ref = pd.DataFrame({energy_column: energy.astype(float).to_numpy()}, index=pd.DatetimeIndex(ts))
    ref = ref.sort_index()

    if ref.index.has_duplicates:
        dup = ref.index[ref.index.duplicated()].unique()[:5]
        raise ValueError(f"Duplicate timestamps in reference energy data {path} (examples): {list(dup)}")

    ref = ref.loc[start:end].copy()
    if ref.empty:
        raise ValueError(f"No reference energy data found in {path} for simulation window {start} to {end}.")

    # The column is already grid-side (the flexband aggregation stored
    # battery demand / charging_efficiency), so meter it directly. Dividing by
    # eta again here would double-count the charging losses.
    ref["Ref_grid_energy_kWh"] = ref[energy_column]
    ref["Reference Energy Cost [EUR/step]"] = ref["Ref_grid_energy_kWh"] * price
    ref["Cumulative Reference Energy Cost [EUR]"] = ref["Reference Energy Cost [EUR/step]"].cumsum()

    summary = {
        "ref_driving_energy_kwh": float(ref[energy_column].sum()),
        "ref_grid_energy_kwh": float(ref["Ref_grid_energy_kWh"].sum()),
        "ref_charging_efficiency": eta,
        "ref_static_price_eur_per_kwh": price,
        "ref_energy_cost_eur": float(ref["Reference Energy Cost [EUR/step]"].sum()),
    }

    return ref, summary
