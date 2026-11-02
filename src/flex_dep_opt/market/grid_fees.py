"""Grid usage fee accounting (energy and capacity price) with optional MiSpeL-A11 netting.

Sign convention follows the optimization model: ``p_net`` is the physical grid flow,
positive on import (grid-to-depot) and negative on export (depot-to-grid). Grid fees are
levied on the billable import only; the capacity price is charged on the yearly peak import.

Everything here is pure (pandas/numpy); the Pyomo counterpart lives in
:mod:`flex_dep_opt.opt.net_tariff` and reuses :func:`month_key` so the running monthly
balances carried through the MPC line up with the final settlement.
"""

from __future__ import annotations

import pandas as pd


def month_key(ts: pd.Timestamp) -> tuple[int, int]:
    """Calendar (year, month) a timestamp belongs to. Netting never crosses this boundary."""
    return (int(ts.year), int(ts.month))


def split_import_export(p_net_kw: pd.Series, timestep_hours: float) -> tuple[pd.Series, pd.Series]:
    """Split a net grid-power series into non-negative per-step import and export energy (kWh)."""
    dt = float(timestep_hours)
    import_kwh = p_net_kw.clip(lower=0.0) * dt
    export_kwh = (-p_net_kw).clip(lower=0.0) * dt
    return import_kwh, export_kwh


def monthly_net_billable(
    import_kwh: pd.Series,
    export_kwh: pd.Series,
    *,
    mispel_applicable: bool,
) -> pd.DataFrame:
    """Aggregate import/export energy per calendar month and derive the billable import.

    With MiSpeL-A11 netting the billable import is ``max(0, import - export)`` per month, so
    export offsets import only within the same month and never turns into a credit. Without it,
    the full gross import is billable.
    """
    idx = import_kwh.index
    keys = pd.Index([month_key(ts) for ts in idx], name="month", tupleize_cols=False)
    grouped = (
        pd.DataFrame(
            {"grid_import_kwh": import_kwh.to_numpy(), "grid_export_kwh": export_kwh.to_numpy()},
            index=keys,
        )
        .groupby(level=0)
        .sum()
    )

    if mispel_applicable:
        billable = (grouped["grid_import_kwh"] - grouped["grid_export_kwh"]).clip(lower=0.0)
    else:
        billable = grouped["grid_import_kwh"]
    grouped["billable_import_kwh"] = billable
    return grouped


def grid_fee_breakdown(
    billable_import_kwh: float,
    peak_kw: float,
    *,
    energy_price_eur_per_kwh: float,
    capacity_price_eur_per_kw_year: float,
) -> dict[str, float]:
    """Energy, capacity and total grid fee for a given billable import and peak."""
    energy_eur = float(energy_price_eur_per_kwh) * float(billable_import_kwh)
    capacity_eur = float(capacity_price_eur_per_kw_year) * float(peak_kw)
    return {
        "energy_eur": energy_eur,
        "capacity_eur": capacity_eur,
        "total_eur": energy_eur + capacity_eur,
    }


def settle_grid_fees(
    p_net_kw: pd.Series,
    *,
    timestep_hours: float,
    energy_price_eur_per_kwh: float,
    capacity_price_eur_per_kw_year: float,
    mispel_applicable: bool,
    initial_annual_peak_kw: float = 0.0,
    initial_peak_year: int | None = None,
    billed_peak_floor_kw: float = 0.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Settle grid fees on a realized net-power series.

    The energy charge is levied on the billable import (per-month MiSpeL netting when applicable).
    The capacity charge is levied once per calendar year on that year's peak import; the configured
    historical peak seeds ``initial_peak_year`` so a sub-annual window keeps its real yearly peak.

    ``billed_peak_floor_kw`` floors every year's billed peak at the contracted connection capacity.
    When peak shaving is not marketed, the demand charge is owed on that contracted capacity (the
    reference peak) regardless of how low the realized peak happens to be, so an incidentally
    sub-reference peak earns no phantom capacity credit. Pass 0.0 (default) to bill the realized peak.

    Returns a summary dict (gross import/export, billable import, headline peak, energy/capacity/
    total fees) and the per-month netting table for transparent export.
    """
    import_kwh, export_kwh = split_import_export(p_net_kw, timestep_hours)
    monthly = monthly_net_billable(import_kwh, export_kwh, mispel_applicable=mispel_applicable)

    gross_import = float(import_kwh.sum())
    gross_export = float(export_kwh.sum())
    billable = float(monthly["billable_import_kwh"].sum())

    import_kw = p_net_kw.clip(lower=0.0)
    year_peak = import_kw.groupby(import_kw.index.year).max()
    if initial_peak_year is not None and float(initial_annual_peak_kw) > 0.0:
        year_peak.loc[initial_peak_year] = max(
            float(year_peak.get(initial_peak_year, 0.0)), float(initial_annual_peak_kw)
        )
    if float(billed_peak_floor_kw) > 0.0:
        year_peak = year_peak.clip(lower=float(billed_peak_floor_kw))
    peak_kw = (
        float(year_peak.max()) if len(year_peak) else float(max(initial_annual_peak_kw, billed_peak_floor_kw))
    )

    energy_eur = float(energy_price_eur_per_kwh) * billable
    capacity_eur = float(capacity_price_eur_per_kw_year) * float(year_peak.sum())
    summary = {
        "grid_import_kwh": gross_import,
        "grid_export_kwh": gross_export,
        "grid_fee_eligible_import_kwh": billable,
        "grid_peak_kw": peak_kw,
        "grid_fee_energy_eur": energy_eur,
        "grid_fee_capacity_eur": capacity_eur,
        "grid_fee_total_eur": energy_eur + capacity_eur,
    }
    return summary, monthly


def reference_grid_fees(
    reference_energy_kwh: float,
    reference_peak_kw: float,
    *,
    energy_price_eur_per_kwh: float,
    capacity_price_eur_per_kw_year: float,
) -> dict[str, float]:
    """Grid fees of the uncontrolled-charging reference.

    The reference draws ``reference_energy_kwh`` grid-side (losses already baked into the flex
    bands, so no efficiency correction) and is assumed to peak at ``reference_peak_kw``. The
    reference never exports, so MiSpeL netting does not apply.
    """
    fees = grid_fee_breakdown(
        reference_energy_kwh,
        reference_peak_kw,
        energy_price_eur_per_kwh=energy_price_eur_per_kwh,
        capacity_price_eur_per_kw_year=capacity_price_eur_per_kw_year,
    )
    return {
        "ref_grid_energy_kwh": float(reference_energy_kwh),
        "ref_grid_peak_kw": float(reference_peak_kw),
        "ref_grid_fee_energy_eur": fees["energy_eur"],
        "ref_grid_fee_capacity_eur": fees["capacity_eur"],
        "ref_grid_fee_total_eur": fees["total_eur"],
    }
