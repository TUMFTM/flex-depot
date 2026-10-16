from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Depot:
    """Charging and discharging efficiencies and the symmetric grid connection limit in kW."""

    eta_grid2depot: float
    eta_depot2grid: float
    grid_connection_limit: float
