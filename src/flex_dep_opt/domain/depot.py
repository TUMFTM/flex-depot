from __future__ import annotations

from dataclasses import dataclass


# =============================================================================
# Depot domain model
# =============================================================================
@dataclass
class Depot:
    """
    Physical abstraction of a grid-connected energy depot.

    Parameters
    ----------
    eta_grid2depot:
        Charging efficiency (grid → depot), unitless.

    eta_depot2grid:
        Discharging efficiency (depot → grid), unitless.

    grid_connection_limit:
        Maximum absolute power exchange with the grid [kW].
        This limit applies symmetrically to charging and discharging.

    """
    eta_grid2depot: float
    eta_depot2grid: float
    grid_connection_limit: float