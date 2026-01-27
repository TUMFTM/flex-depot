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
        Must satisfy: 0 < eta_grid2depot ≤ 1.

    eta_depot2grid:
        Discharging efficiency (depot → grid), unitless.
        Must satisfy: 0 < eta_depot2grid ≤ 1.

    grid_connection_limit:
        Maximum absolute power exchange with the grid [kW].
        This limit applies symmetrically to charging and discharging.

    """
    eta_grid2depot: float
    eta_depot2grid: float
    grid_connection_limit: float

    def __post_init__(self) -> None:
        # ---------------------------------------------------------------------
        # Basic physical consistency checks
        # ---------------------------------------------------------------------
        if not (0.0 < self.eta_grid2depot <= 1.0):
            raise ValueError(
                f"eta_grid2depot must be in (0, 1], got {self.eta_grid2depot}"
            )

        if not (0.0 < self.eta_depot2grid <= 1.0):
            raise ValueError(
                f"eta_depot2grid must be in (0, 1], got {self.eta_depot2grid}"
            )

        if self.grid_connection_limit <= 0.0:
            raise ValueError(
                f"grid_connection_limit must be positive, got {self.grid_connection_limit}"
            )
