from dataclasses import dataclass

@dataclass
class Depot:
    eta_grid2depot: float
    eta_depot2grid: float
    grid_connection_limit: float
