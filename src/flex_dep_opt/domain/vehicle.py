# vehicle.py
# Defines the Vehicle dataclass representing an EV or stationary storage unit.
# Stores basic technical parameters such as capacity, SoC limits, charging/
# discharging power, and efficiencies. Serves as a core domain model for
# optimization and simulation tasks.

from dataclasses import dataclass

@dataclass
class Vehicle:
    capacity_kwh: float
    soc_min: float
    soc_max: float
    soc0: float
    soc_end: float
    p_charge_max_kw: float
    p_discharge_max_kw: float
    eta_charge: float
    eta_discharge: float
