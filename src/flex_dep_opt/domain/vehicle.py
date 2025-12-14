# vehicle.py
# Defines the Vehicle dataclass representing an EV or stationary storage unit.
# Stores basic technical parameters such as capacity, SoC limits, charging/
# discharging power, and efficiencies. Serves as a core domain model for
# optimization and simulation tasks.

from dataclasses import dataclass

@dataclass
class Vehicle:
    eta_charge: float
    eta_discharge: float
