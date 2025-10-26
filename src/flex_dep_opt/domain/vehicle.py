# vehicle.py
# Defines the Vehicle dataclass representing an EV or stationary storage unit.
# Stores basic technical parameters such as capacity, SoC limits, charging/
# discharging power, and efficiencies. Serves as a core domain model for
# optimization and simulation tasks.

from dataclasses import dataclass

@dataclass
class Vehicle:
    capacity_kwh: float                 # Speichergröße
    soc_min: float = 0.1                # Anteil [0..1]
    soc_max: float = 0.9
    soc0: float = 0.5                   # Start-SOC (Anteil)
    p_charge_max_kw: float = 200.0
    p_discharge_max_kw: float = 200.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95