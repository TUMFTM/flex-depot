import toml
from typing import Annotated, Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings import TomlConfigSettingsSource
from pathlib import Path

class SimulationSettings(BaseModel):
    start: datetime
    end: datetime
    timestep_hours: float = 0.25
    name: str = "example_simulation"
    solver: Literal["cbc", "gurobi"] = "cbc"

class MarketDetail(BaseModel):
    enabled: bool = False
    source: str 
    fee_eur_per_kwh: float

class Markets(BaseModel):
    dayahead: MarketDetail
    intraday: MarketDetail

class TradingDayahead(BaseModel):
    gate_closure_hour: str = "12:00"
    closes_previous_day: bool = True

class TradingIntraday(BaseModel):
    offset_minutes_before_delivery: int = 30

class FCRSettings(BaseModel):
    enabled: bool = False
    prices_source: str
    energy_req_hours: float
    frequency_source: Optional[str] = None
    acceptance_rate: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    acceptance_seed: Optional[int] = None
    gate_closure_hour: str = "08:00"
    gate_closure_closes_previous_day: bool = True
    gate_closure_timezone: str = "Europe/Berlin"
    frequency_nominal_hz: float = 50.0
    deadband_hz: float = 0.010
    full_activation_hz: float = 0.200

class TradingSettings(BaseModel):
    mode: Literal["none", "realistic"] = "none"
    dayahead: TradingDayahead
    intraday: TradingIntraday
    fcr: FCRSettings

class ImbalanceSettings(BaseModel):
    enabled: bool = False
    only_on_infeasible: bool
    source_pos: str
    source_neg: str
    imbalance_volume_penalty_eur_per_kwh: float = 1000.0

class MpcSettings(BaseModel):
    da_horizon_hours: int
    id_horizon_hours: int
    terminal_condition: bool
    terminal_weight_eur_per_kwh: float

class CycleRegularization(BaseModel):
    enabled: bool
    cost_eur_per_kwh_throughput: float

class FlexibilitySettings(BaseModel):
    bounds_file: str
    cycle_regularization: CycleRegularization

class DepotSettings(BaseModel):
    eta_grid2depot: Annotated[float, Field(gt=0.0, le=1.0)]
    eta_depot2grid: Annotated[float, Field(gt=0.0, le=1.0)]
    grid_connection_limit: Annotated[float, Field(gt=0)]

class OptimizationSettings(BaseModel):
    markets: Markets
    trading: TradingSettings
    imbalance: ImbalanceSettings
    virtual_arbitrage: bool
    mpc: MpcSettings
    flexibility: FlexibilitySettings
    depot: DepotSettings

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    simulation: SimulationSettings
    optimization: OptimizationSettings

    model_config = SettingsConfigDict(
        toml_file=BASE_DIR / "settings_example.toml",
        env_nested_delimiter="__"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (TomlConfigSettingsSource(settings_cls),)
    
    def save_to_toml(self, path: Optional[str | Path] = None):
        save_path = path or self.model_config.get("toml_file")
        
        if not save_path:
            raise ValueError("no save path provided or configured")

        data = self.model_dump(mode="json")

        with open(save_path, "w") as f:
            toml.dump(data, f)

    def get_source_path(self) -> Optional[str]:
        return self.model_config.get("toml_file")