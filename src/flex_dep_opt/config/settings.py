import toml
from typing import Literal, Optional
from datetime import datetime
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings import TomlConfigSettingsSource
from pathlib import Path

class SimulationSettings(BaseModel):
    start: datetime
    end: datetime
    timestep_hours: float
    name: str
    solver: Literal["cbc", "gurobi"]

class MarketDetail(BaseModel):
    enabled: bool
    source: str
    fee_eur_per_kwh: float

class Markets(BaseModel):
    dayahead: MarketDetail
    intraday: MarketDetail

class TradingDayahead(BaseModel):
    gate_closure_hour: str
    closes_previous_day: bool

class TradingIntraday(BaseModel):
    offset_minutes_before_delivery: int

class TradingSettings(BaseModel):
    mode: Literal["none", "realistic"]
    dayahead: TradingDayahead
    intraday: TradingIntraday

class ImbalanceSettings(BaseModel):
    enabled: bool
    only_on_infeasible: bool
    source_pos: str
    source_neg: str
    imbalance_volume_penalty_eur_per_kwh: float

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
    eta_grid2depot: float
    eta_depot2grid: float
    grid_connection_limit: int

class OptimizationSettings(BaseModel):
    markets: Markets
    trading: TradingSettings
    imbalance: ImbalanceSettings
    virtual_arbitrage: bool
    mpc: MpcSettings
    flexibility: FlexibilitySettings
    depot: DepotSettings

class Settings(BaseSettings):
    simulation: SimulationSettings
    optimization: OptimizationSettings

    model_config = SettingsConfigDict(
        toml_file="settings_example.toml",
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