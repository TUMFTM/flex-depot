import tomllib
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import toml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SimulationSettings(_StrictModel):
    start: datetime = Field(description="Simulation start time (inclusive), format `YYYY-MM-DD HH:MM`.")
    end: datetime = Field(description="Simulation end time (inclusive), format `YYYY-MM-DD HH:MM`.")
    timestep_hours: float = Field(
        default=0.25, description="Simulation timestep in hours. Only 0.25 is tested."
    )
    name: str = Field(default="example_simulation", description="Identifier used for naming result files.")
    solver: Literal["cbc", "gurobi", "highs"] = Field(
        default="highs", description="Optimization solver backend."
    )
    # Default 8 matches the previous hard-coded solve_model default.
    solver_threads: int | None = Field(
        default=8,
        description="Solver thread count. Set 1 for batch runs that parallelize across scenarios (`--jobs`).",
    )
    solver_mip_gap: float | None = Field(
        default=None, description="Relative MIP gap termination criterion. Unset uses the solver default."
    )
    solver_time_limit_s: int | None = Field(
        default=None, description="Solver time limit per optimization in seconds. Unset means no limit."
    )
    solver_tee: bool = Field(
        default=False,
        description="Stream solver output to stdout. Useful to diagnose hangs or slow solves.",
    )
    warmup_hours: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Warm-up lead-in before the reporting window in hours. The MPC starts `warmup_hours` before "
            "`simulation.start`, so positions whose gate closes before the reporting start (e.g. the first "
            "delivery day's day-ahead auction, closing on D-1) are committed on a full horizon instead of "
            "defaulting to zero. Warm-up steps advance the state (SoC, committed positions) but are excluded "
            "from all results and KPIs. Input series must cover `start - warmup_hours`. Set above the "
            "day-ahead gate lead (more than about 12 h; at least about 40 h when FCR is enabled). "
            "0 disables warm-up."
        ),
    )

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, name: str) -> str:
        if not name.strip():
            raise ValueError("simulation.name must not be empty (e.g. 'illustrative_example').")
        return name


class ForecastErrorSettings(_StrictModel):
    # Per-market decision- vs. settlement-price split: the MPC optimizes on a
    # synthetically perturbed price series (settlement + sigma * z, z ~ AR(1)
    # with unit variance), while cashflows are settled on the real CSV prices.
    # Applies to any market (DA or ID) that carries this block.
    enabled: bool = Field(
        default=False,
        description=(
            "Let the MPC decide on a synthetically perturbed price series (realized price plus an AR(1) "
            "error) while cashflows are settled on the realized prices."
        ),
    )
    sigma_eur_per_mwh: Annotated[
        float, Field(ge=0.0, description="Standard deviation of the forecast error in €/MWh.")
    ] = 0.0
    rho: Annotated[
        float,
        Field(ge=0.0, lt=1.0, description="Lag-1 autocorrelation of the forecast error per 15-minute step."),
    ] = 0.944
    seed: int = Field(default=1, description="Random seed of the forecast-error series.")


class MarketDetail(_StrictModel):
    enabled: bool = Field(default=False, description="Enable participation in this market.")
    source: str = Field(description="CSV file with the realized price time series, used for settlement.")
    forecast_source: str | None = Field(
        default=None,
        description=(
            "Optional CSV with a price forecast used for MPC decisions. If omitted, realized prices are used "
            "(perfect foresight, upper-bound potential). Settlement always uses `source`."
        ),
    )
    fee_eur_per_kwh: float = Field(description="Transaction fee applied to trades on this market in €/kWh.")
    forecast_error: ForecastErrorSettings = Field(
        default_factory=ForecastErrorSettings,
        description="Synthetic forecast error applied to the decision prices of this market.",
    )


class Markets(_StrictModel):
    dayahead: MarketDetail = Field(description="Day-ahead market.")
    intraday: MarketDetail = Field(description="Intraday market.")


class TradingDayahead(_StrictModel):
    gate_closure_hour: str = Field(
        default="12:00", description="Day-ahead gate closure time, format `HH:MM`."
    )
    closes_previous_day: bool = Field(
        default=True, description="Whether the day-ahead gate closes on the day before delivery (D-1)."
    )


class TradingIntraday(_StrictModel):
    offset_minutes_before_delivery: int = Field(
        default=30, description="Intraday gate closure as minutes before delivery."
    )


class FCRForecastErrorSettings(_StrictModel):
    enabled: bool = Field(
        default=False,
        description=(
            "Let the MPC plan FCR bids on a perturbed capacity price (clearing price plus an AR(1) error, "
            "floored at 0) while revenue and bid acceptance use the clearing price."
        ),
    )
    sigma_eur_per_mw: Annotated[
        float, Field(ge=0.0, description="Standard deviation of the forecast error in €/MW per product slot.")
    ] = 0.0
    rho: Annotated[
        float,
        Field(ge=0.0, lt=1.0, description="Lag-1 autocorrelation of the forecast error per product slot."),
    ] = 0.8
    seed: int = Field(default=3, description="Random seed of the forecast-error series.")


class FCRSettings(_StrictModel):
    enabled: bool = Field(default=False, description="Enable FCR reserve market participation.")
    prices_source: str = Field(description="Excel file (.xlsx) with FCR capacity prices, one price per slot.")
    frequency_source: str | None = Field(
        default=None, description="CSV file with the grid frequency time series."
    )
    breakeven_analysis: bool = Field(
        default=True,
        description=(
            "Report the FCR capacity price at which bidding breaks even. Costs extra diagnostic solves."
        ),
    )
    breakeven_include_zero_bid: bool = Field(
        default=False, description="Include the zero-bid alternative in the break-even analysis."
    )
    bid_acceptance: bool = Field(
        default=False,
        description=(
            "Offer each planned bid at its break-even price plus `bid_markup_eur_per_mw` and commit it only "
            "if the clearing price reaches that price (pay-as-cleared, indivisible bid). Runs the break-even "
            "solves even if `breakeven_analysis` is off. Without `forecast_error` or a markup, rejections "
            "are rare, because the MPC already knows the clearing price when it plans the bid."
        ),
    )
    bid_markup_eur_per_mw: Annotated[
        float, Field(ge=0.0, description="Markup on the break-even bid price in €/MW per product slot.")
    ] = 0.0
    forecast_error: FCRForecastErrorSettings = Field(
        default_factory=FCRForecastErrorSettings,
        description="Synthetic forecast error applied to the FCR capacity prices the MPC plans on.",
    )
    gate_closure_hour: str = Field(default="08:00", description="FCR gate closure time, format `HH:MM`.")
    gate_closure_closes_previous_day: bool = Field(
        default=True, description="Whether the FCR gate closes on the day before delivery (D-1)."
    )
    gate_closure_timezone: str = Field(
        default="Europe/Berlin", description="Timezone in which the FCR gate closure is evaluated."
    )
    product_hours: float = Field(default=4.0, description="Duration of one FCR product slot in hours.")
    bid_block_mw: float = Field(default=1.0, description="Minimum bid increment in MW.")
    energy_reserve_minutes: float = Field(
        default=15.0,
        description="Required energy reserve per MW of FCR capacity, in minutes of full activation.",
    )
    reserve_penalty_eur_per_kwh: float = Field(
        default=10.0, description="Penalty on violating the FCR energy reserve (soft constraint) in €/kWh."
    )
    balance_penalty_eur_per_kwh: float = Field(
        default=0.0001,
        description=(
            "Soft mid-band pull on the energy state during FCR slots in €/kWh. "
            "Keeps distance to the reserve edge."
        ),
    )
    frequency_nominal_hz: float = Field(default=50.0, description="Nominal grid frequency in Hz.")
    deadband_hz: float = Field(default=0.010, description="Frequency deadband around nominal in Hz.")
    full_activation_hz: float = Field(
        default=0.200, description="Frequency deviation at which FCR is fully activated, in Hz."
    )


class TradingSettings(_StrictModel):
    mode: Literal["none", "realistic"] = Field(
        default="none",
        description=(
            "Trading mode. `none` keeps the day-ahead and intraday markets always open; "
            "`realistic` enforces gate closures."
        ),
    )
    dayahead: TradingDayahead = Field(description="Day-ahead gate closure rules.")
    intraday: TradingIntraday = Field(description="Intraday gate closure rules.")
    fcr: FCRSettings = Field(description="FCR reserve market settings.")

    @model_validator(mode="after")
    def _fcr_requires_realistic_mode(self) -> "TradingSettings":
        # FCR always commits at its real D-1 gate closure; mode="none" leaves
        # DA/ID always-open. Mixing them silently is a footgun, so reject it.
        if self.fcr.enabled and self.mode == "none":
            raise ValueError(
                "optimization.trading.fcr.enabled=true requires trading.mode='realistic' "
                "(FCR has no 'none'/always-open semantics)."
            )
        return self


class ImbalanceSettings(_StrictModel):
    enabled: bool = Field(default=False, description="Enable imbalance settlement as a fallback.")
    source_pos: str = Field(description="CSV file with positive imbalance prices.")
    source_neg: str = Field(description="CSV file with negative imbalance prices.")
    imbalance_volume_penalty_eur_per_kwh: float = Field(
        default=1000.0, description="Penalty on imbalance energy to discourage its use, in €/kWh."
    )


class MpcSettings(_StrictModel):
    da_horizon_hours: int = Field(description="MPC prediction horizon for day-ahead optimization in hours.")
    id_horizon_hours: int = Field(description="MPC prediction horizon for intraday optimization in hours.")
    fcr_price_horizon_hours: int = Field(
        description="Horizon of known FCR capacity prices ahead of each MPC step in hours."
    )
    fcr_frequency_horizon_minutes: int = Field(
        description=(
            "Horizon of known frequency/activation signal inside the MPC window in minutes. 0 means unknown."
        )
    )
    terminal_condition: bool = Field(description="Enable the soft terminal energy condition.")
    terminal_weight_eur_per_kwh: float = Field(
        description="Weight of the terminal energy deviation penalty in €/kWh."
    )


class CycleRegularization(_StrictModel):
    enabled: bool = Field(description="Enable cycling regularization.")
    cost_eur_per_kwh_throughput: float = Field(
        description="Cost per charged/discharged energy throughput in €/kWh."
    )


class FlexibilitySettings(_StrictModel):
    bounds_file: str = Field(description="CSV file with the aggregated power and energy flexibility bands.")
    cycle_regularization: CycleRegularization = Field(description="Penalty on energy throughput.")


class DepotSettings(_StrictModel):
    eta_grid2depot: Annotated[
        float, Field(gt=0.0, le=1.0, description="Efficiency of the grid-to-depot power flow.")
    ]
    eta_depot2grid: Annotated[
        float, Field(gt=0.0, le=1.0, description="Efficiency of the depot-to-grid power flow.")
    ]
    grid_connection_limit: Annotated[float, Field(gt=0, description="Symmetric grid connection limit in kW.")]


class OptimizationSettings(_StrictModel):
    markets: Markets = Field(description="Energy markets and their price sources.")
    trading: TradingSettings = Field(description="Gate closure rules and the FCR reserve market.")
    imbalance: ImbalanceSettings = Field(description="Imbalance settlement.")
    virtual_arbitrage: bool = Field(description="Allow offsetting buy and sell positions across markets.")
    mpc: MpcSettings = Field(description="Rolling-horizon settings.")
    flexibility: FlexibilitySettings = Field(description="Depot flexibility bands.")
    depot: DepotSettings = Field(description="Depot efficiencies and grid connection.")


class ReferenceDrivingEnergyCostsSettings(_StrictModel):
    enabled: bool = Field(default=False, description="Compute the uncontrolled-charging reference cost KPI.")
    static_price_eur_per_kwh: float | None = Field(
        default=None,
        description="Static electricity price for the reference cost in €/kWh. Required when enabled.",
    )
    energy_column: str = Field(
        default="Ref_driving_energy_kWh",
        description="Column in the flexibility CSV with the grid-side reference driving energy in kWh per step.",
    )
    # The reference energy column is already grid-side (the flexband aggregation
    # stores battery demand / charging efficiency), so this value is not applied
    # to the cost again; it is only written to the KPIs for provenance.
    charging_efficiency: Annotated[
        float,
        Field(
            gt=0.0,
            le=1.0,
            description=(
                "Charging efficiency reported with the reference cost KPIs. It does not enter the cost, "
                "because the reference energy column is already grid-side."
            ),
        ),
    ] = 0.98

    @model_validator(mode="after")
    def _price_required_when_enabled(self) -> "ReferenceDrivingEnergyCostsSettings":
        if self.enabled and self.static_price_eur_per_kwh is None:
            raise ValueError(
                "postprocessing.reference_driving_energy_costs.static_price_eur_per_kwh "
                "must be set when the reference calculation is enabled."
            )
        return self


class PostprocessingSettings(_StrictModel):
    # commit.csv / fcr_commit.csv are the dominant disk cost per run. They must
    # still be written by the simulation since postprocessing reads them back
    # from the run directory.
    save_commits: bool = Field(
        default=True,
        description=(
            "Keep `commit.csv` and `fcr_commit.csv` after postprocessing. "
            "`false` deletes them to save disk space."
        ),
    )
    reference_driving_energy_costs: ReferenceDrivingEnergyCostsSettings = Field(
        default_factory=ReferenceDrivingEnergyCostsSettings,
        description="Reference cost of uncontrolled charging at a static price.",
    )


BASE_DIR = Path(__file__).resolve().parent

_DEFAULT_TOML = BASE_DIR / "settings_quickstart.toml"


class Settings(_StrictModel):
    simulation: SimulationSettings = Field(description="Time window, run name and solver.")
    optimization: OptimizationSettings = Field(description="Markets, trading rules, MPC and depot model.")
    postprocessing: PostprocessingSettings = Field(
        default_factory=PostprocessingSettings, description="What the evaluation step computes and keeps."
    )

    _source_path: Path | None = PrivateAttr(default=None)

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_path: str | Path | None = None) -> "Settings":
        instance = cls.model_validate(data)
        instance._source_path = Path(source_path) if source_path is not None else None
        return instance

    @classmethod
    def from_toml(cls, config_path: str | Path | None = None) -> "Settings":
        """Read settings from a TOML file. Without a path, the bundled quick-start settings are used."""
        resolved = _DEFAULT_TOML if config_path is None else Path(config_path)
        if config_path is not None and not resolved.is_file():
            raise FileNotFoundError(f"Config file not found: {resolved}")

        with open(resolved, "rb") as f:
            data = tomllib.load(f)
        return cls.from_dict(data, source_path=resolved)

    def save_to_toml(self, path: str | Path | None = None):
        save_path = path or self._source_path

        if not save_path:
            raise ValueError("no save path provided or configured")

        data = self.model_dump(mode="json")

        with open(save_path, "w") as f:
            toml.dump(data, f)

    def get_source_path(self) -> Path | None:
        return self._source_path
