# FLEX-DEPOT 

FLEX-DEPOT is a Python project for optimizing flexibility at a logistics depot incorporating electric trucks. It combines market price series, flexibility power and energy bands, and market trading rules into a single rolling-horizon optimization (MPC).

The core of the project is a MILP/LP model, depending on settings, built with [Pyomo](https://pyomo.readthedocs.io/en/stable/). It decides market positions and physical power flow between depot and public grid while respecting aggregated asset power and energy flexibility bounds and gate-closure rules.

Key capabilities:
- Unified optimization for day-ahead, intraday, and FCR markets.
- FCR gate-closure handling, capacity commitment, optional activation from frequency data, acceptance sampling, and breakeven analysis.
- Optional imbalance settlement with positive and negative imbalance prices.
- Rolling-horizon MPC that commits market positions when gate closure is reached.
- CSV-based I/O for prices, flexibility bounds, and results.
- Plotly-based visualization of dispatch, prices, and cashflows.

## Created by
Marcel Brödel, M.Sc. <br>
Institute of Automotive Technology <br>
Department of Mobility Systems Engineering <br>
TUM School of Engineering and Design <br>
Technical University of Munich <br>
[marcel.broedel@tum.de](mailto:marcel.broedel@tum.de)

## Licensing
FLEX-DEPOT is licensed under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) open source license. <br>
The full license text can be found in the LICENSE file in the root directory of the repository.

## Related Publications
Brödel, M., Park, W.-H., Rosner, P., Lienkamp, M.: "FLEX-DEPOT: An open-source framework for multi-market flexibility commercialization of a logistics depot incorporating electric trucks." <br>
Manuscript under review at SoftwareX, Elsevier (2026)

## Citation
If you use FLEX-DEPOT in academic work, please cite both:
1. The FLEX-DEPOT software repository
2. The associated publication

Software:
Brödel, M. (2026). FLEX-DEPOT (Version 1.0.0). GitHub repository.  
https://github.com/TUMFTM/flex-depot

Publication:
Brödel, M., Park, W.-H., Rosner, P., Lienkamp, M.  
FLEX-DEPOT: An open-source framework for multi-market flexibility commercialization of a logistics depot incorporating electric trucks.  
SoftwareX, 2026 (under review).

Citation metadata is provided in `CITATION.cff`.

## Repository Architecture
![flex-depot-architecture.svg](images/flex-depot-architecture.svg)

## Module Description
`Configuration Layer` specifies all scenario settings, including simulation horizons, enabled markets, optimization options, and asset parameters, serving as the single entry point.

`Input/Output Layer` manages the ingestion, alignment, and validation of external time series data, such as market prices and depot-based flexibility bands, as well as the structured export of results.

`Workflow Layer` coordinates execution across time and market stages, handling repeated optimizations, market gate closures, and progressive commitment of decisions.

`Optimization Layer` contains the mathematical formulation and solver interface that convert aggregated flexibility into market positions and operational schedules under market and physical constraints.

`Domain Layer` includes domain-specific depot abstractions.

`Market Layer` defines market and trading rules such as gate closures.

`Postprocessing Layer` includes result summary generation and plotting.

## Control Workflow
![flex-depot-workflow.svg](images/flex-depot-workflow.svg)

## Installation
### 1) Clone repository
```
git clone https://github.com/marcelbroedel/flex-depot-dev
```

### 2) Create and activate a virtual environment
Recommendation: create and activate a clean virtual environment in the local `.venv/` directory. For Windows PowerShell:
```
cd <root_directory>
py -m venv .venv
.\.venv\Scripts\activate
```

Alternative virtual environment:
```
py -m venv <name_of_virtual_environment>
.\<name_of_virtual_environment>\Scripts\activate
```

Or using conda:
```
conda create -n <name_of_conda_environment> python=3.11
conda activate <name_of_conda_environment>
```

### 3) Install the package
Install the package and its dependencies using pip:
```
python -m pip install --upgrade pip
pip install -e .
```

### 4) Solver
FLEX-DEPOT uses [Pyomo](https://pyomo.readthedocs.io/en/stable/) compatible MILP solvers. The open-source [cbc](https://github.com/coin-or) solver works for many cases. The proprietary [Gurobi](https://www.gurobi.com/solutions/gurobi-optimizer/) solver is recommended for larger problems and is available with a free academic license.

For Gurobi, install the Python bindings:
```
pip install gurobipy==<your_gurobi_version>
```

Verify that your Gurobi version and license version are compatible:
```
pip show gurobipy
grbprobe
```

Verify that Pyomo can access Gurobi:
```
python -c "import pyomo.environ as pyo; print(pyo.SolverFactory('gurobi').available())"
```

### 5) Quick start
The repository contains a ready-to-run example script for Windows:
```
./run_example.bat
```

Or the Unix shell script for Linux/macOS:
```
bash run_example.sh
```

These commands run the example configuration and generate result files in the `results/` directory.

## How to use
FLEX-DEPOT provides three CLI commands.

Run one MPC simulation:
```
python -m flex_dep_opt run-sim --config src/flex_dep_opt/config/settings_example.toml
```

Run postprocessing for the latest simulation:
```
python -m flex_dep_opt run-post
```

or postprocess with an explicit settings file:
```
python -m flex_dep_opt run-post --config src/flex_dep_opt/config/settings_example.toml
```

Run a batch of scenarios:
```
python -m flex_dep_opt run-batch batches/example --jobs 4
```

`run-batch` executes simulation and postprocessing for every TOML file in the given directory, except `default.toml`. If `default.toml` exists, it is used as the shared base configuration and every other TOML file in the directory is merged into it as an override. This keeps scenario variants small:

```
batches/example/
  default.toml
  da.toml
  da_id.toml
  da_id_fcr.toml
```

Batch outputs are written to a timestamped result directory:

```
results/batches/example__YYYY-MM-DD_HH-MM-SS/
  da/
  da_id/
  da_id_fcr/
  manifest.csv
```

Each run directory contains the regular simulation and postprocessing outputs, such as `dispatch.csv`, `commit.csv`, `cashflow.csv`, `kpis.csv`, `dispatch.html`, and `cashflow.html`. If FCR is enabled, `fcr_commit.csv` is written as well. The batch-level `manifest.csv` collects the run directory and KPI row for every scenario.

The manifest output path can be overridden:
```
python -m flex_dep_opt run-batch batches/example --jobs 4 --manifest path/to/manifest.csv
```

Within the TOML configuration file all parameters for a simulation run can be set.

### Simulation settings
| Parameter | Type | Description | Valid values / format |
|---|---|---|---|
| `simulation.start` | str | Simulation start time | `YYYY-MM-DD HH:MM` |
| `simulation.end` | str | Simulation end time | `YYYY-MM-DD HH:MM` |
| `simulation.timestep_hours` | float | Simulation timestep in hours | `0.25` (others not tested) |
| `simulation.name` | str | Identifier used for naming single-run result folders | arbitrary string |
| `simulation.solver` | str | Optimization solver backend | `gurobi`, `cbc` |

### Market configuration
| Parameter | Type | Description | Valid values / format |
|---|---|---|---|
| `optimization.markets.dayahead.enabled` | bool | Enable day-ahead market participation | `true` / `false` |
| `optimization.markets.dayahead.source` | str | CSV file with day-ahead price time series | path to CSV |
| `optimization.markets.dayahead.fee_eur_per_kwh` | float | Transaction fee for day-ahead trades | `>= 0` EUR/kWh |
| `optimization.markets.intraday.enabled` | bool | Enable intraday market participation | `true` / `false` |
| `optimization.markets.intraday.source` | str | CSV file with intraday price time series | path to CSV |
| `optimization.markets.intraday.fee_eur_per_kwh` | float | Transaction fee for intraday trades | `>= 0` EUR/kWh |

### Trading rules
| Parameter | Type | Description | Valid values / format |
|---|---|---|---|
| `optimization.trading.mode` | str | Trading mode controlling market interaction realism | `none`, `realistic` |
| `optimization.trading.dayahead.gate_closure_hour` | str | Day-ahead gate closure time | `HH:MM` |
| `optimization.trading.dayahead.closes_previous_day` | bool | Whether day-ahead closes on day D-1 | `true` / `false` |
| `optimization.trading.intraday.offset_minutes_before_delivery` | int | Intraday trading offset before delivery | `>= 0` minutes |

### FCR configuration
| Parameter | Type | Description | Valid values / format |
|---|---|---|---|
| `optimization.trading.fcr.enabled` | bool | Enable FCR market participation | `true` / `false` |
| `optimization.trading.fcr.prices_source` | str | FCR capacity price file | path to XLSX |
| `optimization.trading.fcr.frequency_source` | str / null | Frequency time series used for activation energy | path to CSV or `null` |
| `optimization.trading.fcr.acceptance_rate` | float | Probability that a positive FCR bid is accepted | `(0, 1]` |
| `optimization.trading.fcr.acceptance_seed` | int / null | Random seed for reproducible acceptance sampling | integer or `null` |
| `optimization.trading.fcr.breakeven_analysis` | bool | Compute per-slot FCR opportunity-cost and breakeven metrics | `true` / `false` |
| `optimization.trading.fcr.breakeven_include_zero_bid` | bool | Also analyze declined zero-bid slots | `true` / `false` |
| `optimization.trading.fcr.gate_closure_hour` | str | FCR gate closure time | `HH:MM` |
| `optimization.trading.fcr.gate_closure_closes_previous_day` | bool | Whether FCR closes on day D-1 | `true` / `false` |
| `optimization.trading.fcr.gate_closure_timezone` | str | Timezone for FCR gate closure | IANA timezone |
| `optimization.trading.fcr.product_hours` | float | FCR product duration | hours |
| `optimization.trading.fcr.bid_block_mw` | float | Minimum bid block size | MW |
| `optimization.trading.fcr.energy_reserve_minutes` | float | Energy reserve duration held for FCR activation | minutes |
| `optimization.trading.fcr.reserve_penalty_eur_per_kwh` | float | Penalty for reserve infeasibility | EUR/kWh |
| `optimization.trading.fcr.balance_penalty_eur_per_kwh` | float | Penalty for FCR balance deviation | EUR/kWh |
| `optimization.trading.fcr.frequency_nominal_hz` | float | Nominal grid frequency | Hz |
| `optimization.trading.fcr.deadband_hz` | float | Frequency deadband before activation starts | Hz |
| `optimization.trading.fcr.full_activation_hz` | float | Frequency deviation for full activation | Hz |

### Imbalance settlement
| Parameter | Type | Description | Valid values / format |
|---|---|---|---|
| `optimization.imbalance.enabled` | bool | Enable imbalance settlement | `true` / `false` |
| `optimization.imbalance.source_pos` | str | CSV file with positive imbalance prices | path to CSV |
| `optimization.imbalance.source_neg` | str | CSV file with negative imbalance prices | path to CSV |
| `optimization.imbalance.imbalance_volume_penalty_eur_per_kwh` | float | Penalty on imbalance energy to discourage usage | `>= 0` EUR/kWh |

### Optimization logic
| Parameter | Type | Description | Valid values / format |
|---|---|---|---|
| `optimization.virtual_arbitrage` | bool | Allow offsetting buy/sell across markets | `true` / `false` |
| `optimization.mpc.da_horizon_hours` | int | MPC prediction horizon for day-ahead optimization | `>= 1` hours |
| `optimization.mpc.id_horizon_hours` | int | MPC prediction horizon for intraday optimization | `>= 1` hours |
| `optimization.mpc.fcr_price_horizon_hours` | int | MPC prediction horizon for FCR capacity prices | `>= 1` hours |
| `optimization.mpc.fcr_frequency_horizon_minutes` | int | Forward-looking frequency activation horizon for FCR | `>= 0` minutes |
| `optimization.mpc.terminal_condition` | bool | Enable soft terminal energy condition | `true` / `false` |
| `optimization.mpc.terminal_weight_eur_per_kwh` | float | Weight for terminal energy deviation penalty | `>= 0` EUR/kWh |

### Flexibility inputs
| Parameter | Type | Description | Valid values / format |
|---|---|---|---|
| `optimization.flexibility.bounds_file` | str | CSV file with aggregated power and energy flexibility bands | path to CSV |
| `optimization.flexibility.cycle_regularization.enabled` | bool | Enable cycling regularization | `true` / `false` |
| `optimization.flexibility.cycle_regularization.cost_eur_per_kwh_throughput` | float | Cost per charged/discharged energy throughput | `>= 0` EUR/kWh |

### Depot settings
| Parameter | Type | Description | Valid values / format |
|---|---|---|---|
| `optimization.depot.eta_grid2depot` | float | Efficiency for grid-to-depot power flow | `(0, 1]` |
| `optimization.depot.eta_depot2grid` | float | Efficiency for depot-to-grid power flow | `(0, 1]` |
| `optimization.depot.grid_connection_limit` | float | Symmetric grid connection limit | `> 0` kW |

### Postprocessing settings
| Parameter | Type | Description | Valid values / format |
|---|---|---|---|
| `postprocessing.reference_driving_energy_costs.enabled` | bool | Enable static-price reference energy benchmark | `true` / `false` |
| `postprocessing.reference_driving_energy_costs.static_price_eur_per_kwh` | float / null | Static benchmark price | EUR/kWh or `null` |
| `postprocessing.reference_driving_energy_costs.energy_column` | str | Flexibility CSV column used for reference energy | column name |

###