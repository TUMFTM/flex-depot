## Overview

`flex-depot` is a Python project for optimizing battery/EV fleet flexibility at a logistics depot. It combines
price series (day-ahead, intraday, optional imbalance), mobility flex bands, and market trading rules into a
single rolling-horizon optimization (MPC). The core of the project is a Pyomo model that decides market
positions and physical charging/discharging while respecting fleet power/energy bounds and gate-closure rules.

Key capabilities:
- Unified optimization for DA/ID (and optional imbalance) markets.
- Rolling-horizon MPC that commits market positions when gate closure is reached.
- CSV-based I/O for prices, mobility bounds, and results.
- Plotly-based visualization of dispatch, prices, and cashflows.

## Repository Structure

```
.
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── run_mpc.bat
├── data/
│   ├── mobility/
│   │   └── fleet_3_vb_bounds_15min.csv
│   └── prices/
│       ├── epex_prices_DA.csv
│       ├── example_prices_DA.csv
│       ├── example_prices_ID.csv
│       ├── reBAP_prices_pos.csv
│       └── reBAP_prices_neg.csv
├── results/                   # output directory (created by runs)
└── src/
    ├── __init__.py
    └── flex_dep_opt/
        ├── __init__.py
        ├── __main__.py
        ├── cli.py
        ├── core.py
        ├── config/
        │   ├── __init__.py
        │   ├── load_settings.py
        │   └── settings.yaml
        ├── domain/
        │   ├── __init__.py
        │   ├── site.py
        │   └── vehicle.py
        ├── io/
        │   ├── __init__.py
        │   ├── mobility_io.py
        │   ├── prices_io.py
        │   └── results.io.py
        ├── market/
        │   ├── __init__.py
        │   ├── dayahead.py
        │   ├── intraday.py
        │   └── trading_rules.py
        ├── opt/
        │   ├── __init__.py
        │   ├── model.py
        │   └── solve.py
        ├── viz/
        │   ├── __init__.py
        │   └── plots.py
        └── workflows/
            ├── __init__.py
            ├── mpc_workflow.py
            └── plot_workflow.py
```

## Module Guide

### `flex_dep_opt.cli`
Command-line entry point. Provides subcommands to run MPC and plotting workflows from a config file.
It loads YAML settings and dispatches to the workflow modules.

### `flex_dep_opt.config`
Configuration helpers and the default `settings.yaml`. This YAML file defines simulation windows,
market sources, optimization options (e.g., degradation, gate closures), and output paths.

### `flex_dep_opt.domain`
Lightweight domain models:
- `Vehicle`: efficiency parameters used in the optimization model.
- `Site`: grid connection limit for the depot.

### `flex_dep_opt.io`
CSV I/O and validation utilities:
- `mobility_io.py`: reads and validates mobility flex bands (power/energy bounds).
- `prices_io.py`: loads price series and builds the per-market price dictionary from settings.
- `results.io.py`: writes dispatch summaries to CSV.

### `flex_dep_opt.market`
Market abstractions and trading rules:
- `dayahead.py` / `intraday.py`: typed containers for price series with validation.
- `trading_rules.py`: builds activity masks to respect gate closures in MPC.

### `flex_dep_opt.opt`
Optimization core:
- `model.py`: the Pyomo model (fleet commercialization) with constraints for power, energy,
  efficiencies, and market positions.
- `solve.py`: Gurobi-only solver integration and result extraction helpers.

### `flex_dep_opt.workflows`
End-to-end workflows:
- `mpc_workflow.py`: rolling-horizon MPC loop that builds windows, solves the model,
  commits trades at gate closure, and exports results.
- `plot_workflow.py`: loads results and prices, then generates Plotly HTML reports.

### `flex_dep_opt.viz`
Plotting utilities:
- `plots.py`: Plotly-based visualizations for dispatch, prices, and market cashflows.

## How to Run

### 1) Set up a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

### 2) Install dependencies
This repo does not pin dependencies, but the code uses at least:
`pandas`, `pyomo`, `plotly`, `pyyaml`, `tqdm`, and `gurobipy` (Gurobi).
Install them in your environment, for example:
```bash
pip install -e .
pip install pandas pyomo plotly pyyaml tqdm gurobipy
```

### 3) Configure inputs
Edit `src/flex_dep_opt/config/settings.yaml` to point to your price files and mobility bounds, and to
set simulation and MPC parameters (time window, horizons, gate-closure rules, etc.).

### 4) Run MPC
```bash
python -m flex_dep_opt mpc --config src/flex_dep_opt/config/settings_example.yaml
```
This creates `results/dispatch_mpc.csv` and `results/commit_mpc.csv`.

### 5) Plot results
```bash
python -m flex_dep_opt plot-results-mpc --config src/flex_dep_opt/config/settings_example.yaml
```
This writes an interactive HTML plot (default: `results/dispatch_mpc_plot.html`).

### Windows helper
A convenience script is available:
```bat
run_mpc.bat
```
It activates a local `.venv`, runs MPC, and generates plots.

## Outputs
- `results/dispatch_mpc.csv`: dispatch time series (net power, charging, market positions, SoC, etc.).
- `results/commit_mpc.csv`: gate-closure commitments over time.
- `results/*.html`: Plotly dashboards (if plotting is enabled).

## Notes
- The solver integration in `flex_dep_opt.opt.solve` requires Gurobi.
- Price files are expected to include `time` and `price` columns with timezone-aware timestamps.
