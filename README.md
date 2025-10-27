# flex-depot
Off-site marketing of depot flexibility


flex-depot/
├─ data/                          # Input data and generated market price files
│  └─ example_prices.csv
│
├─ results/                       # Model output and visualization results
│  ├─ dispatch.csv
│  └─ dispatch_plot.html
│
├─ src/
│  ├─ __init__.py
│  │
│  └─ flex_dep_opt/               # Main Python package (installed as module)
│     ├─ __init__.py
│     ├─ __main__.py              # Entry point for `python -m flex_dep_opt`
│     ├─ cli.py                   # Command-line interface (defines all subcommands)
│     ├─ core.py                  # Generic helper functions / shared logic
│     │
│     ├─ domain/                  # Core domain models
│     │  ├─ __init__.py
│     │  └─ vehicle.py            # Vehicle class and energy storage representation
│     │
│     ├─ market/                  # Market layer (prices and market data)
│     │  ├─ __init__.py
│     │  ├─ dayahead.py           # Day-ahead market interface and preprocessing
│     │  └─ prices_generator.py   # Dummy price series generator (hourly or 15-min)
│     │
│     ├─ io/                      # Input/output file handling
│     │  ├─ __init__.py
│     │  ├─ prices_io.py          # Load or parse price CSVs
│     │  └─ results_io.py         # Write optimization results
│     │
│     ├─ opt/                     # Optimization logic (Pyomo + Gurobi)
│     │  ├─ __init__.py
│     │  ├─ model.py              # Pyomo model formulation
│     │  └─ solve.py              # Solver setup and model execution
│     │
│     └─ viz/                     # Visualization layer (Plotly)
│        ├─ __init__.py
│        └─ plots.py              # Interactive results visualization
│
├─ .venv/                         # Virtual environment (not versioned)
│
├─ run_optimize.bat               # Windows batch for full run (optimize + plot)
├─ pyproject.toml                 # Project metadata and dependencies
├─ requirements.txt               # Required Python packages
└─ README.md                      # Project documentation (you’re here)



Main CLI commands: 
# Generate example prices
python -m flex_dep_opt generate-prices

# Run optimization
python -m flex_dep_opt optimize --prices data/example_prices.csv --capacity-kwh 1000

# Plot results (interactive Plotly)
python -m flex_dep_opt plot-results \
  --dispatch results/dispatch.csv \
  --prices data/example_prices.csv \
  --capacity-kwh 1000 \
  --open