import logging
from pathlib import Path

import flex_dep_opt as fd

logging.basicConfig(format="%(message)s")
logging.getLogger("flex_dep_opt").setLevel(logging.INFO)

settings = fd.Settings.from_toml("src/flex_dep_opt/config/settings_quickstart.toml")
settings.simulation.name = "python_api_example"

inputs = fd.load_inputs(settings)
result = fd.simulate(settings, inputs)
evaluation = fd.evaluate(result)

for name in ("gross_profit_eur", "ref_gross_profit_eur", "total_potential_gross_profit_delta_eur"):
    print(f"{name}: {evaluation.kpis[name]:.2f}")
print(evaluation.cashflows.drop(columns="Cumulative Profit [€]").sum().round(2))

run_dir = Path("results/python_api_example")
result.save(run_dir)
evaluation.save(run_dir)
print(f"Saved to {run_dir}, open {run_dir / 'dispatch.html'} in a browser.")
