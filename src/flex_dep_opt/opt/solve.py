# solve.py
# Gurobi-only solver integration for Pyomo models.
# - Enforces the use of Gurobi (via gurobipy / SolverFactory("gurobi")).
# - Provides a convenience function to extract dispatch time series.

from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
import pyomo.environ as pyo


def _ensure_gurobi_available() -> None:
    """Raise a RuntimeError if gurobipy/Gurobi is not available to the current Python environment."""
    try:
        import gurobipy  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Gurobi (gurobipy) not found. Install it in your venv with: pip install gurobipy"
        ) from e

    solver = pyo.SolverFactory("gurobi")
    if not solver.available(exception_flag=False):
        raise RuntimeError(
            "Gurobi solver not available to Pyomo. "
            "Check your Gurobi license and ensure 'gurobipy' is installed in the same environment."
        )


def solve_model(
    model: pyo.ConcreteModel,
    *,
    tee: bool = True,
    options: Optional[Dict[str, Any]] = None,
) -> pyo.results.SolverResults:
    """
    Solve a Pyomo model using the Gurobi solver (required).

    Parameters
    ----------
    model : pyo.ConcreteModel
        The Pyomo model to solve.
    tee : bool, optional
        If True, stream solver output to the console (default True).
    options : dict, optional
        Gurobi options to pass via SolverFactory options, e.g. {"Threads": 4, "TimeLimit": 60}.

    Returns
    -------
    pyo.results.SolverResults
        The Pyomo solver result object.

    Raises
    ------
    RuntimeError
        If Gurobi is not available or the solution is not optimal.
    """
    _ensure_gurobi_available()

    # Construct solver and set options (classic Pyomo interface)
    opt = pyo.SolverFactory("gurobi")
    if options:
        for k, v in options.items():
            opt.options[k] = v

    # Solve
    results = opt.solve(model, tee=tee)

    # Basic status checks
    status_ok = (results.solver.status == pyo.SolverStatus.ok)
    term_optimal = (results.solver.termination_condition == pyo.TerminationCondition.optimal)

    if not (status_ok and term_optimal):
        raise RuntimeError(
            f"Gurobi did not report an optimal solution. "
            f"Status={results.solver.status}, Termination={results.solver.termination_condition}"
        )

    return results


def extract_dispatch(model: pyo.ConcreteModel, time_index) -> pd.DataFrame:
    """
    Extract a dispatch time series from a solved model into a pandas DataFrame.

    Expects the model to define variables:
      - m.p_ch[t]   (kW)
      - m.p_dis[t]  (kW)
      - m.soc[t]    (kWh)

    Parameters
    ----------
    model : pyo.ConcreteModel
        A solved Pyomo model with attributes p_ch, p_dis, soc and index set m.T.
    time_index : pandas.DatetimeIndex (or list-like)
        Index to use for the resulting DataFrame (must match the length of m.T).

    Returns
    -------
    pandas.DataFrame
        Columns: ["p_ch_kw", "p_dis_kw", "soc_kwh"], indexed by `time_index`.
    """
    # Basic shape check (optional but helpful)
    T_len_model = len(list(model.T))
    if len(time_index) != T_len_model:
        raise ValueError(f"time_index length ({len(time_index)}) does not match model horizon ({T_len_model}).")

    return pd.DataFrame(
        {
            "p_ch_kw":  [pyo.value(model.p_ch[t])  for t in model.T],
            "p_dis_kw": [pyo.value(model.p_dis[t]) for t in model.T],
            "soc_kwh":  [pyo.value(model.soc[t])   for t in model.T],
        },
        index=time_index,
    )
