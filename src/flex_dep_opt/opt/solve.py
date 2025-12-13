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
    #if options:
    #    for k, v in options.items():
    #        opt.options[k] = v
    opt.options["OutputFlag"] = 0

    # Solve
    results = opt.solve(model, tee=False)

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
    Extract dispatch time series (charging, discharging, SOC, and market positions)
    from a solved multi-market model.

    Handles:
      - p_ch[t]
      - p_dis[t]
      - soc[t]
      - p_market[market, t]  for all markets in model.MARKETS

    Parameters
    ----------
    model : pyo.ConcreteModel
        The solved model.
    time_index : pandas.DatetimeIndex
        Timestamps matching model.T.

    Returns
    -------
    pandas.DataFrame
        Includes physical variables and all market dispatches.
    """
    # Basic consistency check
    T_len = len(list(model.T))
    if len(time_index) != T_len:
        raise ValueError(
            f"time_index length ({len(time_index)}) does not match model horizon ({T_len})."
        )

    # Base dispatch dataframe
    df = pd.DataFrame(index=time_index)

    # Physical variables
    df["p_ch_kw"]  = [pyo.value(model.p_ch[t])  for t in model.T]
    df["p_dis_kw"] = [pyo.value(model.p_dis[t]) for t in model.T]
    df["E_kWh"]  = [pyo.value(model.E[t]) for t in model.T]

    # Add all market variables dynamically
    if hasattr(model, "MARKETS"):
        for mk in model.MARKETS:
            col = f"p_{mk.lower()}_kw"
            df[col] = [pyo.value(model.p_market[mk, t]) for t in model.T]

    return df
