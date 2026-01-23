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
    Extract dispatch time series from a solved multi-market model.

    Convention (new):
      - E[s] is the energy state at the BEGINNING of interval s (state index S = 0..N)
      - Decisions (p_ch, p_dis, p_net, p_market) live on T = 0..N-1
      - Transition: E[t+1] = E[t] + ...
    """
    # Basic consistency check (decisions live on T)
    T_list = list(model.T)
    T_len = len(T_list)
    if len(time_index) != T_len:
        raise ValueError(
            f"time_index length ({len(time_index)}) does not match model horizon ({T_len})."
        )

    df = pd.DataFrame(index=time_index)

    # -------------------------
    # Physical variables (T)
    # -------------------------
    df["p_net_kw"] = [pyo.value(model.p_net[t]) for t in T_list]
    df["p_ch_kw"]  = [pyo.value(model.p_ch[t])  for t in T_list]
    df["p_dis_kw"] = [pyo.value(model.p_dis[t]) for t in T_list]

    # -------------------------
    # Energy state (S or T)
    # -------------------------
    # New model: E is indexed by S (0..N). We publish both E[t] and E[t+1] aligned to time_index.
    if hasattr(model, "S"):
        df["E_kWh"] = [pyo.value(model.E[t]) for t in T_list]           # state at timestamp (start of interval)
        df["E_next_kWh"] = [pyo.value(model.E[t + 1]) for t in T_list]  # state after interval
        # Optional: terminal value (single number) can be useful for debugging
        df.attrs["E_terminal_kWh"] = float(pyo.value(model.E[T_len]))
    else:
        # Fallback to old convention
        df["E_kWh"] = [pyo.value(model.E[t]) for t in T_list]

    # -------------------------
    # Market positions (T)
    # -------------------------
    if hasattr(model, "MARKETS"):
        for mk in model.MARKETS:
            col = f"p_{str(mk).lower()}_kw"
            df[col] = [pyo.value(model.p_market[mk, t]) for t in T_list]

    # -------------------------
    # Bands for plotting/debug (align to T)
    # -------------------------
    if hasattr(model, "E_lower") and hasattr(model, "S"):
        df["E_lower_kWh"] = [pyo.value(model.E_lower[t]) for t in T_list]
        df["E_upper_kWh"] = [pyo.value(model.E_upper[t]) for t in T_list]
        # Optional: next-step bounds aligned to the same row (useful to see upcoming tightening)
        df["E_lower_next_kWh"] = [pyo.value(model.E_lower[t + 1]) for t in T_list]
        df["E_upper_next_kWh"] = [pyo.value(model.E_upper[t + 1]) for t in T_list]
    elif hasattr(model, "E_lower"):
        # Old convention
        df["E_lower_kWh"] = [pyo.value(model.E_lower[t]) for t in T_list]
        df["E_upper_kWh"] = [pyo.value(model.E_upper[t]) for t in T_list]

    if hasattr(model, "P_lower"):
        df["P_lower_kw"] = [pyo.value(model.P_lower[t]) for t in T_list]
        df["P_upper_kw"] = [pyo.value(model.P_upper[t]) for t in T_list]

    # -------------------------
    # Imbalance reBAP
    # -------------------------
    if hasattr(model, "p_imb_pos"):
        df["p_imb_pos_kw"] = [pyo.value(model.p_imb_pos[t]) for t in model.T]
        df["p_imb_neg_kw"] = [pyo.value(model.p_imb_neg[t]) for t in model.T]


    return df