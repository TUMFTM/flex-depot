from __future__ import annotations

from typing import Any, Dict, Optional

from pathlib import Path

import pandas as pd
import os
import pyomo.environ as pyo


# =============================================================================
# Solver helpers & availability checks
# =============================================================================
def _cbc_executable() -> str | None:
    """Return a CBC executable path if we can determine one; otherwise None (use PATH)."""
    env = os.getenv("CBC_PATH")
    if env:
        return env

    win_default = r"C:\coin-or\bin\cbc.exe"
    if os.name == "nt" and Path(win_default).exists():
        return win_default

    return None

def _ensure_solver_available(solver_name: str) -> None:
    """
    Raise a RuntimeError if the requested solver is not available to Pyomo.
    """
    solver_name = solver_name.lower()

    if solver_name == "gurobi":
        try:
            import gurobipy  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "Gurobi (gurobipy) not found. Install it in your environment with: pip install gurobipy"
            ) from e
        solver = pyo.SolverFactory("gurobi")

    elif solver_name == "cbc":
        cbc_exe = _cbc_executable()
        solver = pyo.SolverFactory("cbc", executable=cbc_exe) if cbc_exe else pyo.SolverFactory("cbc")

    else:
        # If you want to allow more solvers later, keep this generic.
        solver = pyo.SolverFactory(solver_name)

    if not solver.available(exception_flag=False):
        hint = ""
        if solver_name == "cbc":
            hint = (
                " CBC not available. Install via conda: `conda install -c conda-forge coincbc` "
                "or ensure `cbc` is on PATH / set CBC_PATH to cbc.exe."
            )
        elif solver_name == "gurobi":
            hint = " Gurobi solver not available to Pyomo. Check license and gurobipy installation."
        raise RuntimeError(f"Solver '{solver_name}' not available to Pyomo.{hint}")


def _apply_solver_options(
    opt: Any,
    solver_name: str,
    *,
    silent: bool = True,
    time_limit_s: Optional[int] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
) -> None:
    solver_name = solver_name.lower()

    if solver_name == "gurobi":
        if silent:
            opt.options["OutputFlag"] = 0
        if time_limit_s is not None:
            opt.options["TimeLimit"] = int(time_limit_s)
        if mip_gap is not None:
            opt.options["MIPGap"] = float(mip_gap)
        if threads is not None:
            opt.options["Threads"] = int(threads)

    elif solver_name == "cbc":
        # CBC:
        # - time limit: seconds
        # - relative gap: ratio
        # - threads: threads
        if time_limit_s is not None:
            opt.options["seconds"] = int(time_limit_s)
        if mip_gap is not None:
            opt.options["ratio"] = float(mip_gap)
        if threads is not None:
            opt.options["threads"] = int(threads)
        # silence is mainly controlled via tee in Pyomo


# =============================================================================
# Solve
# =============================================================================
def solve_model(
    model: pyo.ConcreteModel,
    *,
    solver_name: str = "gurobi",
    time_limit_s: Optional[int] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = 8,
    tee: bool = False,
) -> pyo.results.SolverResults:
    """
    Solve a Pyomo model using the specified solver (e.g. "gurobi" or "cbc").

    Parameters
    ----------
    solver_name:
        "gurobi" (default) or "cbc".
    time_limit_s:
        Optional wall-clock time limit in seconds.
    mip_gap:
        Optional relative MIP gap (e.g. 0.01 for 1%).
    threads:
        Optional number of threads.
    tee:
        If True, stream solver output to stdout (useful for debugging).

    Raises
    ------
    RuntimeError
        If the solver is not available or no acceptable solution is reported.
    """
    _ensure_solver_available(solver_name)

    solver_name_l = solver_name.lower()
    if solver_name_l == "cbc":
        cbc_exe = _cbc_executable()
        opt = pyo.SolverFactory("cbc", executable=cbc_exe) if cbc_exe else pyo.SolverFactory("cbc")
    else:
        opt = pyo.SolverFactory(solver_name_l)

    _apply_solver_options(
        opt,
        solver_name_l,
        silent=not tee,
        time_limit_s=time_limit_s,
        mip_gap=mip_gap,
        threads=threads,
    )

    results = opt.solve(model, tee=tee)

    status_ok = (results.solver.status == pyo.SolverStatus.ok)
    term = results.solver.termination_condition

    if not status_ok:
        raise RuntimeError(
            f"Solver '{solver_name}' did not return status OK. "
            f"Status={results.solver.status}, Termination={term}"
        )

    if term != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"Solver '{solver_name}' did not report an optimal solution. "
            f"Status={results.solver.status}, Termination={term}"
        )

    return results



# =============================================================================
# Result extraction
# =============================================================================
def extract_dispatch(model: pyo.ConcreteModel, time_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Extract dispatch time series from a solved multi-market model.

    Convention
    ----------
    - E[s] is the energy state at the BEGINNING of interval s (state index S = 0..N)
    - Decisions (p_ch, p_dis, p_net, p_market) live on T = 0..N-1
    - Transition: E[t+1] = E[t] + ...

    Parameters
    ----------
    model:
        A solved Pyomo ConcreteModel.
    time_index:
        Decision timestamps aligned to the model's T set (length N).

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by `time_index` containing physical dispatch,
        market positions, energy state, and (if present) imbalance variables.
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
    df["p_ch_kw"] = [pyo.value(model.p_ch[t]) for t in T_list]
    df["p_dis_kw"] = [pyo.value(model.p_dis[t]) for t in T_list]

    # -------------------------
    # Energy state (S or T)
    # -------------------------
    # New model: E is indexed by S (0..N). We publish both E[t] and E[t+1] aligned to time_index.
    if hasattr(model, "S"):
        df["E_kWh"] = [pyo.value(model.E[t]) for t in T_list]            # state at timestamp (start of interval)
        df["E_next_kWh"] = [pyo.value(model.E[t + 1]) for t in T_list]   # state after interval
        # Optional: terminal value (single number) can be useful for debugging
        df.attrs["E_terminal_kWh"] = float(pyo.value(model.E[T_len]))
    else:
        # Fallback to old convention
        df["E_kWh"] = [pyo.value(model.E[t]) for t in T_list]

    if hasattr(model, "S") and hasattr(model, "fcr_droop_signal") and hasattr(model, "S_FCR"):
        dt = pyo.value(model.dt)
        eta_c = pyo.value(model.eta_c)
        eta_d = pyo.value(model.eta_d)

        t_to_cleared: dict[int, float] = {}
        if hasattr(model, "_fcr_slot_starts"):
            slot_duration = pd.Timedelta(hours=4)
            for j in model.S_FCR:
                slot_start = model._fcr_slot_starts[j]
                slot_end   = slot_start + slot_duration
                x_cleared  = float(pyo.value(model.x_fcr[j]))
                if x_cleared <= 0.0:
                    continue
                for t, ts in enumerate(time_index):
                    if slot_start <= ts < slot_end:
                        t_to_cleared[t] = x_cleared
        elif hasattr(model, "FCR_JT"):
            for j, t in model.FCR_JT:
                x_cleared = float(pyo.value(model.x_fcr[j]))
                if x_cleared > 0.0:
                    t_to_cleared[t] = x_cleared

        fcr_power_kw = []
        fcr_e_delta  = []

        for t in T_list:
            d_val = float(pyo.value(model.fcr_droop_signal[t]))
            x_cleared = t_to_cleared.get(t, 0.0)

            if x_cleared > 0.0 and d_val != 0.0:
                p_fcr = d_val * x_cleared   
                if d_val >= 0.0:
                    delta = -(1.0 / eta_d) * p_fcr * dt
                else:
                    delta = eta_c * (-p_fcr) * dt
            else:
                p_fcr = 0.0
                delta = 0.0

            fcr_power_kw.append(p_fcr)
            fcr_e_delta.append(delta)

        df["p_fcr_actual_kw"] = fcr_power_kw   
        df["E_fcr_delta_kWh"] = fcr_e_delta    

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
    # Imbalance reBAP (optional)
    # -------------------------
    if hasattr(model, "p_imb_pos"):
        df["p_imb_pos_kw"] = [pyo.value(model.p_imb_pos[t]) for t in model.T]
        df["p_imb_neg_kw"] = [pyo.value(model.p_imb_neg[t]) for t in model.T]

    return df