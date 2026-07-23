from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
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


def _make_solver(solver_name: str):
    """
    Build a Pyomo solver factory, raising a RuntimeError if it is unavailable.
    """
    name = solver_name.lower()

    if name == "gurobi":
        try:
            import gurobipy  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "Gurobi (gurobipy) not found. Install it in your environment with: pip install gurobipy"
            ) from e
        solver = pyo.SolverFactory("gurobi")
    elif name == "highs":
        try:
            import highspy  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "HiGHS (highspy) not found. Install it with: pip install highspy"
            ) from e
        solver = pyo.SolverFactory("appsi_highs")
    elif name == "cbc":
        cbc_exe = _cbc_executable()
        solver = pyo.SolverFactory("cbc", executable=cbc_exe) if cbc_exe else pyo.SolverFactory("cbc")
    else:
        solver = pyo.SolverFactory(name)

    if not solver.available(exception_flag=False):
        hint = ""
        if name == "highs":
            hint = " HiGHS not available. Install with: pip install highspy"
        elif name == "cbc":
            hint = (
                " CBC not available. Install via conda: `conda install -c conda-forge coincbc` "
                "or ensure `cbc` is on PATH / set CBC_PATH to cbc.exe."
            )
        elif name == "gurobi":
            hint = " Gurobi solver not available to Pyomo. Check license and gurobipy installation."
        raise RuntimeError(f"Solver '{name}' not available to Pyomo.{hint}")

    return solver


def _apply_solver_options(
    opt: Any,
    solver_name: str,
    *,
    silent: bool = True,
    time_limit_s: int | None = None,
    mip_gap: float | None = None,
    threads: int | None = None,
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

    elif solver_name == "highs":
        if silent:
            opt.options["output_flag"] = False
        if time_limit_s is not None:
            opt.options["time_limit"] = float(time_limit_s)
        if mip_gap is not None:
            opt.options["mip_rel_gap"] = float(mip_gap)
        if threads is not None:
            opt.options["threads"] = int(threads)

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
    solver_name: str = "highs",
    time_limit_s: int | None = None,
    mip_gap: float | None = None,
    threads: int | None = 8,
    tee: bool = False,
) -> pyo.results.SolverResults:
    """
    Solve a Pyomo model using the specified solver (e.g. "gurobi", "cbc", or "highs").

    Parameters
    ----------
    solver_name:
        "highs" (default), "gurobi" or "cbc".
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
    opt = _make_solver(solver_name)

    _apply_solver_options(
        opt,
        solver_name.lower(),
        silent=not tee,
        time_limit_s=time_limit_s,
        mip_gap=mip_gap,
        threads=threads,
    )

    results = opt.solve(model, tee=tee)

    status_ok = results.solver.status == pyo.SolverStatus.ok
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
OBJECTIVE_TERM_NAMES = (
    "obj_energy_cashflow",
    "obj_fcr_revenue",
    "obj_fcr_activation_cashflow",
    "obj_imb_cashflow",
    "obj_fee_cost",
    "obj_cycling_cost",
    "obj_imb_vol_penalty",
    "obj_term_penalty",
    "obj_e_slack_penalty",
    "obj_reserve_penalty",
    "obj_balance_penalty",
)


def extract_objective_terms(model: pyo.ConcreteModel) -> dict[str, float]:
    """
    Evaluate the named objective-term expressions of a solved model.

    Returns a dict with one entry per OBJECTIVE_TERM_NAMES (0.0 if the model
    was built without that component) plus "objective" for the total.
    """
    terms: dict[str, float] = {}
    for name in OBJECTIVE_TERM_NAMES:
        comp = getattr(model, name, None)
        terms[name] = float(pyo.value(comp)) if comp is not None else 0.0
    terms["objective"] = float(pyo.value(model.obj))
    return terms


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
        raise ValueError(f"time_index length ({len(time_index)}) does not match model horizon ({T_len}).")

    df = pd.DataFrame(index=time_index)

    # -------------------------
    # Physical variables (T)
    # -------------------------
    df["p_net_kw"] = [pyo.value(model.p_net[t]) for t in T_list]
    df["p_ch_kw"] = [pyo.value(model.p_ch[t]) for t in T_list]
    df["p_dis_kw"] = [pyo.value(model.p_dis[t]) for t in T_list]

    # -------------------------
    # Energy state (S = 0..N): publish both E[t] and E[t+1] aligned to time_index.
    # -------------------------
    df["E_kWh"] = [pyo.value(model.E[t]) for t in T_list]  # state at timestamp (start of interval)
    df["E_next_kWh"] = [pyo.value(model.E[t + 1]) for t in T_list]  # state after interval

    # ---------------------------------------------------------------
    # FCR (read straight from the model; no recomputation downstream)
    # ---------------------------------------------------------------
    if hasattr(model, "S_FCR") and hasattr(model, "_fcr_slot_starts"):
        # Per-slot bid broadcast to every step inside the slot window.
        x_fcr_by_t = [0.0] * T_len
        slot_duration = pd.Timedelta(hours=getattr(model, "_fcr_product_hours", 4.0))
        for j in model.S_FCR:
            slot_start = model._fcr_slot_starts[j]
            slot_end = slot_start + slot_duration
            x_cleared = float(pyo.value(model.x_fcr[j]))
            if x_cleared <= 0.0:
                continue
            for t, ts in enumerate(time_index):
                if slot_start <= ts < slot_end:
                    x_fcr_by_t[t] = x_cleared

        df["x_fcr_kw"] = x_fcr_by_t
        # p_droop is the FCR activation power in the same convention as p_net
        # and the market positions: + = depot imports, - = depot exports.
        df["p_droop_kw"] = [float(pyo.value(model.p_droop[t])) for t in T_list]

    # -------------------------
    # Market positions (T)
    # -------------------------
    for mk in model.MARKETS:
        col = f"p_{str(mk).lower()}_kw"
        df[col] = [pyo.value(model.p_market[mk, t]) for t in T_list]

    # -------------------------
    # Bands for plotting/debug (align to T)
    # -------------------------
    df["E_lower_kWh"] = [pyo.value(model.E_lower[t]) for t in T_list]
    df["E_upper_kWh"] = [pyo.value(model.E_upper[t]) for t in T_list]

    df["P_lower_kw"] = [pyo.value(model.P_lower[t]) for t in T_list]
    df["P_upper_kw"] = [pyo.value(model.P_upper[t]) for t in T_list]

    # -------------------------
    # Imbalance reBAP (optional)
    # -------------------------
    if hasattr(model, "p_imb_pos"):
        df["p_imb_pos_kw"] = [pyo.value(model.p_imb_pos[t]) for t in model.T]
        df["p_imb_neg_kw"] = [pyo.value(model.p_imb_neg[t]) for t in model.T]

    return df
