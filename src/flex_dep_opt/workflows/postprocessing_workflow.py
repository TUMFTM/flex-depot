from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from flex_dep_opt.config.settings import Settings
from flex_dep_opt.io.prices import (
    build_fees_from_settings,
    build_forecast_prices_from_settings,
    build_prices_from_settings,
    get_fcr_frequency_data,
)
from flex_dep_opt.io.results import (
    SimulationResult,
    read_latest_run_pointer,
    save_dispatch_to_csv,
    save_summary_to_csv,
)
from flex_dep_opt.io.time import LOCAL_TIMEZONE, local_config_timestamp_to_utc, validate_regular_index
from flex_dep_opt.market.forecast_error import draw_ar1_shape, perturb_prices
from flex_dep_opt.post.metrics import (
    compute_cashflows_per_step,
    compute_fcr_activation_energy,
    compute_fcr_cashflow_per_slot,
    compute_kpis,
    compute_market_aggregates,
)
from flex_dep_opt.post.plots import plot_market_cashflows_plotly, plot_mpc_dispatch_plotly
from flex_dep_opt.post.reference_energy_costs import (
    DEFAULT_REFERENCE_ENERGY_COLUMN,
    compute_reference_driving_energy_costs,
)

logger = logging.getLogger(__name__)


@dataclass
class Evaluation:
    kpis: dict[str, Any]
    cashflows: pd.DataFrame
    reference_costs: pd.DataFrame | None
    dispatch_figure: go.Figure
    cashflow_figure: go.Figure

    def save(self, run_dir: str | Path) -> Path:
        run_dir = Path(run_dir)
        save_dispatch_to_csv(
            self.cashflows, run_dir / "cashflow.csv", include_time_column=True, output_tz=LOCAL_TIMEZONE
        )
        if self.reference_costs is not None:
            save_dispatch_to_csv(
                self.reference_costs,
                run_dir / "reference_driving_energy_costs.csv",
                include_time_column=True,
                output_tz=LOCAL_TIMEZONE,
            )
        save_summary_to_csv(self.kpis, run_dir / "kpis.csv")
        self.dispatch_figure.write_html(run_dir / "dispatch.html", include_plotlyjs="cdn")
        self.cashflow_figure.write_html(run_dir / "cashflow.html", include_plotlyjs="cdn")
        return run_dir


def evaluate(result: SimulationResult) -> Evaluation:
    """Settle a simulation result on realized prices and compute KPIs, cashflows and figures."""
    settings = result.settings
    sim = settings.simulation

    start = local_config_timestamp_to_utc(sim.start, local_tz=LOCAL_TIMEZONE)
    end = local_config_timestamp_to_utc(sim.end, local_tz=LOCAL_TIMEZONE)

    dispatch = result.dispatch.loc[start:end]
    validate_regular_index(dispatch.index, timestep_hours=float(sim.timestep_hours), name="dispatch index")

    commit_df = result.commits
    if "delivery_time" in commit_df.columns:
        commit_df = commit_df[(commit_df["delivery_time"] >= start) & (commit_df["delivery_time"] <= end)]

    fcr_commit_df = result.fcr_commits if not result.fcr_commits.empty else None

    # -------------------------------------------------------------------------
    # Prices + fees
    # -------------------------------------------------------------------------
    prices_by_market = build_prices_from_settings(settings)
    for mk, s in list(prices_by_market.items()):
        prices_by_market[mk] = s.loc[start:end]
        validate_regular_index(
            prices_by_market[mk].index,
            timestep_hours=float(sim.timestep_hours),
            name=f"{mk} price index",
        )

    fees_by_market = build_fees_from_settings(settings)

    dt = sim.timestep_hours

    # FCR KPIs are sourced from fcr_commit.csv (per-slot EUR, computed at gate
    # closure). The previous implementation multiplied per-step x_fcr by per-step
    # FCR price, which double-counted by the number of steps per slot.
    fcr_kpis: dict = {}
    if fcr_commit_df is not None and not fcr_commit_df.empty:
        accepted = (
            fcr_commit_df[fcr_commit_df["accepted"].astype(bool)]
            if "accepted" in fcr_commit_df.columns
            else fcr_commit_df
        )
        committed_mask = (
            accepted["x_fcr_kw"] > 0
            if "x_fcr_kw" in accepted.columns
            else pd.Series(False, index=accepted.index)
        )
        fcr_kpis = {
            "fcr_revenue_eur": float(accepted["fcr_revenue_eur"].sum())
            if "fcr_revenue_eur" in accepted.columns
            else 0.0,
            "fcr_slots_committed": int(committed_mask.sum()),
            "fcr_avg_capacity_mw": (
                float(accepted.loc[committed_mask, "x_fcr_mw"].mean())
                if "x_fcr_mw" in accepted.columns and committed_mask.any()
                else 0.0
            ),
        }

    # -------------------------------------------------------------------------
    # Compute metrics
    # -------------------------------------------------------------------------
    cf_df = compute_cashflows_per_step(dispatch, prices_by_market, timestep_hours=dt)

    # Add FCR cashflow column (one entry per slot, at the slot start) so the
    # cashflow plot shows one bar per 4 h FCR slot and the gross-profit KPI
    # includes FCR.
    fcr_cf_series = compute_fcr_cashflow_per_slot(cf_df.index, fcr_commit_df)
    if fcr_cf_series is not None:
        total_col = "Total Cashflow [€/step]"
        cum_col = "Cumulative Profit [€]"
        cf_df = cf_df.drop(columns=[c for c in (total_col, cum_col) if c in cf_df.columns])
        cf_df["FCR Cashflow [€/step]"] = fcr_cf_series
        cf_df[total_col] = cf_df.sum(axis=1)
        cf_df[cum_col] = cf_df[total_col].cumsum()

    energy_by_mk, energy_data, cash_data = compute_market_aggregates(
        dispatch, prices_by_market, timestep_hours=dt
    )

    # Surface FCR revenue in the cash sunburst.
    if fcr_kpis.get("fcr_revenue_eur", 0.0) > 0:
        cash_data.append(("FCR", "Sell", float(fcr_kpis["fcr_revenue_eur"])))

    # Surface FCR *activation* energy (actually used capacity) in the energy
    # sunburst — committed MW alone is just an offered headroom; what hit the
    # battery is what's worth showing on the energy chart.
    fcr_act = compute_fcr_activation_energy(dispatch, timestep_hours=dt)
    if fcr_act is not None:
        fcr_buy_kwh, fcr_sell_kwh = fcr_act
        if fcr_buy_kwh > 0:
            energy_data.append(("FCR", "Buy", fcr_buy_kwh))
        if fcr_sell_kwh > 0:
            energy_data.append(("FCR", "Sell", fcr_sell_kwh))
        # Intentionally not added to `energy_by_mk` so the headline
        # buy_kwh / sell_kwh KPIs stay pure scheduled-market volumes.

    # Battery-cycling (degradation-proxy) cost on realized grid-side throughput,
    # mirroring the optimization objective's main deg term (c_deg * sum(p_ch +
    # p_dis) * dt). Folded into gross profit below so the KPI is consistent with
    # the objective. (The objective's small hidden-FCR sub-timestep term is not
    # reconstructed here; the realized p_ch/p_dis throughput is the physical wear.)
    cyc_cfg = settings.optimization.flexibility.cycle_regularization
    c_deg = cyc_cfg.cost_eur_per_kwh_throughput if cyc_cfg.enabled else 0.0
    throughput_kwh = float((dispatch["p_ch_kw"] + dispatch["p_dis_kw"]).sum()) * dt
    cycling_cost_eur = c_deg * throughput_kwh

    kpis = compute_kpis(
        cf_df, energy_by_mk, fees_by_market, commit=commit_df, cycling_cost_eur=cycling_cost_eur
    )
    kpis.update(fcr_kpis)

    pass2_steps = int(dispatch["used_rebap"].astype(bool).sum()) if "used_rebap" in dispatch.columns else 0
    kpis["pass2_steps"] = pass2_steps
    kpis["pass2_fraction_pct"] = round(pass2_steps / len(dispatch) * 100, 4) if len(dispatch) > 0 else 0.0

    # -------------------------------------------------------------------------
    # Price foresight: MPC decisions used forecast prices where configured (an
    # external forecast CSV and/or a synthetic AR(1) forecast error); cashflows
    # above are always settled on realized prices.
    # -------------------------------------------------------------------------
    mk_cfg = settings.optimization.markets
    market_details = (("DA", mk_cfg.dayahead), ("ID", mk_cfg.intraday))
    forecast_markets = [
        mk
        for mk, detail in market_details
        if detail.enabled and (detail.forecast_source or detail.forecast_error.enabled)
    ]
    kpis["price_foresight"] = "forecast" if forecast_markets else "perfect"

    if forecast_markets:
        # Reconstruct the DECISION prices the MPC optimized on so the reported MAE
        # reflects the actual series (forecast CSV where set, else realized, plus
        # the optional AR(1) perturbation). The AR(1) draw must use the SAME index
        # the MPC used: a warm-up lead-in is prepended there (start = report_start
        # - warmup_hours), so with the same seed the shape is reproduced exactly
        # before restricting to the reporting window for the statistic.
        warmup_hours = float(getattr(sim, "warmup_hours", 0.0) or 0.0)
        mpc_start = start - pd.Timedelta(hours=warmup_hours)
        forecast_prices_by_market = build_forecast_prices_from_settings(settings)
        details_by_mk = dict(market_details)
        for mk in forecast_markets:
            decision = forecast_prices_by_market[mk].loc[mpc_start:end]
            fe_cfg = details_by_mk[mk].forecast_error
            if fe_cfg.enabled:
                z_fe = draw_ar1_shape(decision.index, rho=fe_cfg.rho, seed=fe_cfg.seed)
                decision = perturb_prices(decision, z_fe, fe_cfg.sigma_eur_per_mwh)
            decision = decision.loc[start:end]
            realized = prices_by_market[mk]
            if not decision.index.equals(realized.index):
                raise ValueError(
                    f"{mk} decision prices do not cover the simulation window "
                    f"[{realized.index[0]} .. {realized.index[-1]}] — cannot compute forecast MAE."
                )
            mae = float((decision - realized).abs().mean())
            kpis[f"{mk.lower()}_forecast_mae_eur_per_kwh"] = mae

    # -------------------------------------------------------------------------
    # Optional static-price reference scenario for driving energy
    # -------------------------------------------------------------------------
    reference_df = None
    reference_summary = None

    ref_cfg = settings.postprocessing.reference_driving_energy_costs
    if ref_cfg.enabled:
        if ref_cfg.static_price_eur_per_kwh is None:
            raise ValueError(
                "postprocessing.reference_driving_energy_costs.static_price_eur_per_kwh "
                "must be set when the reference calculation is enabled."
            )

        flex_file = settings.optimization.flexibility.bounds_file

        reference_df, reference_summary = compute_reference_driving_energy_costs(
            flex_file,
            start=start,
            end=end,
            static_price_eur_per_kwh=float(ref_cfg.static_price_eur_per_kwh),
            charging_efficiency=float(ref_cfg.charging_efficiency),
            energy_column=str(ref_cfg.energy_column or DEFAULT_REFERENCE_ENERGY_COLUMN),
        )

        # Apply the same cycling cost to the reference so only the *incremental*
        # V2G/arbitrage wear enters the delta. The reference charges grid-side
        # energy to drive (no discharge-to-grid), so its throughput is
        # ref_grid_energy_kwh. Without this, the unidirectional case would be
        # unfairly penalized for baseline charging wear the reference also incurs.
        ref_cycling_cost_eur = c_deg * float(reference_summary["ref_grid_energy_kwh"])
        reference_gross_profit_eur = -float(reference_summary["ref_energy_cost_eur"]) - ref_cycling_cost_eur
        kpis.update(
            {
                "ref_gross_profit_eur": reference_gross_profit_eur,
                "ref_cycling_cost_eur": ref_cycling_cost_eur,
                "ref_driving_energy_kwh": float(reference_summary["ref_driving_energy_kwh"]),
                "ref_grid_energy_kwh": float(reference_summary["ref_grid_energy_kwh"]),
                "ref_charging_efficiency": float(reference_summary["ref_charging_efficiency"]),
                "ref_static_price_eur_per_kwh": float(reference_summary["ref_static_price_eur_per_kwh"]),
                "ref_energy_cost_eur": float(reference_summary["ref_energy_cost_eur"]),
                "total_potential_gross_profit_delta_eur": (
                    float(kpis["gross_profit_eur"]) - reference_gross_profit_eur
                ),
            }
        )

    # -------------------------------------------------------------------------
    # Plot 1: dispatch report
    # -------------------------------------------------------------------------
    fcr_cfg = settings.optimization.trading.fcr
    fcr_frequency_data_pp: pd.DataFrame | None = None
    if fcr_cfg.enabled and getattr(fcr_cfg, "frequency_source", None):
        try:
            fcr_frequency_data_pp = get_fcr_frequency_data(fcr_cfg.frequency_source)
            fcr_frequency_data_pp = fcr_frequency_data_pp.loc[
                (fcr_frequency_data_pp.index >= start) & (fcr_frequency_data_pp.index <= end)
            ]
            # align with the local-time plot axis
            fcr_frequency_data_pp.index = fcr_frequency_data_pp.index.tz_convert(LOCAL_TIMEZONE)
        except Exception:
            fcr_frequency_data_pp = None

    dispatch_plot = dispatch.copy()
    dispatch_plot.index = dispatch_plot.index.tz_convert(LOCAL_TIMEZONE)
    prices_plot = {mk: s.tz_convert(LOCAL_TIMEZONE) for mk, s in prices_by_market.items()}
    commit_plot = commit_df.copy()
    for col in ("delivery_time", "current_time", "next_time", "gate_closure_time"):
        if col in commit_plot.columns and pd.api.types.is_datetime64_any_dtype(commit_plot[col]):
            if getattr(commit_plot[col].dt, "tz", None) is not None:
                commit_plot[col] = commit_plot[col].dt.tz_convert(LOCAL_TIMEZONE)

    fig_dispatch = plot_mpc_dispatch_plotly(
        dispatch=dispatch_plot,
        prices_by_market=prices_plot,
        commit_df=commit_plot,
        fcr_commit_df=fcr_commit_df,
        title="MPC Flexband Dispatch and Market Positions",
        fcr_frequency_data=fcr_frequency_data_pp,
        fcr_product_hours=fcr_cfg.product_hours,
    )

    # -------------------------------------------------------------------------
    # Plot 2: cashflow report (plots consume precomputed metrics)
    # -------------------------------------------------------------------------
    cf_plot = cf_df.copy()
    cf_plot.index = cf_plot.index.tz_convert(LOCAL_TIMEZONE)
    fig_cf = plot_market_cashflows_plotly(
        cf_df=cf_plot,
        energy_data=energy_data,
        cash_data=cash_data,
        kpis=kpis,
        reference_df=(reference_df.tz_convert(LOCAL_TIMEZONE) if reference_df is not None else None),
        reference_summary=reference_summary,
        title="Market Cashflows",
    )

    return Evaluation(
        kpis=kpis,
        cashflows=cf_df,
        reference_costs=reference_df,
        dispatch_figure=fig_dispatch,
        cashflow_figure=fig_cf,
    )


def postprocess_mpc_results(settings: Settings | None = None, run_dir: Path | None = None) -> None:
    if run_dir is None:
        run_dir = read_latest_run_pointer(Path("results"))

    result = SimulationResult.load(run_dir, settings=settings)
    settings = result.settings
    logger.info(f"Settings file used: {settings.get_source_path()}")

    evaluation = evaluate(result)
    kpis = evaluation.kpis
    if "ref_energy_cost_eur" in kpis:
        logger.info(
            "Reference driving energy costs: "
            f"{kpis['ref_energy_cost_eur']:.2f} EUR for "
            f"{kpis['ref_driving_energy_kwh']:.2f} kWh driving energy "
            f"({kpis['ref_grid_energy_kwh']:.2f} kWh grid-side at "
            f"eta={kpis['ref_charging_efficiency']:.3f})"
        )
    evaluation.save(run_dir)

    # Commit logs are only needed as input to this step and dominate the run directory size.
    if not settings.postprocessing.save_commits:
        (run_dir / "commit.csv").unlink(missing_ok=True)
        (run_dir / "fcr_commit.csv").unlink(missing_ok=True)
        logger.info("Commit CSVs removed (postprocessing.save_commits = false)")

    logger.info(f"Result CSV files saved -> {run_dir.as_posix()}")
    logger.info(f"Result HTML plots saved -> {run_dir.as_posix()}")
    logger.info("Postprocessing finished")
