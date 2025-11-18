import argparse                             # read arguments from cmd
from .core import message
from .domain.vehicle import Vehicle
from .market.prices_generator import write_example_prices_csv, write_from_epex_DA_csv
from .opt.model import build_single_vehicle_model
from .opt.solve import solve_model, extract_dispatch
from .viz.plots import plot_dispatch_plotly
import pandas as pd
import webbrowser
import sys



def main() -> None:
    parser = argparse.ArgumentParser(
        prog="flex-dep-opt",
        description="V2G / stationary storage – step-by-step CLI"
    )
    sub = parser.add_subparsers(dest="cmd")

    # Vehicle info command
    # Create a dummy vehicle to read default values
    default_vehicle = Vehicle(capacity_kwh=1.0)  # capacity dummy, just to read other defaults
    p_vi = sub.add_parser("vehicle-info")
    p_vi.add_argument("--capacity-kwh", type=float, required=True)
    p_vi.add_argument("--soc-min", type=float, default=default_vehicle.soc_min)
    p_vi.add_argument("--soc-max", type=float, default=default_vehicle.soc_max)
    p_vi.add_argument("--soc0", type=float, default=default_vehicle.soc0)
    p_vi.add_argument("--p-charge-max-kw", type=float, default=default_vehicle.p_charge_max_kw)
    p_vi.add_argument("--p-discharge-max-kw", type=float, default=default_vehicle.p_discharge_max_kw)
    p_vi.add_argument("--eta-charge", type=float, default=default_vehicle.eta_charge)
    p_vi.add_argument("--eta-discharge", type=float, default=default_vehicle.eta_discharge)

    # generate-prices command (dummy)
    p_gen = sub.add_parser("generate-prices", help="Generate a 24h dummy day-ahead price CSV")
    p_gen.add_argument("--out", default="data/example_prices.csv", help="Path to output CSV file")

    # prepare epex prices command
    p_epex = sub.add_parser("import-epex",help="Convert historic EPEX day-ahead CSV to standard time/price CSV")
    p_epex.add_argument("src",help="Path to raw EPEX CSV file (z.B. Gro_handelspreise_..._Viertelstunde.csv)")
    p_epex.add_argument("--out",default="data/epex_dayahead.csv",help="Path to output cleaned CSV file")

    # generate optimization command
    p_opt = sub.add_parser("optimize", help="Run simple day-ahead storage optimization with Gurobi")
    p_opt.add_argument("--prices", required=True, help="Path to CSV file with columns: time, price")
    p_opt.add_argument("--capacity-kwh", type=float, required=True, help="Storage capacity in kWh")
    p_opt.add_argument("--out", default="results/dispatch.csv", help="Path to output CSV for dispatch results")
    _vehicle_defaults = Vehicle(capacity_kwh=1.0)
    p_opt.add_argument("--soc-min", type=float, default=_vehicle_defaults.soc_min)
    p_opt.add_argument("--soc-max", type=float, default=_vehicle_defaults.soc_max)
    p_opt.add_argument("--soc0", type=float, default=_vehicle_defaults.soc0)
    p_opt.add_argument("--p-charge-max-kw", type=float, default=_vehicle_defaults.p_charge_max_kw)
    p_opt.add_argument("--p-discharge-max-kw", type=float, default=_vehicle_defaults.p_discharge_max_kw)
    p_opt.add_argument("--eta-charge", type=float, default=_vehicle_defaults.eta_charge)
    p_opt.add_argument("--eta-discharge", type=float, default=_vehicle_defaults.eta_discharge)

    # generate plot command
    p_plot = sub.add_parser("plot-results", help="Visualize dispatch and price data using Plotly")
    p_plot.add_argument("--dispatch", required=True,
                        help="Path to dispatch CSV (with columns p_ch_kw, p_dis_kw, soc_kwh)")
    p_plot.add_argument("--prices", help="Optional path to day-ahead price CSV (columns: time, price)")
    p_plot.add_argument("--out", default="results/dispatch_plot.html", help="Path to output HTML file")
    p_plot.add_argument("--open", action="store_true", help="Open the HTML file automatically in your browser")
    p_plot.add_argument("--capacity-kwh", type=float, required=True, help="Nominal capacity used to compute SoC [%]")

    args = parser.parse_args()

    # Default: no subcommand
    if args.cmd is None:
        print(message())
        return

    if args.cmd == "vehicle-info":
        veh = Vehicle(
            capacity_kwh=args.capacity_kwh,
            soc_min=args.soc_min,
            soc_max=args.soc_max,
            soc0=args.soc0,
            p_charge_max_kw=args.p_charge_max_kw,
            p_discharge_max_kw=args.p_discharge_max_kw,
            eta_charge=args.eta_charge,
            eta_discharge=args.eta_discharge,
        )
        print("Vehicle:")
        print(f"  capacity_kwh        : {veh.capacity_kwh:.2f}")
        print(f"  soc_min / soc_max   : {veh.soc_min:.2f} .. {veh.soc_max:.2f}")
        print(f"  soc0                : {veh.soc0:.2f}")
        print(f"  p_charge_max_kw     : {veh.p_charge_max_kw:.2f}")
        print(f"  p_discharge_max_kw  : {veh.p_discharge_max_kw:.2f}")
        print(f"  eta_charge          : {veh.eta_charge:.3f}")
        print(f"  eta_discharge       : {veh.eta_discharge:.3f}")
        return

    if args.cmd == "generate-prices":
        path = write_example_prices_csv(args.out)
        print(f"Dummy day-ahead prices written to: {path}")
        return

    if args.cmd == "import-epex":
        path = write_from_epex_DA_csv(src_path=args.src, dst_path=args.out)
        print(f"EPEX day-ahead prices written to: {path}")

    if args.cmd == "optimize":
        df = pd.read_csv(args.prices)
        if "time" not in df or "price" not in df:
            print("Error: CSV must contain columns 'time' and 'price'", file=sys.stderr)
            sys.exit(2)

        # 1) Parse timestamps robustly: force UTC, then convert to target tz
        ts_utc = pd.to_datetime(df["time"], errors="coerce", utc=True)  # Series[datetime64[ns, UTC]]
        if ts_utc.isna().any():
            print("Error: could not parse some timestamps in 'time' column", file=sys.stderr)
            sys.exit(2)

        idx = ts_utc.dt.tz_convert("Europe/Berlin")  # tz-aware, consistent
        # (Optional) sort and drop dupes if any
        order = idx.argsort()
        idx = idx.iloc[order]
        prices = pd.Series(df["price"].astype(float).values, index=idx).iloc[order]

        # 2) Create Vehicle object
        veh = Vehicle(
            capacity_kwh=args.capacity_kwh,
            soc_min=args.soc_min,
            soc_max=args.soc_max,
            soc0=args.soc0,
            p_charge_max_kw=args.p_charge_max_kw,
            p_discharge_max_kw=args.p_discharge_max_kw,
            eta_charge=args.eta_charge,
            eta_discharge=args.eta_discharge,
        )

        # 3) Build and solve optimization model
        model = build_single_vehicle_model(veh, prices)
        solve_model(model)

        # 4) Extract dispatch and save to CSV
        dispatch = extract_dispatch(model, prices.index)
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        dispatch.to_csv(args.out, index=True)

        print(f" Optimization finished. Results written to: {args.out}")
        return

    if args.cmd == "plot-results":
        from pathlib import Path
        # 1) Load dispatch file
        try:
            df_dispatch = pd.read_csv(args.dispatch)
            # try to detect time column
            if "time" in df_dispatch.columns:
                idx = pd.to_datetime(df_dispatch["time"], errors="coerce", utc=True).dt.tz_convert("Europe/Berlin")
                dispatch = df_dispatch.drop(columns=["time"])
                dispatch.index = idx
            else:
                # assume first column is the datetime index
                idx = pd.to_datetime(df_dispatch.iloc[:, 0], errors="coerce", utc=True).dt.tz_convert("Europe/Berlin")
                dispatch = df_dispatch.copy()
                dispatch.index = idx
            dispatch = dispatch.sort_index()
        except Exception as e:
            print(f"Error reading dispatch file: {e}", file=sys.stderr)
            sys.exit(1)

        # 2) Optionally load prices
        prices = None
        if args.prices:
            try:
                df_prices = pd.read_csv(args.prices)
                ts_utc = pd.to_datetime(df_prices["time"], errors="coerce", utc=True)
                prices = pd.Series(df_prices["price"].astype(float).values, index=ts_utc.dt.tz_convert("Europe/Berlin"))
            except Exception as e:
                print(f"Error reading prices file: {e}", file=sys.stderr)
                sys.exit(1)

        # 3) Generate Plotly figure
        fig = plot_dispatch_plotly(dispatch, prices, capacity_kwh=args.capacity_kwh)

        # 4) Save to HTML
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        print(f" Plot saved to: {output_path.resolve()}")

        # 5) Optionally open in browser
        if args.open:
            webbrowser.open(str(output_path.resolve()))

        return
