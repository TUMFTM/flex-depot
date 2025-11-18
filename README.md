flex-depot
--------------------------------------------------
Installation & Setup
--------------------------------------------------

1. Create a virtual environment:
   python -m venv .venv

2. Activate it:
   Windows:
       .venv\Scripts\activate
   macOS / Linux:
       source .venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Run the CLI:
   python -m flex_dep_opt

--------------------------------------------------
Available CLI Commands
--------------------------------------------------

1) vehicle-info

    Displays the default parameters of the Vehicle model.

    Example:
    flex-dep-opt vehicle-info --capacity-kwh 40

------------------
2) generate-prices

    Generates dummy 15-minute day-ahead price data for testing.

    Example:
    flex-dep-opt generate-prices --out data/example_prices.csv

    Output:
    time, price   (price in EUR/kWh)

---------------
3) import-epex

    Converts a raw EPEX quarter-hour price CSV
    (e.g., Gro_handelspreise_..._Viertelstunde.csv)
    into a standardized output:
    time (timezone-aware), price (EUR/MWh)

    Example:
    flex-dep-opt import-epex path/to/raw.csv --out data/epex_dayahead.csv


------------
4) optimize

    Runs a simple storage optimization (e.g., EV battery, stationary battery).

    Example:
    flex-dep-opt optimize --prices data/epex_dayahead.csv --capacity-kwh 40 --out results/dispatch.csv

    Key parameters:

    --capacity-kwh         Storage capacity

    --soc-min/max/0        State-of-charge limits and initial value

    --p-charge-max-kw      Max charging power

    --p-discharge-max-kw   Max discharging power

    --eta-charge/discharge Efficiencies

    Output columns:
    p_ch_kw, p_dis_kw, soc_kwh

----------------
5) plot-results

    Visualizes dispatch results, SOC evolution, and optional price data.

    Example:
    flex-dep-opt plot-results --dispatch results/dispatch.csv --prices data/epex_dayahead.csv --capacity-kwh 40 --out results/dispatch_plot.html --open

    Produces an interactive Plotly visualization and can open it automatically.


