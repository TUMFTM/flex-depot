@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python -m flex_dep_opt optimize --prices data/example_prices.csv --capacity-kwh 1000
python -m flex_dep_opt plot-results --dispatch results/dispatch.csv --prices data/example_prices.csv --open
pause
