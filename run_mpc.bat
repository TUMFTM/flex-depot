@echo off
REM ================================
REM Flex-Depot: MPC + Plot
REM ================================

cd /d "%~dp0"
call .venv\Scripts\activate

set CONFIG=src\flex_dep_opt\config\settings.yaml

python -m flex_dep_opt mpc --config "%CONFIG%"
python -m flex_dep_opt plot-results-mpc --config "%CONFIG%"
