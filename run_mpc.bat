@echo off
REM ================================
REM Flex-Depot: MPC + Plot
REM ================================

cd /d "%~dp0"
call .venv\Scripts\activate

set CONFIG=src\flex_dep_opt\config\settings.yaml


REM --- Empty result folder ---
del /q results\* >nul 2>&1
for /d %%D in (results\*) do rmdir /s /q "%%D"

python -m flex_dep_opt mpc --config "%CONFIG%"
python -m flex_dep_opt plot-results-mpc --config "%CONFIG%"
