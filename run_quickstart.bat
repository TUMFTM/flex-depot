@echo off
REM ================================
REM Flex-Depot: MPC + Plot
REM ================================

cd /d "%~dp0"

REM Pin the interpreter to THIS repo's venv.
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo ERROR: flex-depot venv interpreter not found at "%PY%". 1>&2
    echo Create it: run  python -m venv .venv  then  .venv\Scripts\pip install -e .  -- or fix the path. 1>&2
    exit /b 1
)

set CONFIG=src\flex_dep_opt\config\settings_quickstart.toml
set RUN_DIR=results\illustrative_example\detail_4day

"%PY%" -m flex_dep_opt run-sim --config "%CONFIG%" --run-dir "%RUN_DIR%"
if errorlevel 1 exit /b 1
"%PY%" -m flex_dep_opt run-post --config "%CONFIG%" --run-dir "%RUN_DIR%"
if errorlevel 1 exit /b 1
"%PY%" examples\illustrative_example\plot_detail.py
if errorlevel 1 exit /b 1
