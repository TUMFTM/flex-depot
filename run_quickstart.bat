@echo off
REM ================================
REM Flex-Depot: MPC + Plot
REM ================================

cd /d "%~dp0"

set CONFIG=src\flex_dep_opt\config\settings_quickstart.toml

python -m flex_dep_opt run-sim --config "%CONFIG%"
python -m flex_dep_opt run-post
