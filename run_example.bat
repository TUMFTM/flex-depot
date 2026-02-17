@echo off
REM ================================
REM Flex-Depot: MPC + Plot
REM ================================

cd /d "%~dp0"

set CONFIG=src\flex_dep_opt\config\settings_example.yaml


REM --- Empty result folder completely ---
REM attrib -r results\* /s >nul 2>&1
REM del /f /q results\* /s >nul 2>&1
REM for /d %%D in (results\*) do rmdir /s /q "%%D"

python -m flex_dep_opt run-sim --config "%CONFIG%"
python -m flex_dep_opt run-post --config "%CONFIG%"
