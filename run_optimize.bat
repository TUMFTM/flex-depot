@echo off
REM ================================
REM Flex-Depot: Optimize + Plot
REM ================================

cd /d "%~dp0"
call .venv\Scripts\activate

set CONFIG=src\flex_dep_opt\config\settings.yaml

python -m flex_dep_opt optimize --config "%CONFIG%"
python -m flex_dep_opt plot-results --config "%CONFIG%"