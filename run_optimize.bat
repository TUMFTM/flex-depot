@echo off
REM ================================
REM Flex-Depot: Optimize + Plot
REM ================================

cd /d "%~dp0"
call .venv\Scripts\activate

REM ---- Vehicle parameters (edit as needed) ----
set CAPACITY_KWH=1000
set SOC_MIN=0.10
set SOC_MAX=0.90
set SOC0=0.50
set P_CHARGE_MAX_KW=1000
set P_DISCHARGE_MAX_KW=1000
set ETA_CHARGE=0.95
set ETA_DISCHARGE=0.95

REM ---- Files ----
set PRICE_FILE=data\example_prices.csv
set DISPATCH_FILE=results\dispatch.csv
set PLOT_FILE=results\dispatch_plot.html

echo.
echo =======================================
echo Running Flexibility Optimization
echo =======================================
echo Capacity: %CAPACITY_KWH% kWh
echo Price file: %PRICE_FILE%
echo ---------------------------------------

python -m flex_dep_opt optimize ^
  --prices %PRICE_FILE% ^
  --capacity-kwh %CAPACITY_KWH% ^
  --soc-min %SOC_MIN% ^
  --soc-max %SOC_MAX% ^
  --soc0 %SOC0% ^
  --p-charge-max-kw %P_CHARGE_MAX_KW% ^
  --p-discharge-max-kw %P_DISCHARGE_MAX_KW% ^
  --eta-charge %ETA_CHARGE% ^
  --eta-discharge %ETA_DISCHARGE% ^
  --out %DISPATCH_FILE%

if errorlevel 1 (
  echo.
  echo Optimization failed.
  pause
  exit /b 1
)

echo.
echo Optimization complete. Creating plot...

python -m flex_dep_opt plot-results ^
  --dispatch %DISPATCH_FILE% ^
  --prices %PRICE_FILE% ^
  --capacity-kwh %CAPACITY_KWH% ^
  --out %PLOT_FILE% ^
  --open

if errorlevel 1 (
  echo Plot generation failed.
  pause
  exit /b 1
)

echo.
echo All tasks completed successfully!
pause