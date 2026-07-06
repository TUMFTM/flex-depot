@echo off
setlocal
cd /d "%~dp0\..\.."
python paper\scripts\idhorizon_sensitivity\plot_idhorizon_sensitivity.py ^
  --manifest results\batches\idhorizon_sensitivity__2026-07-06_07-44-04\manifest.csv ^
  --output-base paper\figures\idhorizon_sensitivity\idhorizon_sensitivity ^
  --xscale log
python paper\scripts\idhorizon_sensitivity\plot_idhorizon_sensitivity.py ^
  --manifest results\batches\idhorizon_sensitivity__2026-07-06_07-44-04\manifest.csv ^
  --output-base paper\figures\idhorizon_sensitivity\idhorizon_sensitivity_linear ^
  --xscale linear
