@echo off
setlocal
cd /d "%~dp0\..\.."
python paper\scripts\idhorizon_sensitivity\plot_idhorizon_sensitivity.py ^
  --manifest results\batches\idhorizon_sensitivity__2026-07-05_07-38-19\manifest.csv ^
  --output-base paper\figures\idhorizon_sensitivity\idhorizon_sensitivity
