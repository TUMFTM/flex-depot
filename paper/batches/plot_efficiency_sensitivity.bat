@echo off
setlocal
cd /d "%~dp0\..\.."
python paper\scripts\efficiency_sensitivity\plot_efficiency_sensitivity.py ^
  --manifest results\batches\efficiency_sensitivity__2026-07-04_02-11-17_idhorizon1h\manifest.csv ^
  --output-base paper\figures\efficiency_sensitivity\efficiency_sensitivity
