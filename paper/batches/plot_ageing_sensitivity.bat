@echo off
setlocal
cd /d "%~dp0\..\.."
python paper\scripts\ageing_sensitivity\plot_ageing_sensitivity.py ^
  --manifest results\batches\ageing_sensitivity__2026-07-04_04-34-49_idhorizon1h\manifest.csv ^
  --output-base paper\figures\ageing_sensitivity\ageing_sensitivity
