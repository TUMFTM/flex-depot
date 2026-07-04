@echo off
setlocal
cd /d "%~dp0\..\.."
rem TODO: replace REPLACE_ME with the actual batch run timestamp once results exist
python paper\scripts\power_sensitivity\plot_power_sensitivity.py ^
  --manifest results\batches\power_sensitivity__2026-07-04_01-48-10_idhorizon1h\manifest.csv ^
  --output-base paper\figures\power_sensitivity\power_sensitivity
