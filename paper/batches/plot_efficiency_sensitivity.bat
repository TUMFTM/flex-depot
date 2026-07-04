@echo off
setlocal
cd /d "%~dp0\..\.."
rem TODO: replace REPLACE_ME with the actual batch run timestamp once results exist
python paper\scripts\efficiency_sensitivity\plot_efficiency_sensitivity.py ^
  --manifest results\batches\efficiency_sensitivity__REPLACE_ME\manifest.csv ^
  --output-base paper\figures\efficiency_sensitivity\efficiency_sensitivity
