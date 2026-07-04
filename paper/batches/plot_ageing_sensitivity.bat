@echo off
setlocal
cd /d "%~dp0\..\.."
rem TODO: replace REPLACE_ME with the actual batch run timestamp once results exist
python paper\scripts\ageing_sensitivity\plot_ageing_sensitivity.py ^
  --manifest results\batches\ageing_sensitivity__REPLACE_ME\manifest.csv ^
  --output-base paper\figures\ageing_sensitivity\ageing_sensitivity
