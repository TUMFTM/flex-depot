@echo off
setlocal
cd /d "%~dp0\..\.."
python paper\scripts\market_comparison\plot_market_comparison.py ^
  --manifest results\batches\market_comparison__2026-07-03_23-45-21_idhorzion1h\manifest.csv ^
  --output-base paper\figures\market_comparison\market_comparison_profit_delta
