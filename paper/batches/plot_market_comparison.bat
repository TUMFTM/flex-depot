@echo off
setlocal
cd /d "%~dp0\..\.."
python paper\scripts\market_comparison\plot_market_comparison.py ^
  --manifest results\batches\market_comparison__REPLACE_WITH_NEW_TIMESTAMP\manifest.csv ^
  --output-base paper\figures\market_comparison\market_comparison_profit_delta
