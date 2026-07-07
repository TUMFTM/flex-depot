@echo off
setlocal
cd /d "%~dp0\..\.."
python paper\scripts\overview\plot_overview.py
