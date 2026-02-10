#!/bin/bash
# ================================
# Flex-Depot: MPC + Plot
# ================================

# Go to the directory of the script
cd "$(dirname "$0")" || exit 1

CONFIG="src/flex_dep_opt/config/settings_example.yaml"

# --- Empty result folder completely ---
# Remove read-only attribute isn't needed on Linux
rm -rf results/*

# --- Run the simulations ---
python -m flex_dep_opt run-sim --config "$CONFIG"
python -m flex_dep_opt run-post --config "$CONFIG"