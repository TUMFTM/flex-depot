#!/bin/bash
# ================================
# Flex-Depot: MPC + Plot
# ================================

# Go to the directory of the script
cd "$(dirname "$0")" || exit 1

CONFIG="src/flex_dep_opt/config/settings_quickstart.toml"

# --- Run the simulations ---
python -m flex_dep_opt run-sim --config "$CONFIG"
python -m flex_dep_opt run-post