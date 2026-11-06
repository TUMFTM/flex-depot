# Changelog

All notable changes to FLEX-DEPOT are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each released version is archived on Zenodo (concept DOI
[10.5281/zenodo.21529546](https://doi.org/10.5281/zenodo.21529546)).

## [Unreleased]

### Added
- Grid fee (Netzentgelte) component: energy- and peak-power-based network
  tariffs integrated into the market/cost layer (`market/grid_fees.py`).
- Peak shaving as an explicit, editable operating mode; when disabled, the
  scenario comparison is based on a common peak-power grid fee across scenarios.
- Battery cycling cost `c_cyc` carried through postprocessing metrics, the
  scenario comparison plot, and `aggregate_results`.

### Changed
- Reference peak power corrected to 1400 kW in the illustrative example.
- Example flexibility bands and generator helper script updated to track the
  upstream fix in the flex-depot virtual-battery repository.
- Golden/expected KPIs refreshed to reflect the grid-fee, cycling-cost, and
  flexibility-band changes above.

### Fixed
- Imbalance (IMB) sign bug in postprocessing.
- Missing FCR cycling-cost term in postprocessing.

## [2.1.0] - 2026-10-26

Reproducibility & Python API release.

### Added
- Public Python API namespace with a stable surface
  (`simulate`, `evaluate`, `load_inputs`, `run_batch`, `Settings`,
  `SimulationResult`, `SimulationError`, `Evaluation`) and explicit `__all__`.
- Reproducible example pipeline: illustrative unidirectional scenarios S1–S4,
  AR(1) forecast-error generation, and a reproducible generator for the example
  virtual-battery flexibility bands.
- "Warm-up" lead-in to compute the initial day-ahead schedule, eliminating the
  second optimization pass under perfect foresight.
- Runnable Python API example and an extended README, including an
  auto-generated settings/parameter reference.
- `Settings.from_toml` / `from_dict` with per-field descriptions; strict
  (validated) settings model.
- Explicit `SimulationError`, simulation-result loading, and improved batch
  handling.

### Changed
- Minimum supported Python raised to 3.11 (resolves `tomllib` issues on 3.10).
- Modularized logger, evaluation, simulation results, and input handling into
  dedicated modules; renamed `run_mpc` to `simulate` for clarity.
- Switched runtime diagnostics from `print` to the `logging` framework.
- FCR model refined: hard power constraint added, symmetric capacity
  calculation and charge/discharge exclusivity fixed.

### Fixed
- Several CI and interpreter-selection bugs affecting the quickstart and
  example scenarios.

## [2.0.1] - 2026-07-29

### Changed
- Zenodo DOI and citation metadata updates; quickstart grid-limit adjustment.

### Added
- `CONTRIBUTING.md`.
- Additional tests: arbitrage, efficiency, and flex-band feasibility.

## [2.0.0] - 2026-07-24

### Added
- FCR activation cashflow objective term (`obj_fcr_activation_cashflow`),
  settling FCR droop energy.
- Zenodo DOI, project logo, and architecture/workflow diagrams.
- IO-layer separation for FCR inputs and other market data.

### Changed
- Virtual arbitrage disabled by default.
- Architecture and postprocessing (`aggregate_results`) reworked.

## [1.0.0] - 2026-02-18

### Added
- Initial public release: rolling-horizon MPC optimization framework for
  energy-flexibility commercialization in logistics depots with electric trucks.

[Unreleased]: https://github.com/TUMFTM/flex-depot/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/TUMFTM/flex-depot/compare/v2.0.1...v2.1.0
[2.0.1]: https://github.com/TUMFTM/flex-depot/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/TUMFTM/flex-depot/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/TUMFTM/flex-depot/releases/tag/v1.0.0
