# Source data for reproducing the example flexibility bands

These files let anyone regenerate the two bundled example virtual-battery (VB)
flexibility bands from raw fleet data:

- `../vb_bounds_example.csv` — bidirectional (V2G) case
- `../vb_bounds_example_uni.csv` — unidirectional (charge-only) case

Run the reproducer from the repository root:

```bash
python scripts/generate_example_vb_bounds.py          # regenerate both files
python scripts/generate_example_vb_bounds.py --check   # verify against the bundled files
```

## What the bands represent

Both bands describe the same electrified truck fleet aggregated into one
depot-level virtual battery, at 15-min resolution. Model parameters
(identical for both cases except the discharge power):

| Parameter | Value |
|-----------|-------|
| Fleet | `fleet_test_id = 3` (12 vehicles) |
| Battery capacity | 572 kWh per vehicle |
| Max charge power | 350 kW per vehicle |
| Max discharge power | **0 kW (uni)** / **350 kW (bidi)** |
| Charge / discharge efficiency | 0.98 / 0.98 |
| SoC bounds | min 0.0, max 1.0; departure SoC target 1.0 |
| Grid resolution / alignment | 15 min, conservative (arrival→ceil, departure→floor) |
| Time anchor | series shifted to start 2026-01-01 (weekday preserved) |
| Energy carry-forward after departure | enabled |

The aggregation method (per-vehicle robust energy envelopes summed into
fleet-level power/energy bounds) is that of **Park et al.**, *"Scalable EV
Flexibility Aggregation with Guaranteed Disaggregation via Fast Projection onto
Reachable Sets: Application to Real Truck Fleet Data"* (methods paper, under
review) — see the paper for the formal derivation.
`scripts/generate_example_vb_bounds.py` is a
stand-alone reimplementation of exactly the path that produced these two files;
it is a reproducibility helper and not part of the `flex_dep_opt` package.

## Provenance of these files

| File | Content | Source |
|------|---------|--------|
| `tracks_with_energy_fleet3.csv` | Raw driving tracks with per-trip energy consumption | GPS fleet data + energy simulation from [TUMFTM/truck_fleet_electrification](https://github.com/TUMFTM/truck_fleet_electrification) (public) |
| `fleet_fleet3.csv` | Vehicle-to-fleet assignment | same repository |
| `activities_constant_charging_350-350_fleet3.csv` | Per-activity state-of-charge reconstruction under a constant-charging scenario (charge/discharge = 350/350 kW), used to set depot-arrival SoC | output of the SoC reconstruction (`sequential_analysis.py`) in the same repository |

All three files are the **`fleet_test_id = 3` slice** of the corresponding
full-fleet files. Because the VB bands are computed per fleet independently,
this slice reproduces the fleet-3 example bands byte-for-byte while keeping the
bundled data small (~2.5 MB instead of ~28 MB). The full-fleet data can be
regenerated from the TUMFTM repository above.

## Terms of use

These files are included solely to make the bundled examples reproducible.
Rights to the underlying fleet data remain with their originators. For any use
beyond reproducing the bundled examples, obtain the data from the original
source under its terms.
