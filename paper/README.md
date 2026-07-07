# Paper plotting

This folder keeps paper-specific plotting separate from the simulation model.

## Structure

One subfolder per batch in `scripts/` (plot code) and `figures/` (outputs);
one launcher per batch in `batches/`:

```text
paper/
  batches/  plot_<batch>.bat        (run from anywhere; cds to repo root)
  scripts/  <batch>/plot_<batch>.py
  figures/  <batch>/<figure files>
```

Batches: `market_comparison`, `power_sensitivity`, `efficiency_sensitivity`,
`ageing_sensitivity`, plus `overview` (methodology pipeline figure, Fig. 1;
currently drawn from synthetic placeholder data, no batch results needed).
The efficiency/ageing scripts are placeholders until
their batch results are available; update the `REPLACE_ME` manifest path in
the corresponding `.bat` once a run exists.

Shared style (Elsevier sizing, serif fonts, TUM palette from
`paper/style/TUM.gpl`, manual fleet sizes for per-BET normalization) lives in
`paper/scripts/common.py`. All figure scripts write both PDF and SVG (SVG
with text kept as text, so it stays editable).

## Power sensitivity

Single-column line figure (90 mm): annual gross profit per BET vs.
per-vehicle charging power, one line per fleet. Writes
`paper/figures/power_sensitivity/power_sensitivity.{pdf,svg,csv}`. Runs with
non-ok manifest status are skipped with a warning.

## Market comparison

Run from the repository root:

```powershell
.\paper\batches\plot_market_comparison.bat
```

The script reads the batch `manifest.csv` (path set in the `.bat`) and writes a
single double-column figure (Elsevier full width, 190 mm; panel (a) per fleet,
panel (b) per BET) plus the underlying data:

```text
paper/figures/market_comparison/market_comparison_profit_delta.{pdf,svg,csv}
```