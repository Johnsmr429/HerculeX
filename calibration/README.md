# Calibration

Use these studies to measure unwrapping stages against controls with known geometry before you publish a result.

## Modules

| Path | Purpose |
| --- | --- |
| `unwrap/` | This module builds sheet controls and measures surface recovery, winding consistency, and continuous area. |

## Data and results

The studies read controls from `${HERCULEX_LIBRARY:-data/library}/controls/` and store large intermediate results under `calibration/` in the same library. They write reviewed summary measurements to `records/surface/` in this repository.

Add a measurement to a record only after you run its control with the method that produced the measurement.
