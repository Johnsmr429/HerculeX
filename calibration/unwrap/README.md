# Unwrapping calibration

This module builds geometric controls and measures the accuracy and usable area of unwrapped surfaces.

## Files

| File | Purpose |
| --- | --- |
| `sheet_controls.py` | Builds control bands from scans that have published mesh references. |
| `mesh_reference.py` and `mesh_surfaces.py` | Convert published meshes into reference vertices, tiers, and continuous grid surfaces. |
| `sheet_scoring.py` | Measures recovery, purity, joints, and frame agreement for each control. |
| `sheet_wrap_truth.py` and `sheet_wrap_score.py` | Build winding truth and score predicted node labels against it. |
| `sheet_wrap_study.py` and `sheet_wrap_report.py` | Measure contiguous correct area at fixed wrong-wrap rates and summarize the studies. |
| `sheet_wrap_survey.py` and `continuous_area.py` | Measure continuous surface area for survey bands and completed scans. |
| `raw_sheet_ruler.py` | Compares solved levels with sheets counted along straight lines in the source scan. |
| `synthetic_scroll.py` | Creates a crushed synthetic scroll with known windings for end-to-end tests. |

## Commands

```sh
./run.sh calibration.unwrap.sheet_controls plan
./run.sh calibration.unwrap.sheet_controls build [SCROLL ...]
./run.sh calibration.unwrap.sheet_scoring [CONTROL ...]
./run.sh calibration.unwrap.sheet_wrap_study [CONTROL ...]
./run.sh calibration.unwrap.continuous_area SCROLL SCAN
./run.sh calibration.unwrap.raw_sheet_ruler SCROLL SCAN UNIT PLANE0 PLANE1 OUT.json
```

Set `HERCULEX_STUDY_WORKERS` to change the wrap study worker count from its default of `3`.
