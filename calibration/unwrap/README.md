# calibration/unwrap

These tools score sheet detection and unwrapping on two kinds of control: bands of scans that have published meshes,
and a synthetic scroll whose windings are known exactly.

| File | Description |
| --- | --- |
| `sheet_controls.py` | Builds a band of every scan that has published meshes, as sheet nodes with mesh truth, in the library's `controls/sheets/`. |
| `mesh_reference.py` | Extracts the reference vertices of a band from the published meshes and works out which meshes lie on the same sheet. |
| `mesh_surfaces.py` | Provides the grid surfaces, normals and scroll axis that references and predictions share. |
| `sheet_scoring.py` | Unwraps every control and scores its recovery, purity and frame agreement. |
| `sheet_wrap_truth.py` | Builds the wrap truth of each control from its reference meshes. |
| `sheet_wrap_score.py` | Scores node labels against the wrap truth. |
| `sheet_wrap_study.py` | Unwraps every control and measures the contiguous correct area against the wrong-wrap rate. |
| `sheet_wrap_report.py` | Prints several wrap studies side by side as a Markdown table. |
| `sheet_wrap_survey.py` | Measures the contiguous area of every survey band and compares it with the published meshes. |
| `continuous_area.py` | Reports how much of a scan's readable area lies in continuous areas of 4 cm² or more. |
| `raw_sheet_ruler.py` | Counts the sheets in the raw scan along straight lines through a unit and compares them with the levels that the unit's solve assigns. |
| `synthetic_scroll.py` | Builds a crushed scroll of exactly known windings, voxel by voxel, for the tests. |

## Commands

```sh
# List the planned control bands, then build them for every scroll or for the scrolls you name.
./run.sh calibration.unwrap.sheet_controls plan
./run.sh calibration.unwrap.sheet_controls build [SCROLL ...]

# Score the controls. Add joint to score two halves of each control joined across a seam.
./run.sh calibration.unwrap.sheet_scoring [joint] [CONTROL ...]
./run.sh calibration.unwrap.sheet_wrap_study [CONTROL ...]

# Measure the continuous area of one scan.
./run.sh calibration.unwrap.continuous_area SCROLL SCAN

# Compare a unit's levels with sheets counted in the raw scan. It fetches CT, so run it on the board with --link.
./run.sh calibration.unwrap.raw_sheet_ruler SCROLL SCAN UNIT PLANE0 PLANE1 OUT.json
```

`HERCULEX_STUDY_WORKERS` sets the number of worker processes for `sheet_wrap_study` (default `3`).
