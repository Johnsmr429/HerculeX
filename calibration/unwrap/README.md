# calibration/unwrap

```text
sheet_controls.py                          a band of every scan with published meshes, as sheet nodes with mesh truth, in controls/sheets/
mesh_reference.py, mesh_surfaces.py        reference vertices, tiers, coincidence groups and continuous winding from published meshes
sheet_scoring.py                           recovery, purity and frame agreement of every control
sheet_wrap_truth.py, sheet_wrap_score.py   wrap truth per control and node labels scored against it
sheet_wrap_study.py, sheet_wrap_report.py  contiguous correct area against wrong-wrap rate, side by side
sheet_wrap_survey.py                       contiguous area of every survey band, against published meshes
continuous_area.py                         readable area of a scan in continuous areas of 4 cm² or more
raw_sheet_ruler.py                         levels a unit solve gains along straight lines, against sheets counted in the raw scan
synthetic_scroll.py                        a crushed scroll of exactly known windings, for tests

./run.sh calibration.unwrap.sheet_controls plan | build [SCROLL ...]
./run.sh calibration.unwrap.sheet_scoring [CONTROL ...]
./run.sh calibration.unwrap.sheet_wrap_study [CONTROL ...]     HERCULEX_STUDY_WORKERS, 3 by default
./run.sh calibration.unwrap.continuous_area SCROLL SCAN
./run.sh calibration.unwrap.raw_sheet_ruler SCROLL SCAN UNIT PLANE0 PLANE1 OUT.json     on the board with --link
```
