# calibrate/unwrap

These tools score sheet detection and unwrapping on bands with published meshes and on
synthetic scrolls with known windings.

| File | Description |
| --- | --- |
| `sheet_wrap_score.c` | Builds controls and winding truth, scores recovery and seam agreement, and surveys band area. |
| `mesh_surfaces.c` | Computes reference surface normals and unwraps connected mesh rows. |
| `sheet_wrap_report.c` | Prints wrap studies side by side as a Markdown table. |
| `continuous_area.c` | Reports how much readable area lies in continuous areas of 4 cm² or more. |
| `raw_sheet_ruler.c` | Compares sheet counts along straight CT lines with the unit solve's levels. |

```sh
./run.sh calibrate.unwrap.sheet_controls plan
./run.sh calibrate.unwrap.sheet_controls build [SCROLL ...]
./run.sh calibrate.unwrap.sheet_controls truth
./run.sh calibrate.unwrap.sheet_wrap_truth [tier-a] CONTROL ...
./run.sh calibrate.unwrap.sheet_scoring [joint] [--curl 1|-1] [CONTROL ...]
./run.sh calibrate.unwrap.sheet_wrap_study [CONTROL ...] [NAME=VALUE ...]
./run.sh calibrate.unwrap.sheet_wrap_survey [PART [PARTS]]
./run.sh calibrate.unwrap.continuous_area SCROLL SCAN
./run.sh calibrate.unwrap.raw_sheet_ruler SCROLL SCAN UNIT PLANE0 PLANE1 OUT.json
```

The scoring commands use the scan's manually recorded curl. Set `HERCULEX_STUDY_WORKERS`
to change the winding study's worker count from three. Run the raw sheet ruler on the
job board with `--link` when it needs to fetch CT.

Use the archived vertex arrays to frame selected reference tiers locally.

```sh
./run.sh calibrate.unwrap.mesh_reference input.npz framed.npz AB --by-name
```

The input holds coordinate and normal columns `x,y,z,nx,ny,nz`, windings `W`, vertex
`unit` and `mesh` indices, `unit_mesh`, mesh `names`, and one-letter mesh `tiers`.
The output retains the input arrays and writes directed coincidence pairs and the selected
tiers' group, winding, duplicate, and unresolved arrays. The `.audit.json` sidecar includes
the directed edge votes, rejected name conflicts, and inconsistent cycles.

Run an isolated winding study with a numerical trial setting.

```sh
./run.sh calibrate.unwrap.sheet_wrap_study CONTROL PIECE_DEPTH=.15
```

Trial settings can tune edge depth, node quality, patch size, ray weights, vote budget,
and guard flattening. `GUARD_CUT=0` uses the guard fallback. Set `HERCULEX_RAY_MODEL` to
an archived model NPZ, then pass `RAY_MODEL=1` to use its same-sheet and skipped-sheet
weights. `RAY_MODEL_SCALE` defaults to 1. Invalid trial values produce an error.
