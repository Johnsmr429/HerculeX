# level

The `level` package is the complete level 1.1 engine. It turns CT voxels into sheet nodes, sheet nodes into
single-sheet pieces, and pieces into one laid-out sheet using the exact winding-level fit.

| Folder | Contents |
| --- | --- |
| `detection/` | Finds sheet nodes, with their normals, in every 8-voxel cell on the GPU, with a C reference. |
| `level/` | Builds pieces from exact integer winding levels and fits one frame per scan. |
| `layout/` | Places pieces along the roll, builds mosaics, and exports tifxyz grids. |
| `pipeline.c` and `unwrap.inc` | Run the level stages individually or end to end with `./run.sh level`. |

## Outputs

Each scan's outputs are in `unwrapping/level/SCROLL/SCAN/` in the outputs folder:

| Path | Contents |
| --- | --- |
| `nodes/` | A link to the scan's sheet nodes, which are stored in `unwrapping/nodes/`. |
| `pieces/UNIT/` | The pieces of one unit, as tifxyz grids. |
| `pieces/joints/` | The joins between pieces across unit seams. |
| `pieces/frame/` | The scan's frame. |
| `sheet/layout.json` | The place of every piece on the sheet. |

## Units and reruns

Each stage works in units. A stage writes each unit to a sibling folder, swaps it in whole, and writes `record.json`
last. The record names the digest of the code that made the unit, and a stage remakes any unit whose record names a
different digest.

The `same_output_methods.txt` file in `detection/` and `level/` lists earlier digests that are proven to give the same
output, so a refactor that keeps the output doesn't force a rerun. If your change alters the output, empty the list:
an entry only says that two digests agree with each other, and an old entry left in place makes units with the earlier
output count as current.
