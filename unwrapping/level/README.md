# unwrapping/level

Stage 3 turns sheet nodes into single-sheet pieces from exact integer winding levels. It writes the pieces, and
everything made after them, to `unwrapping/level/` in the outputs folder.

| File | Description |
| --- | --- |
| `run_levels.py` | Runs the stage unit by unit: it writes the pieces as tifxyz grids, joins them across chunk seams, and links the scan's sheet nodes. |
| `sheet_graph.py` | Builds the sheet graph of a field: cells, sheet edges and rays. |
| `piece_growth.py`, `patch_agglomeration.c` | Turn normals to agree along the sheets, and merge small patches into quick pieces. |
| `winding_order.py`, `l1_synchronization.py`, `network_simplex.c` | Choose a unit's sense of winding (its curl) by solving exactly over its quick pieces. |
| `level_pieces.py`, `level_solver.c` | Solve exact integer levels and cut pieces from whole levels. |
| `orientation.py` | Turns all normals to face out of the scroll with one minimum cut. |
| `piece_flattening.py`, `sheet_flattening.c` | Lay each piece flat in frames carried along the sheet, then grid it and bridge small holes. |
| `piece_joiner.py` | Joins the pieces of neighboring chunks across their seam. |
| `scroll_axis.py` | Finds the scroll axis from the sheet normals. |
| `scroll_frame.py` | Fits one frame for a whole scan. |
| `scroll_layout.py` | Places every piece on one sheet by that frame. |
| `kernel_bindings.py`, `piece_kernel_bindings.py`, `piece_kernels.c` | Declare the ctypes signatures of the C kernels, and provide the piece kernels. |
| `same_output_methods.txt` | Lists earlier code digests that are proven to give the same pieces. |

## Commands

```sh
./run.sh runner.pipeline run pieces SCROLL                 # the pipeline stage: every unit, then joins and frame
./run.sh unwrapping.level.run_levels curl SCROLL           # find the scan's sense of winding and keep it in curl.json
./run.sh unwrapping.level.run_levels unit SCROLL UNIT      # one unit
./run.sh unwrapping.level.run_levels finish SCROLL         # joins, frame and layout, once every unit is done
```

For a scan of several units, you can run `curl` first, then each unit as its own board job, then `finish`.

`HERCULEX_PIECE_SCRATCH` sets the folder for the sheet graph's temporary files (default
`runtime/scratch/herculex-pieces`).
