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
| `threads_apple/parallel_blocks.h`, `threads_cuda/parallel_blocks.h` | Give the C kernels one loop whose rounds run at once: `dispatch_apply` on the apple lane, an OpenMP loop on the cuda lane. |
| `same_output_methods.txt` | Lists earlier code digests that are proven to give the same pieces as today's code. Empty it when a change alters the pieces, or units made before the change count as done. |

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

## Ray costs

A ray says that the sheet it meets lies one turn further out. By default its cost of a smaller step (it met its own
sheet) is its weight, from its length against the median gap (`piece_growth.ray_weights`), and its cost of a larger
step (it skipped sheets) is `HIDDEN` times that weight. With `piece_growth.RAY_MODEL = 1`, both costs come per ray from
a fitted model of the odds that the ray met its own sheet and that it skipped one (`piece_growth.ray_costs`, read from
the `.npz` file that `HERCULEX_RAY_MODEL` names). The exact solve and its reweighted draws take a per-ray skip cost
(`kernel_bindings.term_weights` and `votes`, argument `ray_up`); without it they behave exactly as before.

## Fold guard

Before a piece is written, the guard lays it flat and counts the cells that hold two layers of the sheet. A piece
with more than `SPREAD` (0.30) of its cells doubled has folded over itself. The guard doesn't refuse such a piece
whole (`level_pieces.GUARD_CUT`, default 2). It drops from each doubled cell every node except the layer nearest the
depth that the cell's single-layer neighbours give, the same rule writing applies to contact cells. The rest splits
along the piece's own edges, and each part is checked again, up to `GUARD_ROUNDS` times. Only what still fails falls
back to quick pieces. `GUARD_CUT = 1` drops whole doubled cells; `0` refuses the whole piece, as before.

On PHerc1667 rows 8-20, the guard had refused 38% of the nodes, which then broke into hundreds of fragments.
Cutting at the doubled cells lifts the area in pieces of 4 cm² from 147 to 199 cm². Written area stays the same
(288 cm²), and rendered, the new pieces are single sheets up to 31 cm long with their fibres continuous. Scans whose
pieces seldom fold, such as PHerc0139 (0.1% refused), are unchanged.
