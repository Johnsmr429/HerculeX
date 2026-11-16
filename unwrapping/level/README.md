# Level unwrapping

This stage converts sheet nodes into single-sheet pieces by solving exact integer levels. It flattens each piece, joins pieces across unit seams, and fits one frame for the scan.

## Files

| File | Purpose |
| --- | --- |
| `run_levels.py` | Coordinates resumable unit solves and final scan-wide outputs. |
| `sheet_graph.py`, `piece_growth.py`, and `patch_agglomeration.c` | Build the sheet graph, align normals, and merge cells into initial pieces. |
| `winding_order.py`, `l1_synchronization.py`, `network_simplex.c`, `level_pieces.py`, and `level_solver.c` | Solve winding relations and exact integer levels. |
| `orientation.py` | Turns piece normals outward with a minimum cut. |
| `piece_flattening.py` and `sheet_flattening.c` | Flatten and grid each piece at its surface spacing. |
| `piece_joiner.py` | Joins pieces across neighboring unit seams. |
| `scroll_axis.py`, `scroll_frame.py`, and `scroll_layout.py` | Fit the scan axis and frame, then place every piece on one sheet. |
| `kernel_bindings.py`, `piece_kernel_bindings.py`, and `piece_kernels.c` | Provide typed calls to the native solvers. |

## Commands

```sh
./run.sh unwrapping.level.run_levels curl SCROLL
./run.sh unwrapping.level.run_levels unit SCROLL UNIT
./run.sh unwrapping.level.run_levels finish SCROLL
```

For normal operation, run `./run.sh runner.pipeline run pieces SCROLL`. The stage writes pieces and scan-wide metadata under `${HERCULEX_OUTPUTS:-data/outputs}/unwrapping/level/SCROLL/SCAN/`.
