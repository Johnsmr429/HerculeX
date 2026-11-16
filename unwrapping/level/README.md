# unwrapping/level

Stage 3: sheet nodes to single-sheet pieces from exact integer levels, written with everything after them in `unwrapping/level/` in the outputs folder.

```text
run_levels.py                              the stage per unit, pieces written as tifxyz, joints across chunk seams; links the scan's sheet nodes
sheet_graph.py                             cells, sheet edges and rays of a field
piece_growth.py, patch_agglomeration.c     normals turned to agree along the sheets, patches merged into quick pieces
winding_order.py, l1_synchronization.py, network_simplex.c   the curl of a unit, solved exactly over its quick pieces
level_pieces.py, level_solver.c            exact integer levels and pieces from whole levels
orientation.py                             normals turned out of the body by one minimum cut
piece_flattening.py, sheet_flattening.c    pieces laid flat in frames carried along the sheet, gridded and bridged
piece_joiner.py                            pieces joined across chunk seams
scroll_axis.py                             the scroll axis from sheet normals
scroll_frame.py                            one frame for a whole scan
scroll_layout.py                           every piece placed on one sheet by that frame
kernel_bindings.py, piece_kernel_bindings.py, piece_kernels.c   ctypes signatures of the C kernels; the piece kernels

./run.sh unwrapping.level.run_levels curl SCROLL
./run.sh unwrapping.level.run_levels unit SCROLL UNIT      once per unit
./run.sh unwrapping.level.run_levels finish SCROLL         joints, frame, layout
```

A scan of several units runs these as board jobs. Every scan's sheet nodes stay in `unwrapping/grown/` (`data_roots.NODE_ROUTE`), the folder of an earlier route.
