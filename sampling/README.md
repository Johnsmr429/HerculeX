# sampling

The `sampling` package turns a piece's mesh into a layer stack (stage 5). It samples raw scan values along the mesh
normals, and it finds which face of the sheet carries the writing.

| File | Description |
| --- | --- |
| `column_sampling.py` | Bridges and smooths a piece's mesh, and samples its layer stack straight from a brick store. |
| `column_kernels.c`, `kernel_bindings.py` | Provide the C sampling kernel and its typed Python binding. |
| `piece_images.py` | Builds the layer stack and scan images of every piece of a unit, and fetches the CT bricks that the unit needs. |
| `touched_planes.py`, `touched_planes.c` | Find which z-planes of each brick a stack reads, so a fetch downloads only those planes. |
| `fibre_faces.py`, `fibre_kernels.py`, `fibre_kernels.c` | Identify the recto from the direction of the papyrus fibers. |
| `sheet_images.py` | Provides the mosaics, margins and grids that every sheet image shares. |

## Commands

```sh
./run.sh sampling.piece_images SCROLL SCAN [UNIT,UNIT|all] [largest]
```

Name the units to sample, or pass `all`. Add `largest` to sample only the scan's largest continuous area. The ink
stage (`ink.reading.piece_reading`) runs the same sampling and also reads the ink.

## Outputs

`piece_images` writes to `faces/level-0/UNIT/` in the scan's folder (`unwrapping/level/SCROLL/SCAN/` in the outputs):

- `piece-NNNN.png`: the brightest voxel across the sheet.
- `piece-NNNN-layerD.png`: the scan layer at depth `D`.
- `piece-NNNN-side.json`: which way the piece was turned and which face carries the writing.

## Settings

| Variable | Default | Effect |
| --- | --- | --- |
| `HERCULEX_FACE_PROCESSES` | `2` | Sets the number of worker processes. |
| `HERCULEX_FACE_ALL_LAYERS` | unset | Set to any value to write every scan layer across the sheet, not only the middle one. |
