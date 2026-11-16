# sampling

Stage 5: a piece's mesh to a layer stack of raw scan values along its normals, and which face is written.

```text
column_sampling.py, column_kernels.c, kernel_bindings.py   mesh bridged and smoothed, stacks read straight from a brick store
piece_images.py                                            every piece of a unit: stack, scan images, written face; fetches its CT bricks
touched_planes.py, touched_planes.c                        which brick z-planes a stack reads, so a fetch moves only those
fibre_faces.py, fibre_kernels.py, fibre_kernels.c          the recto identified from fibre direction
sheet_images.py                                            mosaics, margins and grids shared by every sheet image

./run.sh sampling.piece_images SCROLL SCAN [UNITS|all] [largest]
```

Outputs `papyrus-faces/level-0/UNIT/piece-NNNN*.png` in the scan folder.
`HERCULEX_FACE_PROCESSES` sets workers; `HERCULEX_FACE_ALL_LAYERS` keeps every layer across the sheet.
