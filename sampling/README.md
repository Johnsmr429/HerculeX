# Sampling

The sampling stage converts each unwrapped mesh into raw CT layers along its normals and determines the written face.

## Files

| File | Purpose |
| --- | --- |
| `column_sampling.py`, `column_kernels.c`, and `kernel_bindings.py` | Smooth the mesh and sample layer stacks from a CT brick store. |
| `piece_images.py` | Fetches the required CT bricks and writes stacks and scan images for each piece. |
| `touched_planes.py` and `touched_planes.c` | Find the brick planes required by a layer stack. |
| `fibre_faces.py`, `fibre_kernels.py`, and `fibre_kernels.c` | Determine the recto from papyrus fibre direction. |
| `sheet_images.py` | Provides shared mosaic, margin, and grid operations for sheet images. |

## Command

```sh
./run.sh sampling.piece_images SCROLL SCAN [UNIT,UNIT|all] [largest]
```

The command writes piece images under `unwrapping/level/SCROLL/SCAN/faces/level-0/` in the outputs folder. `HERCULEX_FACE_PROCESSES` sets the worker count and defaults to `2`. Set `HERCULEX_FACE_ALL_LAYERS` to retain every sampled layer.
