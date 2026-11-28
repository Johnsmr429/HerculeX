# sampling

Stage 5 turns each piece's `tifxyz` segment mesh into a normal-aligned CT surface volume (layer stack) and detects the recto (written face) from surface fiber orientation.

## Commands

```sh
./run.sh sample SCROLL SCAN UNIT PIECE [--turn -1]  # sample one piece for inspection
./run.sh ink SCROLL [SCAN | --scan ID]              # sample all laid-out pieces and read ink
```

`sample` writes `piece-NNNN.png`, `piece-NNNN-layer0.png`, and `piece-NNNN-stack.npy` into `unwrapping/level/SCROLL/SCAN/faces/sample/UNIT/` in the outputs folder.

## Files

| File | Role |
| --- | --- |
| `column_sampling.c` | Bridges and smooths piece meshes, then samples layer stacks directly from a brick store. |
| `column_kernels.c` | Native per-row sampling kernels. |
| `touched_planes.c` | Computes exact z-planes touched per CT brick so fetches download only required slices. |
| `fibre_kernels.c` | Symmetric 1D Gaussian smoothing used for written-face fiber detection (matches SciPy rounding). |
