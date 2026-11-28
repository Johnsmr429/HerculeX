# Lasagna

Native port of Lasagna's seed fit ([villa](https://github.com/ScrollPrize/villa) `lasagna` at `7e6b82164`) for Apple silicon.
It grows cylinder shells from the umbilicus to a seed, fits a mesh and exports it as tifxyz.

## Build

```sh
fit=$(./run.sh port lasagna lasagna_fit)
```

Requires Homebrew `lz4`, and libigl 2.6.0 and Eigen 3.4.0 headers (found under `runtime/deps`, or pass `IGL_INCLUDE_DIR` and `EIGEN_INCLUDE_DIR`).

## Use

```sh
"$fit" fit PRODUCT.lasagna.json CYL_STAGES.json MESH_STAGES.json "X Y Z" MODEL_H OUT_DIR
"$fit" cyl-fit PRODUCT.lasagna.json CYL_STAGES.json "X Y Z" MODEL_H OUT_DIR
"$fit" mesh-fit PRODUCT.lasagna.json MESH_STAGES.json MESH_FLAT.npy MESH_STEP OUT_DIR
```

`fit` runs the cylinder stages of a Lasagna stage file (`cyl_init`, `cyl_grow` with its `cyl_grow_refine` passes, later cylinder stages at the same model step), then the mesh stages of a second stage file on the fused shells, one winding per shell, and writes `OUT_DIR/tifxyz/winding_NNNN.tifxyz`, the fitted mesh (`mesh_flat.npy`, 3 × windings × H × W) and a record of each stage (`mesh.json`). `cyl-fit` and `mesh-fit` run the two halves alone; `cyl-fit` writes every kept shell, the fused shells as `mesh_flat.npy` and `cylinder.json`. The tifxyz directories hold no `model.pt`: the fitted state is `mesh_flat.npy`.

## Status

- Data: a `.lasagna.json` product read from local zarr chunks into the fit's 34³ padded blocks and sampled with the fit's arithmetic, as bytes and differentiably.
- Cylinder stages: the first shell around the umbilicus, each pass's five-level residual pyramid stepped by Adam, the width step ramped while a shell grows and resampled to the grown target, the previous shell's violation-depth barrier (libigl winding numbers and distances), refine passes, and the stop when too many width edges leave the grad_mag mask; losses smooth, z-smooth, z-centre, step, bend, step-push, normal and outside.
- Mesh stages: the mesh as a five-level residual pyramid with losses normal, step, smooth, bend and station (ray–quad anchors to the seed, tracked locally between steps), and the tifxyz export with Lasagna's TIFF layout.
- Not ported: the U-Net inference that makes the product, and mesh terms the seed fit leaves at zero weight (data, pred_dt, winding density, correction points, snapping); a stage file that weights them is refused.

## Verification

The historical PHerc0211 goldens compared sparse samples, cylinder losses, the barrier
volume, the resulting shells, mesh losses and the exported windings against the
original fit. The exported windings agreed within 0.05 voxels across 51 golden tests (20–40 s on M1 Pro vs 164 s CUDA original).
