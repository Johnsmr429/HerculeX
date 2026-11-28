# level/layout

Stage 4 arranges a scan's fitted pieces and seam joints on one flat sheet (`sheet/layout.json`). Multi-unit scans use `level.level.scroll_layout`; single-unit or band runs use `level.layout.mosaic_layout`.

## Commands

```sh
./run.sh level.level.scroll_layout SCROLL SCAN
./run.sh level.layout.mosaic_layout SCROLL SCAN [framed|rows|band]
./run.sh level.layout.tifxyz_export SCROLL SCAN NAME    # writes website/sheets/NAME/tifxyz/ in outputs
```

`tifxyz_export` links against `libtiff-4`, `json-c`, and `libcrypto` via `pkg-config`.

## Files

| File | Role |
| --- | --- |
| `mosaic_layout.inc` | Joins pieces into mosaics and writes `sheet/layout.json`. |
| `packing.c` | Grows mosaics, orients pieces, and packs occupancy grids with 64-bit bitmasks. |
| `seams.c` | Computes cross-unit seam joints and endpoint coordinates on piece meshes. |
| `axis_fit.c` | Fits a scroll center from sheet normals. |
| `tifxyz_export.c` | Exports a laid-out sheet as a single `tifxyz` grid for VC3D. |
