# unwrapping

Stages 2 to 4: CT voxels to sheet nodes, nodes to single-sheet pieces, pieces to one laid-out sheet.

```text
detection/   sheet nodes with normals per 8-voxel cell, Metal with a C reference
level/       pieces from exact integer levels and one frame per scan
layout/      pieces placed along the roll, mosaics, tifxyz export
```

Outputs: `unwrapping/level/SCROLL/SCAN/` in the outputs folder with `nodes/` (a link to the scan's sheet nodes, kept in `unwrapping/grown/`), `pieces/UNIT/`, `pieces/joints/`, `pieces/frame/`, `sheet/layout.json`.
Each unit is written to a sibling folder and swapped in whole; `record.json`, written last, names the digest of the code that made it.
`same_output_methods.txt` lists earlier digests proven to give the same output, so a refactor that keeps the output does not force a rerun. When a change alters the output, empty the list: an entry only says two digests agree with each other, and an old one left in place makes units of the earlier output count as current. Pieces units must also carry the isometry record (`piece_flattening.UNIT_NEEDS`), which units made before the isometric grids lack.
A unit whose record names any other digest is made again by its stage and replaced.
