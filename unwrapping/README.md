# Unwrapping

The unwrapping pipeline converts CT voxels into sheet nodes, separates those nodes into single-sheet pieces, and places the pieces on one two-dimensional sheet.

## Modules

| Path | Purpose |
| --- | --- |
| `detection/` | This stage detects sheet nodes and normals in CT brick stores. |
| `level/` | This stage solves exact integer levels, creates pieces, joins seams, and fits a frame for the scan. |
| `layout/` | This stage places pieces along the roll and exports sheet layouts. |

## Outputs and resuming

HerculeX writes unwrapping artifacts under `${HERCULEX_OUTPUTS:-data/outputs}/unwrapping/`. A level-route scan contains per-unit pieces, cross-unit joints, a scan frame, and `sheet/layout.json` under `level/SCROLL/SCAN/`.

Each unit is built in a temporary sibling directory and moved into place only after it succeeds. Its `record.json` stores the method digest that produced it, so a changed method rebuilds stale units automatically.

A stage's `same_output_methods.txt` may list earlier digests that are proven to produce identical output. Clear the list when a change alters output.
