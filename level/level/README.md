# level/level

Stage 3 turns detected sheet nodes into single-sheet `tifxyz` pieces using exact integer winding levels. Outputs live under `unwrapping/level/` in the outputs folder; [level/layout](../layout/README.md) arranges the pieces on one sheet.

## Commands

```sh
./run.sh level.pipeline run pieces SCROLL               # fit all units, then join seams and frame
./run.sh curl SCROLL SCAN 1                             # set hand-verified curl (1 or -1) in curl.json
./run.sh curl SCROLL SCAN                               # print stored curl; fails if unset
./run.sh level.level.umbilicus SCROLL SCAN              # print kept umbilicus and all available sources
./run.sh level.level.umbilicus SCROLL SCAN SOURCE       # select another centerline source
./run.sh level.level.run_levels ready SCROLL SCAN       # verify curl and keep umbilicus
./run.sh level.level.run_levels unit SCROLL UNIT [SCAN] # fit one unit
./run.sh level.level.run_levels finish SCROLL           # join seams, fit frame, and lay out finished units
```

## Curl and umbilicus

- **Curl:** Always set by hand with `./run.sh curl SCROLL SCAN 1` or `-1` before fitting. Curl `1` means the spiral radius increases with `atan2(y - cy, x - cx)` around the center; `-1` means it decreases. Legacy auto-chosen `curl.json` files are rejected.
- **Umbilicus:** Every node's angle, outward normal, seam join, and layout uses the scan's umbilicus at that height `z`. The first fitted unit caches the best available line in `sheet/axis.json` (provenance in `sheet/axis-source.json`). Resolution priority:
  1. Scan override in `records/umbilici/chosen.json` (if present).
  2. Human centerlines in `records/umbilici/vesuvius-center-lines/` (organizers first, then community by name; install from `SharpMeow/vesuvius-center-lines` if not bundled).
  3. Human centerline transferred from another scan of the same scroll.
  4. Model centerline (Center Finder, newest version first).
  5. Open-data scan catalog umbilicus (available out of the box for scans that publish one, such as `PHerc0125` and `PHerc0826`).

## Solver behavior

- **Winding levels:** Minimizes $\sum w_{ab} (\text{up}\cdot\max(r, 0) + \text{down}\cdot\max(-r, 0))$ over patch tangency edges ($k = 0$) and across-sheet normal rays ($k = 1$, larger-step weight `HIDDEN`), solved to certified integer optimality by network simplex across reweighted bootstrap draws.
- **Fold guard:** Before writing a piece, checks for doubled cells. If more than `SPREAD` (`0.30`) of cells are doubled, drops layers farthest from single-layer neighbors, splits along piece edges, and rechecks before falling back to quick pieces.
- **Grid centering:** Fits a local 3×3 sheet plane (≥ 4 nodes) to center grid vertices (84–93% of neighbor steps within 10% of 3D length on PHerc1545/PHerc1667 units) and runs 150 Jacobi iterations across small holes.

## Settings

| Variable | Default | Effect |
| --- | --- | --- |
| `HERCULEX_PIECE_SCRATCH` | `runtime/scratch/herculex-pieces` | Disk backing folder for graph arrays exceeding the RAM cap (1/64 of physical/cgroup RAM, max 512 MiB per process). |

## Files

| File | Role |
| --- | --- |
| `run_levels.c` | Builds the sheet graph, writes `tifxyz` pieces, and joins unit seams. |
| `patch_agglomeration.c` | Merges small patches into quick fallback pieces. |
| `l1_synchronization.c`, `network_simplex.c` | Exact integer L1 winding solver via network simplex. |
| `level_solver.c`, `orientation.c` | Integer level assignment and min-cut outward normal orientation. |
| `sheet_flattening.c`, `piece_joiner.c` | Isometric flattening, hole bridging, and cross-seam piece joins. |
| `umbilicus.c`, `scroll_axis.c` | Umbilicus resolution and scroll-center evaluation. |
| `piece_kernels.c` | Sheet-graph and piece-building C kernels. |
| `threads_apple/parallel_blocks.h`, `threads_cuda/parallel_blocks.h` | Lane-native parallel loops (`dispatch_apply` on Apple, OpenMP on CUDA). |
| `same_output_methods.txt` | Prior code digests proven to emit byte-identical pieces. Clear when output changes. |
