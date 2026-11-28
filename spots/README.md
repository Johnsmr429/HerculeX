# spots

The `spots` package rebuilds circled regions drawn with the viewer's **Spot** tool into continuous surfaces, runs multi-model ink sweeps across depths `-15` to `+15`, publishes review pages under `/sheets/spot-candidates/`, and stores human verdicts in Parquet tables.

## Commands

Run `./run.sh setup spots` once to install the local DuckDB CLI.

```sh
./run.sh spots.catalogue                                    # map standing circles to pieces, windings, and boxes
./run.sh spots.pilot SPOT [build] [grow] [render] [read] [images]
./run.sh spots.database [TABLE]                             # print table sizes or dump one table as JSONL
./run.sh spots.database verdict SPOT SURFACE --by NAME --readable no|traces|letters|passage \
    [--letters N] [--stack ensemble] [--side inward] [--depth D] [--reading TEXT] [--note TEXT]
./run.sh spots.flatten STACK.npy OUT.npz [--mask VALID.npy]
./run.sh spots.reflatten XYZ.npy STEP OUT.npz REPORT.json [ITERATIONS]
```

Queue spot jobs on the Apple job board:

```sh
./run.sh board add spot-SPOT-build --link --memory 3 -- ./run.sh spots.pilot SPOT build render
./run.sh board add spot-SPOT-read --gpu --engine --memory 5 --after spot-SPOT-build -- ./run.sh spots.pilot SPOT read images
```

View results at `http://localhost:8765/sheets/spot-candidates/index.html` (arrow keys step through depths and stacks).

## Method

- **Targeting:** Every standing circle is targeted by default. Set `HERCULEX_SPOTS_SINCE=YYYY-MM-DD` to target only recent circles. For non-pipeline pages, register `{key, scroll, scan, route, page, pieces}` in `data/library/marks/sheets.json`.
- **Surfaces:** `original` keeps the fitted piece; `m7` snaps each node along its normal onto the nearest published m7 sheet; optional `grow` extends `m7` up to 30 cells (4.5 mm) wherever a sheet lies within 6 voxels along the normal. Each piece under a circle is rebuilt independently.
- **Render and sweep:** Renders 67 layers spaced 1 voxel apart (`d = layer - 33`, positive outward). Runs 17-layer ink windows at every `d` in `-15..+15` in both `outward` and `inward` directions using Reader v3 (on `original`) and all installed models from `./run.sh models` (on `m7`/`grown`).
- **Display:** Stretches each model once across all depths (median to 99.9th percentile), computes z-score ensembles (`ensemble` and 5-depth average), and opens at the depth where the 97th percentile inside the circle peaks.

## Data (`data/library/studies/shared/spot-candidates/`)

| Folder | Contents |
| --- | --- |
| `db/` | Parquet tables (`spots`, `surfaces`, `reads`, `verdicts`). |
| `m7/` | Cached m7 surface-prediction chunks. |
| `spots/SPOT/` | Rebuilt surfaces, CT bricks, depth stacks, and ink reads per spot. |

## Files

| File | Role |
| --- | --- |
| `catalogue.c` | Reads mark logs, maps standing circles to pieces, and updates `spots`. |
| `surfaces.c`, `flatten.c`, `reflatten.c` | Piece cropping, m7 snapping/growth, Kabsch reflattening, and 67-layer rendering. |
| `sweep.c`, `images.c` | Depth-sweep inference, ensemble statistics, and sweep viewer pages. |
| `pilot.c`, `spots.h` | Resumable per-spot orchestrator and Parquet/verdict database interface. |
