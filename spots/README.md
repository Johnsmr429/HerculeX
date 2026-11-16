# Spots

The spots workflow turns circles drawn with the viewer's Spot tool into rebuilt surfaces, depth sweeps, review pages, and structured verdicts. It reads hand marks without modifying their append-only logs.

## Files

| File | Purpose |
| --- | --- |
| `sources.py` and `catalogue.py` | Locate each standing circle on its source page, pieces, and scan, then write the spots table. |
| `surfaces.py` | Narrows the covered pieces, rebuilds them against the m7 sheet prediction, and renders 67 CT layers. |
| `flatten.py` and `reflatten.py` | Flatten rebuilt meshes while preserving their local surface distances. |
| `sweep.py` | Reads each rendered surface with stock Reader v3 at depths from `-15` through `+15` and in both stack directions. |
| `images.py` | Arranges each sweep as an interactive review page. |
| `pilot.py` | Runs the resumable build, grow, render, read, and image stages for one spot. |
| `database.py` | Stores spots, surfaces, reads, and append-only verdicts in Parquet tables. |

## Commands

```sh
./run.sh spots.catalogue
./run.sh spots.pilot SPOT [build] [grow] [render] [read] [images]
./run.sh spots.database [TABLE]
./run.sh spots.database verdict SPOT SURFACE --by NAME --readable VALUE
```

Use `spots`, `surfaces`, `reads`, or `verdicts` for `TABLE`. A verdict's `VALUE` is `no`, `traces`, `letters`, or `passage`. The workflow stores study data under `${HERCULEX_LIBRARY:-data/library}/studies/shared/spot-candidates/` and publishes pages under `${HERCULEX_OUTPUTS:-data/outputs}/website/sheets/spot-candidates/`. Set `HERCULEX_SPOTS_SINCE=YYYY-MM-DD` to catalog only circles drawn on or after that date.
