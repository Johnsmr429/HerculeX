# HerculeX for agents

HerculeX unwraps Herculaneum scrolls from X-ray CT (`trace 1.5` default, `level 1.1` fallback) and reads their ink. Read a module's `README.md` before changing it.

## Rules

- Run every module from repo root via `./run.sh MODULE ARGS` (`./run.sh list` shows all modules).
- Resolve data paths through `storage/data_roots.c` (`HERCULEX_DATA`, `HERCULEX_LIBRARY`, `HERCULEX_OUTPUTS`); never hardcode absolute paths.
- Keep lane-specific code inside `*_apple` or `*_cuda` directories. Test the available lane; leave unavailable lanes unchanged and report them as unverified.
- Unit `record.json` hashes source digests; editing stage code invalidates cached units. Only add an old digest to `same_output_methods.txt` (or `ink/reading/same_output_models.txt`) after proving byte-identical output.
- On `apple`, queue GPU, Neural Engine, network, or long-running jobs on the board, and check `./run.sh board room GB` before manual GPU runs.
- Treat `data/library/marks/` as append-only user data; never rewrite or delete it.
- Never hand-edit published viewer HTML; publish via the pipeline and add extra pages with `./run.sh card PAGE`.
- Write no comments in code and at most one-sentence docstrings. Verify edits with `./run.sh check lint --fix && ./run.sh check lint && ./run.sh check cpu` (plus the affected GPU lane if available).
- Always name the dataset and hardware when reporting measurements.

## Modules

Flow: `scans` → `unroll` (default trace 1.5) or `level` (level 1.1) → `sampling` + `ink` → `viewer` ⇄ `spots`.

| Module | Role | Guide |
| --- | --- | --- |
| `unroll/` | Default trace 1.5 pipeline: slab tracing, stitching, strip layout, preview. | [unroll/README.md](unroll/README.md) |
| `level/` | Level 1.1 pipeline: sheet detection, integer winding solve, piece layout. | [level/README.md](level/README.md) |
| `scans/` | Open-data scan catalog, byte-range CT transfer, and chunk store. | [scans/README.md](scans/README.md) |
| `sampling/` | Normal-aligned depth stacks and fiber-based written-face selection. | [sampling/README.md](sampling/README.md) |
| `ink/` | Ink inference (`ink.h`), native Apple/CUDA readers, Studio worker & `studio_cuda/`, CT prefetch. | [ink/README.md](ink/README.md) |
| `viewer/` | Tile pyramid composer, WebGL2 viewer, start page, local server, mark logs. | [viewer/README.md](viewer/README.md) |
| `spots/` | Rebuilds circled viewer spots across depths `-15..+15` and tracks verdicts. | [spots/README.md](spots/README.md) |
| `runner/` | Lane detection (`apple`/`cuda`), job board, CUDA scheduler, build/test/lint. | [runner/README.md](runner/README.md) |
| `storage/` | Data roots and binary brick, node, route, and tile stores. | [storage/README.md](storage/README.md) |
| `calibrate/` | Frozen mesh-truth controls (`bench/`) and unwrap scoring (`unwrap/`). | [calibrate/bench/README.md](calibrate/bench/README.md) |
| `ml-experimental/` | Experimental CUDA models outside the default pipeline (`logos/`). | [ml-experimental/logos/README.md](ml-experimental/logos/README.md) |
| `ports/` | Native Apple silicon ports (`lasagna/`, `scrollfiesta/`, `spiral/`, `surface/`). | [ports/surface/README.md](ports/surface/README.md) |
| `synthetic/` | Procedural 2D/3D CT scroll generators and fragment injection benchmarks. | [synthetic/README.md](synthetic/README.md) |
| `tests/` | Native C and shell golden and end-to-end test suite. | [tests/README.md](tests/README.md) |

## Core workflows

**Unwrap and publish (trace 1.5 default):**
```sh
./run.sh setup
./run.sh catalog SCROLL
./run.sh curl SCROLL SCAN SIGN        # hand-verified spiral chirality: 1 or -1
./run.sh unwrap SCROLL:SCAN            # add --queue (apple board), --from STAGE, or --rotate180
./run.sh serve                         # verify at http://localhost:8765/sheets/index.html
```

- Single trace stage: `./run.sh unroll SCROLL SCAN STAGE` (`plan|trace|stitch|layout|root|read|page`).
- Level 1.1 pipeline: `./run.sh level SCROLL:SCAN`, `./run.sh status SCROLL`, `./run.sh level.pipeline run STAGE SCROLL [--scan ID]`.
- Umbilicus override: `./run.sh level.level.umbilicus SCROLL SCAN [SOURCE]` (stored in `sheet/axis.json`).
- Ink models: `./run.sh models`, `./run.sh models add NAME "TITLE" FILE [reader|studio]`; select via `UNROLL_INK=NAME` (trace) or `HERCULEX_INK_MODEL=NAME` (level). Non-default reads write `faces/level-0-MODEL`.
- Operations: `./run.sh publish SCROLL SCAN`, `./run.sh viewer.overview PAGE OUT.jpg`, `./run.sh board status|log|retry`, `./run.sh runner.disk`, `./run.sh calibrate.bench.score CONTROL PATH`.

**Act on viewer marks (`data/library/marks/.../marks.jsonl`):**
```sh
./run.sh spots.catalogue
./run.sh board add spot-SPOT-build --link --memory 3 -- ./run.sh spots.pilot SPOT build render
./run.sh board add spot-SPOT-read --gpu --engine --memory 5 --after spot-SPOT-build -- ./run.sh spots.pilot SPOT read images
./run.sh spots.database verdict SPOT SURFACE --by NAME --readable no|traces|letters|passage
```

## Common errors

- `no umbilicus on hand` → install centerlines in `records/umbilici/vesuvius-center-lines/` or supply `sheet/axis.json`.
- `holds a curl the level fit once chose itself` → set manually with `./run.sh curl SCROLL SCAN 1` (or `-1`).
- `HERCULEX_INK_MODEL` / `no checkpoint at` → run `./run.sh models` and `./run.sh weights fetch [MODEL]`.
- `this machine runs the apple lane` → unset `HERCULEX_LANE`.
- `HerculeX data folder is unavailable` → mount the data drive or set `HERCULEX_DATA`.
- Units rerun unexpectedly → stage code digest changed; do not whitelist in `same_output_methods.txt` without byte-identical proof.
