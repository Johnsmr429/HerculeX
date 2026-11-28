# ink/reading

Runs the production ink stage over laid-out pieces and prefetches per-unit CT brick stores ahead of inference.

## Commands

```sh
./run.sh level.pipeline run ink SCROLL [--scan ID]  # refresh layout if changed, then read ink
./run.sh ink SCROLL [SCAN | --scan ID]              # read an existing layout
./run.sh fetch SCROLL [--scan ID]                   # prefetch CT bricks as units finish fitting
./run.sh ink.reading.unit_fetch SCROLL SCAN         # direct unit-fetch invocation
```

For whole scrolls on macOS, queue the read on the [job board](../../runner/README.md#run-a-scroll-level-11).

## Outputs

Writes per-piece scan images and ink maps to `unwrapping/level/SCROLL/SCAN/faces/level-0/UNIT/` (or `faces/level-0-MODEL/UNIT/` for non-default models) in the outputs folder, and optional layer stacks to `stacks/level/SCROLL/SCAN/`. Temporary per-unit CT stores are deleted after reading unless `HERCULEX_KEEP_BRICKS=1` or the read fails.

## Settings

| Variable | Default | Effect |
| --- | --- | --- |
| `HERCULEX_INK_MODEL` | `reader-v3` | Selects an ink model from `./run.sh models`. |
| `HERCULEX_WINDOWS` | `stacks/level/` in outputs | Directory for saved layer stacks. |
| `HERCULEX_KEEP_STACKS` | `1` | Set to `0` to skip saving layer stacks. |
| `HERCULEX_KEEP_BRICKS` | unset | Set to `1` to retain per-unit CT stores after reading. |
| `HERCULEX_SAMPLERS` | `clamp(cores - 2, 1, 6)` | Number of CPU sampling worker threads (up to 6). |
| `HERCULEX_FACES` | `level-0` (or `level-0-MODEL`) | Writes faces into `faces/NAME/` instead of the default path. |
| `HERCULEX_UNITS` | all units | Comma-separated subset of units to read. |

## Files

| File | Role |
| --- | --- |
| `../ink.h` | Native ink-stage driver: overlaps CPU sampling with GPU/ANE inference and manages temporary CT stores. |
| `unit_fetch.c` | Prefetches CT brick stores as piece units complete, stamping each store with the unit record. |
| `same_output_models.txt` | Lists, per checkpoint, current and prior digests proven to produce byte-identical ink maps. |
