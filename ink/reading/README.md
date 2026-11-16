# Ink reading

This module samples every unwrapped piece, determines its written face, and reads an ink map with stock Reader v3.

## Files

| File | Purpose |
| --- | --- |
| `piece_reading.py` | Runs the ink stage and overlaps CT fetching, stack sampling, and tile inference. |
| `spot_depths.py` | Re-reads viewer spots at nearby depths and publishes a panel page. |
| `passage_reads.py` | Creates panel pages for separately requested passage reads. |
| `same_output_models.txt` | Lists earlier reader digests that produce byte-identical maps. |

## Commands

```sh
./run.sh runner.pipeline run ink SCROLL [--scan ID]
./run.sh ink.reading.piece_reading SCROLL SCAN [ORDER.json]
./run.sh ink.reading.piece_reading ahead SCROLL SCAN STORES [ORDER.json]
./run.sh ink.reading.spot_depths SCROLL SCAN SHEET_NAME [MARKS]
```

## Settings and outputs

`HERCULEX_SAMPLERS` sets the sampler count and defaults to `4`. `HERCULEX_WINDOWS` overrides the stack directory. `HERCULEX_UNITS` limits a read to a comma-separated set of units, and `HERCULEX_DEPTH_SHIFTS` selects comma-separated layer offsets with `0` included automatically.

Production reads use `faces/level-0/`. Set `HERCULEX_FACES` to write an isolated read under `faces/NAME/`. Set `HERCULEX_KEEP_BRICKS=1` to retain fetched CT stores or `HERCULEX_KEEP_STACKS=0` to discard sampled stacks after the read.
