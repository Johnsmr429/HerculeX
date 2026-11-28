# tests

The native suite runs C and shell end-to-end tests and goldens. These tests build small
fixtures or compare results with a saved reference. `./run.sh check` runs them one at a
time by default. Use `--workers N` to set concurrency.

```sh
./run.sh setup spots
./run.sh board room 3
./run.sh check cpu
./run.sh test-source tests/layout_golden.c
./run.sh test-source tests/unroll_pipeline_e2e.c
./run.sh check lint
```

`setup spots` installs the DuckDB CLI used by the spot fixtures. Set
`HERCULEX_DUCKDB` to use a DuckDB executable you already have.

The unroll pipeline fixture checks trace failures, queue dependencies, saved settings, resuming reads, publication
and per-model face backups with stub stages. It needs no CT downloads or GPU.

On Apple silicon, use the job board for the GPU and reader tests.

```sh
./run.sh board room 3
./run.sh board add full-check --gpu --engine --memory 3 --cores 2 -- ./run.sh check all
```

The reader end-to-end test reads two generated layer stacks with Reader v3. It checks the
value range, that two reads are identical, and that the read isn't flat. The sheet-stage
end-to-end test reads three PHerc0211 pieces of `rows-0000-0012`. It fetches their CT into a
temporary directory, checks every output, and checks that a second read skips finished pieces.
It needs the scan's pieces, layout and axis under the outputs root, the network, and `jq`, and
it skips without them.

The sheet viewer end-to-end test serves a synthetic tile pack, checks the tile map and decoded
batches, and then drives headless Chrome over the DevTools pipe to draw, erase and restyle the
sheet. It runs the browser checks only when Chrome is at its usual macOS path or at `CHROME`.
