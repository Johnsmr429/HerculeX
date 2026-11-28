# runner

The `runner` package drives pipeline execution on either lane: a resource-aware job board on `apple` (macOS) and a multi-GPU/multi-core foreground scheduler on `cuda` (Linux).

## Files

| File | Role |
| --- | --- |
| `lane.c` | Selects `apple` on macOS and `cuda` on Linux; validates `HERCULEX_LANE`. |
| `sheet_stage.inc` | Lays out pieces, runs ink inference, and publishes the viewer page. |
| `job_board.c` | Job queue with GPU, Neural Engine, link, core, memory, priority, and swap guards. |
| `stage_queue.c` | Computes code digests and publishes finished units under file locks. |
| `keep_faces.c` | Preserves per-model face hard links before tile composition removes raw faces. |
| `machine_apple/probes.c` | Reads macOS CPU, GPU, memory, network, and disk counters. |
| `machine_cuda/probes.c` | Reads Linux `/proc` and `nvidia-smi` counters. |
| `events.c`, `timeline.c` | Logs job spans (`HERCULEX_EVENTS`) and visualizes slot utilization and idle time. |
| `wait_for_queue.c` | Blocks until named board jobs finish or fail. |
| `review_check.c` | Verifies published viewer pages, marks, and full-resolution tile byte ranges. |
| `place_build.c` | Installs imported builds and archives displaced builds with their viewer pages. |
| `disk.c`, `memory_trace.c`, `speed.c` | Disk reclamation audit, peak RSS profiler, and fixed-workload benchmarks. |
| `check.c`, `lint.c`, `modules.c`, `native_ports.c` | Test runner, formatter/linter driver, `./run.sh list` registry, and port launcher. |

*(Level 1.1 stage orchestration lives in `level/pipeline.c` and `level/unwrap.inc`.)*

## Commands and checks

```sh
./run.sh list                              # list all runnable modules
./run.sh setup [native|lint|spots]         # install native deps, linters, or DuckDB
./run.sh paths [data|library|outputs]      # print active resolved data directories
./run.sh check cpu                         # run CPU-only golden and e2e tests
./run.sh check all                         # include GPU and Neural Engine tests
./run.sh check lint [--fix]                # format and lint C, CUDA, Python, shell, JS/HTML/CSS
```

## Run a scroll (level 1.1)

Run `./run.sh level SCROLL[:SCAN]` from the repository root.

- **Apple lane:** Queues four dependent board jobs per scroll (`SCROLL-sheets`, `SCROLL-pieces`, `SCROLL-fetch`, `SCROLL-sheet`). `pieces` and `fetch` run concurrently after `sheets` finishes so CT downloads overlap CPU fitting.
- **CUDA lane:** Runs all named scrolls in the foreground, writing logs to `runtime/logs/unwrap/`. One scroll runs sheet detection per GPU; a CPU fit pool processes detected units largest-first (1 fit per 3 cores and 8 GB RAM, up to 12 OpenMP threads); one ink read runs per GPU by default (`--reads-per-gpu N` overrides); two CPU publishers compose pages. Failed stages retry once.
- **WSL 2:** Keep the repository inside the Linux filesystem, not `/mnt/c`.

| Stage (`./run.sh level.pipeline run STAGE SCROLL [--scan ID]`) | Action | Resources |
| --- | --- | --- |
| `sheets` | Detect sheet nodes and fetch CT bricks. | GPU, link |
| `pieces` | Solve winding levels, emit pieces, join seams, and fit the frame. | CPU |
| `fetch` | Prefetch CT bricks as units finish fitting. Stops at completion, `HERCULEX_FETCH_AHEAD_GIB` (200), or `HERCULEX_FETCH_IDLE_HOURS` (1.5). | Link |
| `layout` | Place pieces on one sheet. | CPU |
| `ink` | Refresh layout if changed and run the selected ink model. | GPU, ANE, link |
| `sheet` | Run `ink`, compose tiles, publish viewer page, and add start-page card. | GPU, ANE, link |
| `all` | Run `sheets`, `pieces`, `fetch`, and `sheet` directly without the board. | All |

Every stage resumes from completed units. Add `--band` to `sheets`, `pieces`, or `all` to process only a central band. Check progress with `./run.sh status SCROLL`, rebuild the unit index with `./run.sh level.pipeline index`, or republish an existing read with `./run.sh publish SCROLL SCAN`.

## Settings

| Variable | Default | Effect |
| --- | --- | --- |
| `HERCULEX_INK_MODEL` | `reader-v3` | Selects the ink model (`./run.sh models`); non-default models publish to `sheet-SCAN-MODEL`. |
| `HERCULEX_KEEP_STACKS` | `1` | Set to `0` to skip saving per-piece layer stacks. |
| `HERCULEX_BRICKS` | `data/library/cache/bricks/` | Moves the CT brick cache. |
| `HERCULEX_LIST_CT_BRICKS` | unset | Set to `1` on `sheets` to list published bricks first and skip empty ones. |
| `HERCULEX_SITE` | `http://127.0.0.1:8765` | Local viewer URL verified after publication. |
| `HERCULEX_BOARD` | `runtime/board/` | Directory for job board database, logs, and dispatcher lock. |
| `HERCULEX_BOARD_TICK` | `2` | Dispatcher poll interval in seconds. |
| `HERCULEX_BOARD_MARGIN_GB` | `1.0` | RAM reserved free by the job board. |
| `HERCULEX_BOARD_SWAP_GB` | `max(4.2, 25% RAM)` | Swap ceiling in GB before pausing launches > 1 GB when free RAM is tight. |
| `HERCULEX_BOARD_CORES` | Host core count | CPU core budget managed by the board. |
| `HERCULEX_UNIT_ROWS` | `12` | Brick rows per sheet-detection unit. Must stay constant for a scan once started. |

## Job board

```sh
./run.sh board add NAME [OPTIONS] -- COMMAND
./run.sh board status | log NAME [LINES] | room GB | wait NAME | forecast [NAME] | usage [HOURS]
./run.sh runner.timeline [PATTERN] [--hours H] [--json OUT]
./run.sh board hold | release | cancel | retry | remove NAME
./run.sh board resize NAME GB
./run.sh board change NAME [--priority N] [--cores N] [--share 0|1] [--restartable 0|1] [--owner NAME]
```

Key `board add` options:

- `--gpu`, `--engine`, `--link`: Claim the GPU, Neural Engine, or network link.
- `--memory GB`, `--cores N`: Reserve peak RAM (defaults to prior peak for the same job kind) and CPU cores (default `1`; measured CPU usage overrides if higher).
- `--after JOB ...`, `--priority N`, `--held`, `--owner NAME`, `--setting KEY=VALUE`: Dependencies, ordering, ownership, and saved environment.
- `--share`: Allows bursty GPU/link jobs (like `sheets`) to share slots (up to 1 sharing job on GPU, 2 on link; `--engine` is exclusive).
- `--restartable`: Allows preemption when a higher-priority GPU job needs resources or swap pressure rises; jobs exiting with code `75` requeue automatically.

Run `./run.sh board room GB` before starting manual tests or GPU checks beside active board jobs. After rebuilding `job_board.c`, let the board go idle so the next command restarts the background dispatcher.
