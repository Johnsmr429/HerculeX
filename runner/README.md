# runner

The `runner` package runs the pipeline on one machine of either lane. It defines the stages, makes each stage
resumable, and runs scrolls through them with the lane's unwrap: the job board on the apple lane, which shares the GPU,
the Neural Engine, the network link and memory between long jobs, and a scheduler that keeps every core and GPU busy on
the cuda lane.

| File | Description |
| --- | --- |
| `lane.py` | Sets the lane: `apple` on a Mac, `cuda` elsewhere. `HERCULEX_LANE`, when set, must name the same one. |
| `unwrap.py` | Runs `./run.sh unwrap SCROLL[:SCAN] ...` with the lane's unwrap. |
| `machine_apple/` | The apple lane's machine. `unwrap.py` queues a scroll's four board jobs, `probes.py` reads memory, CPU, GPU, network and disc use for the board (vm_stat, libproc, the IOKit registry), and `build.py` holds clang's flags. |
| `machine_cuda/` | The cuda lane's machine. `unwrap.py` runs scrolls with every core and GPU busy, `probes.py` reads the same from `/proc` and `nvidia-smi`, and `build.py` builds kernels with OpenMP. |
| `pipeline.py` | Defines the pipeline's stages and runs or resumes each one from its records. |
| `sheet_stage.py` | Takes a scan's pieces to the website: it lays them out, reads every piece, and then publishes the scroll's page. |
| `review_page.py` | Publishes a read scan as its whole-scroll page and puts its card on the start page. |
| `place_build.py` | Places scrolls built on another machine among the outputs, so the one viewer shows them; a build it replaces moves to the archive and its page stays beside the new one. |
| `stage_queue.py` | Provides what every stage shares: code digests, atomic units, unit claims and stage locks. |
| `job_board.py` | Runs one queue for every long job, with resource slots, memory limits, priorities and a swap guard. |
| `wait_for_queue.py` | Exits once none of the named board jobs can still run, however they end. |
| `disk.py` | Reports where the space on the Mac, the library and the outputs goes, and lists what can be removed without losing work: units a killed run left, CT stores already read, detection work stores, old scratch. It never deletes. |
| `speed.py` | Times the pipeline's hot paths on fixed inputs (the exact solve, the level fit on a frozen sheet control, the brick store, both fetch clients) and records them in `records/speed.jsonl`; `--compare` fails on a benchmark 15% slower than its recent best. |
| `modules.py` | Lists every module `./run.sh` can run with the first line of its docstring (`./run.sh list`). |
| `check.py` | Runs the test suite with each file in its own process (`./run.sh check`, `check cpu`, `check ports`), and reports the slowest files and their peak footprint. |
| `lint.py` | Runs every tracked file through the pinned linters of its language (`./run.sh check lint`, `--fix` to format first): ruff, clang-format, clang-tidy, Biome, shellcheck and shfmt. |
| `events.py` | Writes spans of a job's work as JSON lines to `HERCULEX_EVENTS`, which the board sets for every job, so the timeline can show what a job did when. |
| `timeline.py` | Shows where a run's time went: the GPU, Neural Engine, link, cores and discs over the run, the jobs that held them, and each slot's idle time while a job waited. |
| `memory_trace.py` | Runs a module or script and reports where its memory footprint peaks, line by line, including C and GPU memory. |
| `compiled_kernels.py` | Compiles the C and Objective-C kernels once per source digest into `runtime/kernels/`, with the lane's compiler and flags. |

## Run a scroll

Run every command from the repository root, after `./run.sh setup`. `./run.sh unwrap PHerc0826` runs a scroll on
either lane. On the apple lane it queues these four board jobs:

```sh
S=PHerc0826
./run.sh board add $S-sheets --gpu --link --share --memory 5 --setting HERCULEX_LIST_CT_BRICKS=1 -- \
  ./run.sh runner.pipeline run sheets $S
./run.sh board add $S-pieces --memory 5 --cores 3 --restartable --after $S-sheets -- \
  ./run.sh runner.pipeline run pieces $S
./run.sh board add $S-fetch --link --share --memory 2 --cores 2 --restartable --after $S-sheets -- \
  ./run.sh runner.pipeline run fetch $S
./run.sh board add $S-sheet --gpu --engine --link --memory 5 --cores 5 --after $S-pieces --setting HERCULEX_KEEP_STACKS=0 -- \
  ./run.sh runner.pipeline run sheet $S
./run.sh board log $S-sheet
```

`pieces` and `fetch` start when `sheets` finishes and run side by side: as each unit's pieces are written, `fetch`
downloads the CT their ink read needs, so the link works while the CPU unwraps. `sheet` starts when `pieces` finishes,
reads every piece and publishes the scroll's page, with its card on the start page. A unit's pieces never change once
written, and a stack reads the same planes whichever way the layout turns it, so the read finds exactly the bytes it
would have fetched itself.

On the cuda lane, `unwrap` runs the same stages for every scroll named, in the foreground. Sheet detection takes one
scroll at a time on a GPU; a pool of level fits takes each detected scroll's units, largest first; each GPU reads one
scroll's ink at a time; and two publishers compose the pages on the CPU. The reads and publishers run on the first
cores, up to half of them, and the fits on the rest, pinned with `taskset`: one fit for every 3 cores and 8 GB of
memory, each with 12 OpenMP threads. A container's CPU quota counts as its cores. A unit that the level fit refuses
stops its scroll's read, since the read takes a scan whole.

On WSL 2, keep the checkout on the Linux disk. On a Windows drive (`/mnt/c`), temporary files fail and imports are slow.

| Stage | What it does | Uses |
| --- | --- | --- |
| `sheets SCROLL` | Detects sheet nodes in the scroll's survey scan and fetches the CT as it goes. | GPU, link |
| `pieces SCROLL` | Unwraps the nodes into pieces, joins them across unit seams, and fits the scan's frame. | CPU |
| `fetch SCROLL` | Fetches each unit's CT for the ink read as soon as the unit's pieces are written, while later units are still being unwrapped. It stops once every unit is fetched, the layout is written, the stores reach `HERCULEX_FETCH_AHEAD_GIB` (200), or no unit finishes for `HERCULEX_FETCH_IDLE_HOURS` (4). | Link |
| `layout SCROLL` | Places the pieces on one sheet. | CPU |
| `ink SCROLL` | Lays the pieces out again if they changed, then reads every piece into an ink map. | GPU, Neural Engine, link |
| `sheet SCROLL` | Runs `ink`, then publishes the scroll's page and its card on the start page. | GPU, Neural Engine, link |
| `all SCROLL` | Runs `sheets`, `pieces`, `fetch` and `sheet` at once, each following the one before, without the board. | All |

Every stage resumes from its finished units. If a job fails, fix the cause and run `./run.sh board retry NAME`. Add
`--scan ID` to a single stage to use a scan other than the survey scan, or `--band` to `sheets`, `pieces` or `all` to
process only a central band of the scan. To see what is done, run `./run.sh runner.pipeline status [SCROLL]`; to
rebuild the index of finished units, run `./run.sh runner.pipeline index`.

The `sheet` stage ends with `runner.review_page SCROLL SCAN`. It composes the tiles of the scan and the ink, writes the
viewer page and its card on the start page, and, if the local site is running, checks the tiles and marks over HTTP.
To publish a read scan again, run `./run.sh runner.review_page SCROLL SCAN`.

## Settings

| Variable | Default | Effect |
| --- | --- | --- |
| `HERCULEX_INK_MODEL` | `reader-v3` | Reads and publishes with another ink model of `./run.sh ink.model.models`; its page is `sheet-SCAN-NAME`, with its own card. |
| `HERCULEX_KEEP_STACKS` | `1` | Set to `0` to skip keeping each piece's layer stack. |
| `HERCULEX_BRICKS` | `cache/bricks/` in the library | Puts the CT brick cache on another drive. |
| `HERCULEX_LIST_CT_BRICKS` | unset | Set to `1` on `sheets` to list the published bricks first and skip masked-out ones. |
| `HERCULEX_SITE` | `http://127.0.0.1:8765` | Sets the local site that the `sheet` stage checks after it publishes. |
| `HERCULEX_BOARD` | `runtime/board/` | Sets the folder for the board's database, logs and dispatcher lock. |
| `HERCULEX_BOARD_MARGIN_GB` | `1.0` | Sets the memory, in GB, that the board leaves free. |
| `HERCULEX_BOARD_CORES` | the machine's cores | Sets the number of CPU cores the board shares out. |
| `HERCULEX_CC` | `cc` | Sets the C compiler for kernels on the cuda lane. |
| `HERCULEX_CFLAGS` | `-march=native` | Sets extra C compiler flags for kernels on the cuda lane. |
| `HERCULEX_UNIT_ROWS` | `12` | Sets the brick rows of each sheet-detection unit, which the level solve takes whole; fewer need less memory. Set it before a scan's first `sheets` job: the stage refuses to add units of another size beside the ones a scan has. |

## Job board

```sh
./run.sh board add NAME [OPTIONS] -- COMMAND
./run.sh board status | log NAME [LINES] | room GB | wait NAME | forecast [NAME] | usage [HOURS]
./run.sh runner.timeline [PATTERN] [--hours H] [--json OUT]
./run.sh board hold|release|cancel|retry|remove NAME
./run.sh board resize NAME GB
./run.sh board change NAME [--priority N] [--cores N] [--share 0|1] [--restartable 0|1] [--owner NAME]
```

`status` lists the jobs and why each waiting job waits, `room GB` checks whether that much memory fits now, `wait`
blocks until a job ends, and `forecast` estimates when the queue or one job ends. Every minute the board samples the GPU, the CPU's
busy share, the network MB/s received, the MB/s on the internal disc, the library and the outputs, and the waiting
jobs for each slot. `runner.timeline 'PHerc0826-*'` draws those samples and the jobs' spans over a run, and splits
each slot's idle time in two: while a job that wanted it was ready to start, which the board's decisions cost, and
while it waited behind an earlier stage, which the pipeline's stage barriers cost. The main options of `board add` are
these:

- `--gpu`, `--engine` and `--link` claim the GPU, the Neural Engine and the network link.
- `--memory GB` is the job's peak memory. If you leave it out, the board uses the last peak of jobs of the same kind.
- `--cores N` is the number of CPU cores the job uses (default `1`). The board also measures the cores each running
  job uses and counts the larger number; `status` names a job that uses more than it declared.
- `--after JOB ...` waits until those jobs are done; `--priority N` orders waiting jobs; `--held` waits for `release`.
- `--owner NAME` names who the job is for; `usage` reports GPU and core hours by owner.
- `--setting KEY=VALUE ...` sets environment variables for the job.
- `--share` marks a job that mostly downloads and uses the GPU in short bursts, such as sheet detection. It can run
  beside the one job that holds the GPU or the link: one sharing job on the GPU and two on the link. The Neural Engine
  is never shared.
- `--restartable` lets the board stop the job and run it again later.

When a GPU job is held back only by cores or memory that lower-priority restartable jobs hold, the board stops one of
them, at most one a minute, and the stopped job runs again once there is room. A job that a waiting GPU job needs
within two steps, such as its layout, can make room the same way, but it stops only sharing jobs. When swap grows too large, the board stops the largest
restartable job of the lowest priority. A job that exits with code 75 goes back to the queue and runs again later.

On Linux the board reads process memory and groups from `/proc` and the GPU's load from `nvidia-smi`. A job's footprint
is its resident memory, and the Neural Engine slot (`--engine`) is kept so that the same commands work; it guards
nothing there.

Before you run a test or a quick GPU check by hand, run `./run.sh board room GB` to check that it fits beside the
running jobs. The first board command starts a dispatcher in the background. If you change `job_board.py`, stop the
dispatcher with `kill $(cat runtime/board/dispatcher.lock)`; the next board command starts it again with the new code.
