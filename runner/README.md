# Runner

The runner coordinates resumable pipeline stages and long-running work.

## Files

| File | Purpose |
| --- | --- |
| `pipeline.py`, `sheet_stage.py`, and `review_page.py` | Run the route from sheet detection through page publication. |
| `job_board.py` and `wait_for_queue.py` | Schedule long jobs and wait for their dependencies. |
| `stage_queue.py` | Tracks method digests, atomic outputs, unit claims, and stage locks. |
| `memory_trace.py` | Reports the source lines at which a process reaches its peak footprint. |
| `compiled_kernels.py` | Compiles C and Objective-C kernels once per source digest. |

## Commands

```sh
./run.sh runner.pipeline status [SCROLL ...]
./run.sh runner.pipeline run STAGE SCROLL [--scan ID]
./run.sh board add NAME [OPTIONS] -- COMMAND
./run.sh board status
./run.sh board room GB
./run.sh board log NAME
./run.sh board wait NAME
```

Set `STAGE` to `sheets`, `pieces`, `layout`, `ink`, or `sheet`. Use the job board for long jobs. A stage resumes from completed units, so fix a failed job and run `./run.sh board retry NAME`. Use `hold`, `release`, `cancel`, or `remove` to manage a named job. The `sheet` stage publishes the viewer page and adds its card to the start page.
