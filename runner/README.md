# runner

How work runs on one Mac: 16 GB, one GPU, one Neural Engine, one network link.

```text
pipeline.py           the route: sheets, pieces, sheet stages, each resumable from its records
sheet_stage.py        pieces to the website: layout, every piece read, then the scroll's page
review_page.py        a read scan published as its whole-scroll page with its card on the start page
stage_queue.py        method digests, atomic units, claims and stage locks
job_board.py          one queue for every long job: slots, memory, priorities, swap guard; state in runtime/board/
fetch_ahead.py        survey fetches split from reads
wait_for_queue.py     a gate that ends when named jobs can no longer run
memory_trace.py       true footprint by source line, C and GPU included
compiled_kernels.py   C and Objective-C compiled once per source digest into runtime/kernels/

./run.sh board status | room GB | log NAME | wait NAME | usage | forecast
./run.sh board add NAME [--memory GB] [--gpu] [--engine] [--link] [--share] [--restartable] [--after JOB] [--priority N] [--held] -- COMMAND
./run.sh board hold|release|resize|cancel|retry|remove NAME
./run.sh board change NAME [--priority N] [--cores C] [--share 0|1] [--restartable 0|1]
```

A `--share` job (sheet detection, a read's CT fetched ahead: mostly downloading, short GPU bursts) runs beside the one job holding the GPU or the link whole: one sharer on the GPU, two on the link; the Neural Engine is never shared. When a GPU job is held back only by cores or memory that lower-priority `--restartable` jobs hold, the board stops one of them (the lowest that alone makes room, the latest started first), one a minute, and it comes back on its own once there is room. A job a waiting GPU job needs within two steps (its layout, its fetch ahead) does the same, but stops only sharing jobs: a detection resumes from its bricks, a unit of pieces would be lost.

Ask `board room GB` before any test. After editing `job_board.py`, `kill $(cat runtime/board/dispatcher.lock)`.

## Running a scan

Work from the HerculeX folder. `./run.sh ink.model.weights fetch` once, for Reader v3. Then one scroll is three board jobs:

```sh
S=PHerc0826
./run.sh board add $S-sheets --gpu --link --share --memory 5 --setting HERCULEX_LIST_CT_BRICKS=1 -- \
  ./run.sh runner.pipeline run sheets $S
./run.sh board add $S-pieces --memory 5 --cores 3 --restartable --after $S-sheets -- \
  ./run.sh runner.pipeline run pieces $S
./run.sh board add $S-sheet --gpu --engine --link --memory 5 --cores 5 --after $S-pieces --setting HERCULEX_KEEP_STACKS=0 -- \
  ./run.sh runner.pipeline run sheet $S
./run.sh board log $S-sheet
```

| Stage | Work |
| --- | --- |
| `sheets SCROLL` | Detect sheet nodes in the scroll's survey scan, fetching the CT as it goes. GPU and link. |
| `pieces SCROLL` | Unwrap the nodes into pieces, join them across unit seams and fit the scan's frame. CPU. |
| `layout SCROLL [--scan ID]` | Place the pieces on one sheet. |
| `ink SCROLL [--scan ID]` | Refresh the layout and read every piece into an ink map. GPU, Neural Engine and link. |
| `sheet SCROLL [--scan ID]` | Ink, then publish the scroll's page and its card on the start page. GPU, Neural Engine and link. |
| `all SCROLL` | The three stages at once, each following the one before: about 14 GB at its peak. |

Every stage resumes from what is finished: `./run.sh board retry NAME` after a failure, and a failed read keeps the CT it fetched. `--scan ID` picks a scan other than the survey scan. `./run.sh runner.pipeline status [SCROLL]` lists what is done.

The read uses stock Reader v3; `HERCULEX_WEIGHTS=/path/to/checkpoint.pth` reads with another (its page is labelled `HERCULEX_MODEL_NAME`, default "post-trained Reader v3"). `HERCULEX_BRICKS` puts the CT cache on another drive, and `HERCULEX_KEEP_STACKS=0` skips keeping each piece's layer stack (about 20 GB per 1,000 cm² of sheet). The `sheet` stage ends in `runner.review_page SCROLL SCAN`: tiles of the scan and the ink, the viewer page, its card on the start page, and a check of tiles and marks over HTTP when the site is running. Run it alone to republish a read scan.
