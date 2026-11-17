# HerculeX for agents

HerculeX unwraps Herculaneum scrolls from X-ray CT on an Apple silicon Mac. A person reads the results in a local
viewer while you run the pipeline and act on what they label. This page lists the rules and commands you need. Each
folder's README describes its files, commands and outputs.

## Rules

- Run every module as `./run.sh MODULE ARGS` from the repository root.
- Put long jobs on the job board: anything that runs for more than a few minutes or uses the GPU, the Neural Engine or
  the network. Use `./run.sh board add NAME [slots] -- COMMAND`. Before you run a test or a quick GPU check, run
  `./run.sh board room GB` to check that it fits beside the running jobs.
- Get data paths only from `storage/data_roots.py` (`LIBRARY`, `OUTPUTS` and the paths built on them). Don't write
  absolute paths into code. `HERCULEX_DATA`, `HERCULEX_LIBRARY` and `HERCULEX_OUTPUTS` move the data folders.
- Stages resume from finished units. Each unit's `record.json` names the digest of the code that made it, so when you
  edit a stage, its units become out of date. `same_output_methods.txt` lists digests that give the same output.
- Don't edit the viewer's pages by hand. The pipeline publishes scroll pages; add any other page to the start page
  with `./run.sh website.site_pages card PAGE`. Card and page titles follow `scroll · ink model · surface · date`.
- Treat the label logs in `data/library/marks/` as the person's data: they are append-only, so never rewrite or
  delete them.
- When you report a number, say what data it was measured on.
- Before you finish a change, run `./run.sh check cpu`. If the change touches the GPU or the ink reader, also run
  `./run.sh check`. Format and lint with `runtime/python/bin/python -m ruff format .` and `-m ruff check .`
  (120 columns). Write one-sentence docstrings and avoid comments that repeat the code.

## Unwrap and publish a scroll

Queue three jobs. The last one reads the ink and adds the scroll to the start page:

```sh
S=PHerc0826
./run.sh board add $S-sheets --gpu --link --share --memory 5 --setting HERCULEX_LIST_CT_BRICKS=1 -- ./run.sh runner.pipeline run sheets $S
./run.sh board add $S-pieces --memory 5 --cores 3 --restartable --after $S-sheets -- ./run.sh runner.pipeline run pieces $S
./run.sh board add $S-sheet --gpu --engine --link --memory 5 --cores 5 --after $S-pieces --setting HERCULEX_KEEP_STACKS=0 -- ./run.sh runner.pipeline run sheet $S
```

To follow the jobs, run `./run.sh board status`, `./run.sh board log NAME` and `./run.sh runner.pipeline status $S`.
If a job fails, fix the cause and run `./run.sh board retry NAME`. When the scroll is published, check that its page
loads, and tell the person that it's ready.

| Task | Command |
| --- | --- |
| Republish a scan that was read | `./run.sh runner.review_page SCROLL SCAN` |
| Add another published page to the start page | `./run.sh website.site_pages card PAGE [SCROLL] [--surface WORD]`; `--remove` takes it off |
| Read with another checkpoint of the Reader v3 network | Set `HERCULEX_WEIGHTS=/path.pth` and `HERCULEX_MODEL_NAME="name"` on the `sheet` job |
| Read separately from the published ink, for example at several depths | Set `HERCULEX_FACES=NAME` and `HERCULEX_DEPTH_SHIFTS=-4,-2,0,2,4`, then run `website.sheet_tiles` and `website.site_pages part` |
| Draw a top-down picture of a scroll | `./run.sh website.overview PAGE OUT.jpg [--width 2400]` |
| Start the viewer | `./run.sh website.site_server`, then open http://localhost:8765/sheets/index.html |

## Act on the person's labels

The viewer saves every label as one JSON line in `data/library/marks/ROUTE/SCROLL/SCAN/marks.jsonl`: `kind` is `ink`,
`blank` or `spot`, `points` are millimeters on the unrolled sheet, and `layout` names the snapshot of the piece layout
in `layouts/` beside the log. A line with `remove` withdraws an earlier label. Spot circles mark places that may hold
letters. To work on them while the person keeps reading:

1. Run `./run.sh spots.catalogue`. It maps every standing circle to its pieces, windings and bounding box in scan
   voxels, and writes the spots table to `data/library/studies/shared/spot-candidates/db/`.
2. For each spot, queue two jobs, a build on the network link and a read on the GPU and Neural Engine. The spot is
   rebuilt as a continuous surface, read at every depth from -15 to +15, and published as a page under
   `/sheets/spot-candidates/`. See `spots/README.md`.
3. The person chooses the depth and judges the letters on those pages. Record each verdict with
   `./run.sh spots.database verdict SPOT SURFACE --by NAME --readable no|traces|letters|passage`.
4. For a First Letters submission (10 legible letters in 4 cm²), package the chosen spot's surface, image and depth,
   and the commands that reproduce them. Keep the package plain, and state what you didn't run.

## Where things are

```text
README.md            what HerculeX is, how it works, results, setup, the viewer and labels
runner/README.md     the stages, the job board and running a scan
storage/README.md    data folders and file formats
spots/README.md      from circled spots to depth sweeps and verdicts
tests/README.md      the two kinds of tests
```
