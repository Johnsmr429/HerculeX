# HerculeX for agents

HerculeX unwraps a Herculaneum scroll from its CT scan and reads its ink on one Apple-silicon Mac. A person reads the
result in a local viewer while you run the pipeline and follow up on what they mark. This page is everything you need
to start; each folder's README (10-30 lines) has its files, commands and outputs.

## Ground rules

- Run everything as `./run.sh MODULE ARGS` from this folder. Long jobs (anything over a few minutes, anything on the
  GPU, Neural Engine or network) go on the job board: `./run.sh board add NAME [slots] -- COMMAND`. Before a test or
  a quick GPU check, `./run.sh board room GB` says whether it fits beside the jobs running.
- Data paths come only from `storage/data_roots.py` (`LIBRARY`, `OUTPUTS` and the names built on them); never write
  an absolute path into code. `HERCULEX_DATA`, `HERCULEX_LIBRARY`, `HERCULEX_OUTPUTS` move them.
- Stages resume from finished units. Each unit's `record.json` names the digest of the code that made it, so editing a
  stage's code makes its units out of date; `same_output_methods.txt` lists digests proven to give the same output.
- Never edit the website's pages by hand. Scroll pages are published by the pipeline; other pages reach the start page
  through `website.site_pages card`. Every card and viewer title is `scroll · ink model · surface · date`.
- Marks logs (`library/marks/`) are append-only and belong to the person; never rewrite or delete them.
- State numbers with what they were measured on, and say when a result is on data the method was tuned on.
- Before finishing a change: `./run.sh check cpu` (5 min), and `./run.sh check` when it touches the GPU or reader.
  Code style is ruff at 120 columns (`runtime/python/bin/python -m ruff check .` and `-m ruff format .`),
  one-sentence docstrings, no comments that restate the code.

## Recipes

Unwrap and publish a scroll (three board jobs; the last one puts the scroll on the start page):

```sh
S=PHerc0826
./run.sh board add $S-sheets --gpu --link --share --memory 5 --setting HERCULEX_LIST_CT_BRICKS=1 -- ./run.sh runner.pipeline run sheets $S
./run.sh board add $S-pieces --memory 5 --cores 3 --restartable --after $S-sheets -- ./run.sh runner.pipeline run pieces $S
./run.sh board add $S-sheet --gpu --engine --link --memory 5 --cores 5 --after $S-pieces --setting HERCULEX_KEEP_STACKS=0 -- ./run.sh runner.pipeline run sheet $S
```

Follow it: `./run.sh board status`, `./run.sh board log NAME`, `./run.sh runner.pipeline status $S`. After a failure, fix
the cause and `./run.sh board retry NAME`. When it is done, check the page loads and tell the person it is up.

| Task | Command |
| --- | --- |
| Republish a read scan | `./run.sh runner.review_page SCROLL SCAN` |
| Put another published page on the start page | `./run.sh website.site_pages card PAGE [SCROLL] [--surface WORD]` (`--remove` takes it off) |
| Read with another checkpoint | `HERCULEX_WEIGHTS=/path.pth HERCULEX_MODEL_NAME="name"` on the `sheet` job |
| Read apart from production, e.g. several depths | `HERCULEX_FACES=NAME HERCULEX_DEPTH_SHIFTS=-4,-2,0,2,4`, then `website.sheet_tiles` and `website.site_pages part` |
| A top-down picture of a scroll | `./run.sh website.overview PAGE OUT.jpg [--width 2400]` |
| The site | `./run.sh website.site_server`, http://localhost:8765/sheets/index.html |

## When the person has marked a scroll

Spot circles (viewer key `C`) are candidate letters. Work on them while the person keeps reading:

1. `./run.sh spots.catalogue` builds the spots table from the live marks and prints each spot's id.
2. Each spot is two board jobs, a build on the link and a read on the GPU and engine (see `spots/README.md`):
   it is narrowed to its pieces, rebuilt as continuous surfaces, read at every depth from -15 to +15 by several
   models, and given a sweep page under `/sheets/spot-candidates/`.
3. The person chooses the depth and judges the letters on those pages; record verdicts with
   `./run.sh spots.database verdict SPOT SURFACE --by NAME --readable no|traces|letters|passage`.
4. For a First Letters submission (10 legible letters in 4 cm²), package the chosen spot's surface, image, depth and
   the commands that reproduce them. Keep it plain and say what was not run.

Ink strokes (`G`) and blank strokes (the Blank tool) are hand labels; `ink/datasets/hand_mark_strokes.py` carries them into training data.

## Where things are

```text
README.md            what HerculeX is, how it works, results, setup
runner/README.md     stages, the job board, running a scan
storage/README.md    data folders and file formats
unwrapping/, sampling/, ink/, website/, spots/, calibration/, scans/   one README each
tests/README.md      the two kinds of test
```
