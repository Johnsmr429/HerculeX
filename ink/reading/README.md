# ink/reading

```text
piece_reading.py         ink stage: one pass per piece (stack, scan images, written face, ink map), next store fetched during the read
spot_depths.py           places circled with the viewer's Spot tool read again at nearby depths
passage_reads.py         whole pieces read again from fresh stacks by several models at depths -4..+4, both faces, beside the site's map
same_output_models.txt   one line per checkpoint: its current read digest, then earlier digests with byte-identical maps

./run.sh runner.pipeline run ink SCROLL [--scan ID]           refresh the layout and read ink
./run.sh ink.reading.piece_reading SCROLL SCAN [ORDER.json]
./run.sh ink.reading.piece_reading ahead SCROLL SCAN STORES     fetch stores while the GPU is taken
./run.sh ink.reading.spot_depths SCROLL SCAN SHEET_NAME [MARKS]
./run.sh ink.reading.passage_reads SCROLL SCAN PLACES.json PAGE_NAME WEIGHTS...
```

Samplers build stacks ahead of the reader (`HERCULEX_SAMPLERS`, default 4). One reader keeps the engine busy, and tiles are blended in order.
Stacks are kept in `stacks/ROUTE/SCROLL/SCAN/` in the outputs folder (`HERCULEX_WINDOWS` moves them).
A read kept apart from production writes to `papyrus-faces/NAME` (`HERCULEX_FACES`, default `level-0`), may average reads at layer shifts (`HERCULEX_DEPTH_SHIFTS`, e.g. `-4,-2,0,2,4`) and take some units (`HERCULEX_UNITS`); it reads the same face as production.
A fetch stops at 25 GiB free on its disc, 22 GiB once the read waits on it.
Each unit's fetched CT store is deleted after its read; `HERCULEX_KEEP_BRICKS=1` keeps them, so a read of the same units after a surface rebuild fetches only the bricks the new pieces add. A whole-scroll read moves 150-400 GiB of CT; on a home connection the download, not the GPU, sets its pace.
Use the [job board](../../runner/README.md#running-a-scan) for full runs.
