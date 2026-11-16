# spots

From a circle drawn on a scroll page to a judged candidate: each circle drawn with the viewer's Spot tool is
narrowed to the pieces under it, rebuilt as continuous surfaces, read at every depth from -15 to +15 by several
models, and gathered in one database, so candidates for a First Letters submission (10 readable letters in 4 cm²)
can be found and judged.

```text
sources.py     which scroll, scan, route, page and pieces each marks log belongs to
catalogue.py   every standing Spot circle located on its pieces and in the scan, as the spots table
surfaces.py    narrowing, the m7 snap (the PHerc0343 First Letters rule), the 67-layer render with d = layer - 33
sweep.py       Reader v3 (and any listed readers) at every depth, both stack orders
images.py      the sweep laid out as the page showed it, one stretch per model, z-score ensemble, a page per spot
pilot.py       one spot through build, render, read and images, each stage resumable
database.py    Parquet tables spots, surfaces, reads, verdicts under SPOT_CANDIDATES/db; the verdict command

./run.sh spots.catalogue                              the spots table from the live marks
./run.sh spots.pilot SPOT [build] [grow] [render] [read] [images]
./run.sh spots.database [TABLE]                       table sizes, or one table as JSON lines
./run.sh spots.database verdict SPOT SURFACE --by NAME --readable no|traces|letters|passage [--letters N]
        [--stack ensemble] [--side inward] [--depth D] [--reading TEXT] [--note TEXT]
```

On the board, a spot is two jobs:

```sh
./run.sh board add spot-SPOT-build --link --memory 3 -- ./run.sh spots.pilot SPOT build render
./run.sh board add spot-SPOT-read --gpu --engine --memory 5 --after spot-SPOT-build -- ./run.sh spots.pilot SPOT read images
```

Pages are at http://localhost:8765/sheets/spot-candidates/index.html: the catalogue of every circle, linked to the
sweep page of each spot that has one (arrow keys step the depth and the stack).

Every standing circle is a target; `HERCULEX_SPOTS_SINCE=YYYY-MM-DD` keeps only those drawn since that day. Circles on
a page the pipeline published are found by themselves. A page made another way (a rebuilt surface, an older route)
is listed in the library's `marks/sheets.json` as `{key, scroll, scan, route, page, pieces}`, with `pieces` relative to
the outputs folder.

## The recipe and where it comes from

The two 2026 First Letters winners both read surfaces that sat on the organisers' m7 sheet prediction: PHerc0343
(Nieuwlaar) snapped spiral-fitted windings onto m7, PHerc0306B (Warner and Bodill) grew a patch on m7 in VC3D. The
PHerc0343 code is public and fixes the rest: render 66 layers one voxel apart, d = layer - 33 with + away from the
axis, read a 17-layer window centred on every d from -15 to +15 in both stack orders ("reverse" is the reading side),
stretch each model once over all depths, never per depth, ensemble by z-scores, then choose the depth where the
letters are. Here 67 layers are rendered so d = 0 is the surface node itself.

The sweep reads with stock Reader v3, and with any other checkpoints of the same network listed in the library's
`weights/sweep-readers.json` as `{"name": "path under weights/"}`. The original surface is read by stock Reader v3
alone, as the baseline for the m7 rebuild.

A third surface, `grown`, carries the m7 surface past its edges and into its holes on the prediction (`grow` stage):
two cells a round, a new node kept only where m7 sheet lies within 6 voxels along its normal, then the winners' snap
over the whole grid, cut to the circle's box and margin.

Images stretch a read from its median to its 99.9th percentile over all depths (the PHerc0306B display), CT from its
1st to 99.5th percentile; `ensemble 5 depths` is the ensemble averaged over d-2..d+2. The page opens at the depth
where the ensemble's 97th percentile inside the circle peaks, and can mirror or turn the view.

## What a circle covers

Most circles lie over several pieces, often from different windings, placed side by side by the layout. Each piece is
rebuilt and read on its own as one continuous surface and only laid out again for the images, so letters that run
across two pieces of different windings will not join there.

Data: `SPOT_CANDIDATES` (`studies/shared/spot-candidates` in the library): `db/` the tables, `m7/` the prediction
chunks fetched, `spots/SPOT/` surfaces, CT bricks, stacks and reads of each spot.
