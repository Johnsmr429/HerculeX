# website

The `website` package builds the local site where you read sheets (stage 7), and stores your hand marks. Each sheet
is a tile store, `tiles.index` plus one pack of tiles, in `website/sheets/NAME/` in the outputs. The tiles hold raw
scan values, and the viewer stretches each layer between the two gray levels recorded for it.

| File | Description |
| --- | --- |
| `site_server.py` | Serves the site at `http://localhost:8765/sheets/index.html`: `/sheets/` and `/atlas/` from the outputs and `/website/` from `assets/`. It appends marks to the library's `marks/`. |
| `sheet_tiles.py` | Composes one continuous sheet from every piece's images as a tile store, with the scan's own values without loss at level 0 and previews above it. |
| `site_pages.py` | Writes the start page, a viewer for each sheet and its largest area, the window proposals, and viewers of selected units. |
| `panel_pages.py` | Writes pages of pictures place by place, such as spots or passages; each place's pictures switch in place. |
| `two_unwraps.py` | Shows two unwraps of one scan, from `unwrapping/grown/` and `unwrapping/level/`, on one page. |
| `window_proposals.py` | Proposes 2 x 2 cm windows of a sheet where the ink map is most confident. |
| `suggested_areas.py` | Lists the squares that a page suggests for a first look. |
| `mosaic_outline.py` | Draws the outline of a mosaic on its sheet's viewer, without recomposing tiles or changing marks. |
| `atlas_build.py` | Builds the letter atlas from known letters, infrared fragments, published maps and a Greek alphabet. |
| `refresh_viewers.py` | Updates the viewer controls of published pages in place, without recomposing tiles or changing marks. |
| `survey_status.py` | Reports the measured coverage of a scan as JSON. |
| `overview.py` | Draws a published page as one picture: the whole unrolled sheet, shelf under shelf. |
| `assets/` | Holds the viewer HTML, the byte-range tile reader, the WebGL decoder, and the keyboard shortcuts that every page shares. Press `?` in the viewer to list them. |

## Commands

```sh
./run.sh website.site_server                     # serve the site
./run.sh website.site_pages                      # rewrite every page and the start page
./run.sh website.site_pages card PAGE [SCROLL] [--surface WORD] [--remove]
./run.sh website.site_pages part NAME TITLE SCROLL SCAN [UNIT,UNIT]
./run.sh website.sheet_tiles SCROLL SCAN PIXEL_MM NAME [LAYERS] [largest] [--units UNIT,UNIT] [--model NAME]
./run.sh website.panel_pages NAME...
./run.sh website.two_unwraps SCROLL SCAN PIXEL_MM
./run.sh website.mosaic_outline PAGE SCROLL SCAN [RANK]
./run.sh website.survey_status SCROLL SCAN
./run.sh website.atlas_build everything          # needs the labeled cases from ink.datasets.labelled_cases
./run.sh website.overview PAGE OUT.jpg [--width 2400]
```

## The start page

A scroll's page reaches the start page automatically when the pipeline's `sheet` stage publishes it
(`runner.review_page`). To add any other published page, run `./run.sh website.site_pages card PAGE`. The command
lists the page in `website/sheets/cards.json`, makes its card picture, and rewrites the start page; `--remove` takes
the card off again. Don't edit the start page by hand.

Every card and viewer is labeled `scroll · ink model · surface · date` from the page's `layers.json`. The ink model is
`Reader v3` for the stock checkpoint, and `HERCULEX_MODEL_NAME` or the checkpoint's file name for any other.

## Hand marks

The server appends each mark as one JSON line to `marks/level_route/SCROLL/SCAN/marks.jsonl` in the library. Each line
names the piece placement it was drawn on, which the server keeps in `layouts/` beside the log, so marks stay attached
to the scan when the layout changes. Removing a mark appends a new line. The logs are append-only: never rewrite or
delete them.

## Settings

| Variable | Default | Effect |
| --- | --- | --- |
| `HERCULEX_FACES` | `level-0` | Makes `sheet_tiles` and `site_pages part` use a separate read in the scan's `faces/NAME/`. Such a sheet keeps its own placement and marks, under the key `part/NAME`. |
| `HERCULEX_COMPOSE_PROCESSES` | `4` | Sets the number of processes that compose tiles. |
