# viewer

Stage 7 builds the local WebGL2 scroll viewer (`website/sheets/NAME/` in outputs) and records hand marks. Each sheet is stored as a `tiles.index` + tile pack of raw scan values stretched per layer in the browser.

## Commands

```sh
./run.sh serve [PORT]                                   # serve current and archived pages (default port 8765)
./run.sh publish SCROLL SCAN [--url URL]                # compose tiles, write viewer page, add start-page card
./run.sh publish --current SCROLL SCAN                  # check publication freshness (exits 0 if current, 1 if stale)
./run.sh card PAGE [SCROLL] [--surface WORD] [--fit WORD] [--remove]
./run.sh viewer.site_pages [home]                       # rewrite all eligible sheet pages or start page only
./run.sh viewer.site_pages part NAME TITLE SCROLL SCAN [UNIT,UNIT]
./run.sh viewer.sheet_tiles SCROLL SCAN PIXEL_MM PAGE [LAYERS] [largest] \
    [--units UNIT,UNIT] [--model TITLE] [--ink-name TITLE] [--written-face] [--review-surface SURFACE]
./run.sh viewer.sheet_tiles --verify FOLDER SCAN URL
./run.sh viewer.panel_pages NAME...
./run.sh viewer.mosaic_outline PAGE SCROLL SCAN [RANK]
./run.sh viewer.survey_status SCROLL SCAN
./run.sh viewer.window_proposals SCROLL SCAN
./run.sh viewer.refresh_viewers                         # update viewer UI controls without recomposing tiles
./run.sh viewer.overview PAGE OUT.jpg [--width 2400]    # render whole sheet as one JPEG
./run.sh viewer.atlas_build everything|pages|bodies|renders|segment_maps|segment_scores|match_letters|letter_cards|letter_windows|fragments|compare|alphabet_examples
```

- `publish` accepts faces read on either lane (`apple` or `cuda`) whose digest matches current code or `ink/reading/same_output_models.txt`.
- Never edit the start page or viewer HTML by hand. Use `./run.sh card PAGE` to register non-pipeline pages in `website/sheets/cards.json` (`--remove` unregisters). Titles follow `scroll · ink model · surface · date`.
- Open `http://localhost:8765/sheets/index.html` for the start page, `/sheets/PAGE/index.html` for archived pages, or `/atlas/` for Atlas pages.

## Server protocol and hand marks

- `assets/sheet_gl.js` requests `GET /tilemap?dir=/sheets/NAME/` for stored tile bitmaps, fetches decoded 8-bit batches via `POST /tiles` (409 on stale `stamp`), and renders tiles in WebGL2 texture arrays with Canvas 2D marks on top. `window.viewer` exposes `go`, `voxel`, `fit`, and `view()` for automated tests.
- The server appends every mark as one JSON line to `data/library/marks/ROUTE/SCROLL/SCAN/marks.jsonl` (or `.../pages/PAGE/` for linked external pages), saving the referenced layout under `layouts/`. Mark logs are append-only: never rewrite or delete them.

## Settings

| Variable | Default | Effect |
| --- | --- | --- |
| `HERCULEX_FACES` | `level-0` | Reads a separate face folder (`faces/NAME/`) in `sheet_tiles` and `site_pages part`. |
| `HERCULEX_COMPOSE_PROCESSES` | `4` | Worker threads used to compose tile packs. |
| `HERCULEX_ATLAS_WORK` | `website/atlas-work/` in outputs | Scratch folder for Atlas renders. |
| `HERCULEX_ATLAS_SECONDS` | unlimited | Time budget in seconds for Atlas rendering. |

## Files

| File | Role |
| --- | --- |
| `site_server.c`, `tile_decode.c` | HTTP server (`/sheets/`, `/atlas/`, `/website/`, `/tilemap`, `/tiles`) and append-only mark writer. |
| `sheet_tiles.c`, `sheet_page.c` | Native tile pack composer, verifier, shelf layout, and thumbnails. |
| `site_pages.c`, `panel_pages.c`, `refresh_viewers.c` | Start page, sheet/part/panel HTML generators, and in-place UI refresher. |
| `window_proposals.c`, `suggested_areas.c`, `mosaic_outline.c`, `survey_status.c`, `overview.c` | High-confidence window proposals, mosaic outlines, coverage JSON, and overview JPEGs. |
| `atlas_build.c` | Letter extraction, matching, fragment alignment, and Atlas page builder. |
| `assets/` | WebGL2 viewer (`sheet_viewer.html`, `sheet_gl.js`, `mark_writes.js`, `spot_coordinates.js`, `shortcuts.js`), start page, panel template, and Atlas assets (linted by Biome). |
