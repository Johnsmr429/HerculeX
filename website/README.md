# Website

This stage builds and serves the local viewer for unwrapped sheets. It stores hand marks as append-only logs in the library.

## Files

| File | Purpose |
| --- | --- |
| `site_server.py` | Serves the generated site and accepts viewer marks. |
| `sheet_tiles.py` | Composes piece images into a tiled sheet with lossless source pixels at level 0. |
| `site_pages.py` and `refresh_viewers.py` | Build the start page and sheet viewers, then refresh viewer HTML without changing tiles or marks. |
| `panel_pages.py` | Builds pages that switch among images for each reviewed place. |
| `window_proposals.py` and `suggested_areas.py` | Select promising windows and squares for review. |
| `atlas_build.py` and `survey_status.py` | Build the reference atlas and report coverage for surveyed scans. |
| `overview.py` | Renders a published page as one overview image. |
| `mosaic_outline.py` and `two_unwraps.py` | Add mosaic outlines and comparison layouts to viewer pages. |
| `assets/` | Contains the viewer templates, styles, scripts, shortcuts, and tile decoder. |

## Commands

```sh
./run.sh website.site_server
./run.sh website.sheet_tiles SCROLL SCAN PIXEL_MM NAME [LAYERS] [largest] [--units UNIT,UNIT]
./run.sh website.site_pages
./run.sh website.site_pages card PAGE [SCROLL] [--surface WORD] [--remove]
./run.sh website.site_pages part NAME TITLE SCROLL SCAN [UNIT,UNIT]
./run.sh website.overview PAGE OUT.png [--width 2400]
```

The pipeline adds a scroll page to the start page when the `sheet` stage publishes it. Add any other published page with `website.site_pages card`. Generated files live under `${HERCULEX_OUTPUTS:-data/outputs}/website/`, and marks live under `${HERCULEX_LIBRARY:-data/library}/marks/`. `HERCULEX_FACES` selects a named read and defaults to `level-0`.
