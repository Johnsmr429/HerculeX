# website

Stage 7: the local site for reading sheets, and the store of hand marks. A sheet is `tiles.index` and one pack in `website/sheets/NAME/`; the viewer stretches raw values between the layer's two grey levels itself.

```text
site_server.py          http://localhost:8765/sheets/index.html; /sheets/ and /atlas/ from the outputs, /website/ from assets/; marks to marks/
sheet_tiles.py          one continuous sheet from every piece's raw images, as tiles of a tile store: the scan's own values without loss at level 0, previews above it
site_pages.py           start page, a viewer per sheet and largest area, proposals; a viewer of some units
panel_pages.py          pages of pictures place by place (spots, passages, circles), each place's pictures switched in place
two_unwraps.py          both unwrapping routes of one scan on one page
window_proposals.py, suggested_areas.py   windows and squares to look at first
atlas_build.py          letter atlas
refresh_viewers.py      viewer controls of published pages updated in place
survey_status.py        coverage of every scan
overview.py             a published page as one picture: the whole unrolled sheet, shelf under shelf
assets/                 viewer HTML, tile reader by byte range, WebGL decoder, shortcuts shared by every page (? lists them)

./run.sh website.site_server
./run.sh website.sheet_tiles SCROLL SCAN VOXEL_MM NAME [LAYERS] [largest] [--units UNIT,UNIT] [--model NAME]
./run.sh website.site_pages
./run.sh website.site_pages card PAGE [SCROLL] [--surface WORD] [--remove]
./run.sh website.site_pages part NAME TITLE SCROLL SCAN [UNIT,UNIT]
./run.sh website.panel_pages NAME...
./run.sh website.two_unwraps SCROLL SCAN PIXEL_MM
./run.sh website.overview PAGE OUT.png [--width 2400]
```

A scroll page reaches the start page by itself when the pipeline publishes it (`runner.review_page`). Any other published page is put there with `site_pages card`, which lists it in `website/sheets/cards.json`, makes its card picture and rewrites the start page; never edit the start page by hand. Every card and viewer is labelled `scroll · ink model · surface · date`, from the page's `layers.json`.

`HERCULEX_FACES` names a separate read in `papyrus-faces/` for `sheet_tiles` and `site_pages part`. Such a sheet keeps its own placement and marks (key `part/NAME`).
