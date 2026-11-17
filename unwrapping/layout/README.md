# unwrapping/layout

Stage 4 places a scan's pieces and their joins on one sheet along the roll. For a scan of several units, the pipeline
uses `unwrapping.level.scroll_layout`; for a scan of one unit or one band, it uses `mosaic_layout.py`.

| File | Description |
| --- | --- |
| `mosaic_layout.py` | Joins pieces into mosaics and places the mosaics on one sheet, in the scan's `sheet/layout.json`. |
| `roll_placement.py` | Places the pieces of one height band along the roll by their winding. |
| `tifxyz_export.py` | Exports a laid-out sheet as one tifxyz grid for VC3D. |

```sh
./run.sh unwrapping.layout.mosaic_layout SCROLL SCAN [band]
./run.sh unwrapping.layout.tifxyz_export SCROLL SCAN NAME    # writes website/sheets/NAME/tifxyz/ in the outputs
```
