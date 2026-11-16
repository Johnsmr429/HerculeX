# Sheet layout

This stage places unwrapped pieces on one two-dimensional sheet.

## Files

| File | Purpose |
| --- | --- |
| `mosaic_layout.py` | Joins connected pieces into mosaics and writes their sheet placement. |
| `roll_placement.py` | Orders pieces from one height band by winding around the scroll. |
| `tifxyz_export.py` | Exports a completed layout as a tifxyz grid for VC3D. |

## Commands

```sh
./run.sh unwrapping.layout.mosaic_layout SCROLL SCAN
./run.sh unwrapping.layout.tifxyz_export SCROLL SCAN PAGE
```

The layout command writes `sheet/layout.json` under the scan's level-route output. The export command writes `tifxyz/` under the named website page.
