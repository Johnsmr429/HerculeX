# storage

Where data lives and how it is packed.

```text
data_roots.py       every data folder (the library, the outputs, the checkout); scan_folder, ct_bricks, calibration_output
brick_store.py      a scan level or surface volume as one file of raw bricks plus an offset table
node_store.py       sheet nodes packed losslessly to about 0.43 of their size
route_database.py   SQLite index of every stage's finished units, rebuilt from their records
tile_store.py       a sheet's image pyramid as one pack of tiles plus an index, read by byte range
```

The library (`data/library`, or `HERCULEX_LIBRARY`) holds scan catalogues, published products, labels, weights, studies and live marks; temporary CT bricks use `cache/bricks/`.
The outputs (`data/outputs`, or `HERCULEX_OUTPUTS`) hold `unwrapping/`, `stacks/`, `website/`, and isolated rebuilds under `runs/`.
`HERCULEX_DATA` moves both at once; a symlinked folder (another drive) is resolved to its real place.

Use `scan_folder()` for the scan root, `stage_folder()` for its output role and `stage_entries()` to enumerate scans. Use `route_id()` for the historical keys in datasets and marks.

Stages write to the level route's folder, `unwrapping/level/`; every scan's sheet nodes stay in `unwrapping/grown/` (`NODE_ROUTE`). `HERCULEX_BRICKS` overrides the CT brick cache location.
