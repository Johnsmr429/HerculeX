# storage

The `storage` package defines where data lives and how it's packed on disk.

| File | Description |
| --- | --- |
| `data_roots.py` | Defines every data folder: the library, the outputs and the checkout. It also provides the path helpers below. |
| `brick_store.py` | Stores a scan level or a surface volume as one file of raw bricks plus an offset table. |
| `node_store.py` | Packs a chunk's sheet nodes losslessly with Zstandard. |
| `route_database.py` | Keeps an SQLite index of every stage's finished units, rebuilt from their records. |
| `tile_store.py` | Stores a sheet's image pyramid as one pack of tiles plus an index (`tiles.index`), which the browser reads by byte range. |

## Data folders

HerculeX keeps its data in two folders under one data root, `data/` in the checkout by default:

| Folder | Variable | Contents |
| --- | --- | --- |
| `data/library/` | `HERCULEX_LIBRARY` | What the pipeline reads and knows: the scan catalog, published products, model weights, datasets, controls, studies and hand marks (`marks/`). |
| `data/outputs/` | `HERCULEX_OUTPUTS` | What the pipeline makes: `unwrapping/` (sheet nodes, pieces and layouts), `stacks/` (layer stacks) and `website/`. |

`HERCULEX_DATA` moves both folders at once; `HERCULEX_LIBRARY` and `HERCULEX_OUTPUTS` move one each. HerculeX resolves
symbolic links, so `data/library` or `data/outputs` can link to another drive. Fetched CT bricks go to the library's
`cache/bricks/`, and `HERCULEX_BRICKS` moves them.

Sheet nodes are stored in `unwrapping/grown/` (`NODE_ROUTE`). Every later stage writes to `unwrapping/level/`
(`LEVEL_ROUTE`), where each scan's `nodes` folder links to its sheet nodes. The index of finished units is
`unwrapping/level_route.sqlite`; `./run.sh runner.pipeline index` rebuilds it.

## Paths in code

Don't write absolute paths into code. Build every path from the names in `data_roots.py`:

- `scan_folder(scroll, scan)` returns the folder that a scan's stages write to.
- `stage_folder(scan, stage)` returns one stage's folder in it, such as `nodes`, `faces` or `sheet`.
- `stage_entries(route, stage, pattern)` lists the matching entries of that stage in every scan.
- `ct_bricks(scroll, store)` returns the folder of a fetched CT store.
- `calibration_output(name)` returns the folder of one calibration study's outputs.
- `route_id(route)` returns the route key that dataset records and marks logs store.
