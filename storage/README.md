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
| `data/outputs/` | `HERCULEX_OUTPUTS` | What the pipeline makes: `unwrapping/` (sheet nodes, pieces and layouts, and the winding network's fields, training labels and checkpoints), `stacks/` (layer stacks), `website/`, the bench's truth and scores (`bench/`) and the CT chunk store (`cache/chunks/`). |

`HERCULEX_DATA` moves both folders at once; `HERCULEX_LIBRARY` and `HERCULEX_OUTPUTS` move one each. HerculeX resolves
symbolic links, so `data/library` or `data/outputs` can link to another drive. Fetched CT bricks go to the library's
`cache/bricks/`, and `HERCULEX_BRICKS` moves them. The chunk store of the ml module and the bench is the outputs'
`cache/chunks/`, and `HERCULEX_CHUNKS` moves it.

Sheet nodes are stored in `unwrapping/nodes/` (`NODE_ROUTE`). Every later stage writes to `unwrapping/level/`
(`LEVEL_ROUTE`), where each scan's `nodes` folder links to its sheet nodes. The index of finished units is
`unwrapping/level_route.sqlite`; `./run.sh runner.pipeline index` rebuilds it.

The ml module and the bench use these folders in the outputs:

| Folder | Name in `data_roots.py` | Contents |
| --- | --- | --- |
| `unwrapping/network/` | `ML_OUTPUTS` | The winding network's fields, in `field/` of each scan, and each scan's CT axis. |
| `unwrapping/network-training/labels/` | `WINDING_LABELS` | Training labels built from the bench's truth packages. |
| `unwrapping/network-training/cylinders/` | | The winding network's cached training and validation windows. |
| `unwrapping/network-training/models/` | `WINDING_MODELS` | Checkpoints of the winding network; `net/best.npy` is the default (`WINDING_WEIGHTS`). |
| `bench/` | `SURFACE_BENCH` | Truth packages, scores, the results database and the holdout log. |
| `cache/chunks/` | `CT_CHUNKS` | CT chunks of OME-Zarr level 0, and `arrays.sqlite`. |
| `cache/open_data/` | `OPEN_DATA_EXTRA` | Published meshes that the library's mirror holds as empty files. |

## Paths in code

Don't write absolute paths into code. Build every path from the names in `data_roots.py`:

- `scan_folder(scroll, scan)` returns the folder that a scan's stages write to.
- `stage_folder(scan, stage)` returns one stage's folder in it, such as `nodes`, `faces` or `sheet`.
- `stage_entries(route, stage, pattern)` lists the matching entries of that stage in every scan.
- `ct_bricks(scroll, store)` returns the folder of a fetched CT store.
- `calibration_output(name)` returns the folder of one calibration study's outputs.
- `route_id(route)` returns the route key that dataset records and marks logs store.
