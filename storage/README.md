# storage

Defines HerculeX's data directory layout and binary storage formats. Build all native paths through `storage/data_roots.c`; never hardcode absolute paths.

## Data folders

By default, both roots live under `data/` in the repository checkout (symlinks are resolved):

| Folder | Override | Contents |
| --- | --- | --- |
| `data/library/` | `HERCULEX_LIBRARY` | Inputs and persistent state: scan catalog, published products, model weights (`weights/`), controls, studies, trace work cache (`cache/unroll/`), CT brick cache (`cache/bricks/`), and append-only hand marks (`marks/`). |
| `data/outputs/` | `HERCULEX_OUTPUTS` | Pipeline outputs: `runs/unroll/` (trace 1.5), `unwrapping/nodes/` and `unwrapping/level/` (level 1.1), `stacks/` (layer stacks), `website/` (viewer pages and tiles), `bench/` (truth controls and scores), and `cache/chunks/` (level-0 Zarr chunk store). |

Set `HERCULEX_DATA` to move both folders together, `HERCULEX_BRICKS` to move `cache/bricks/`, and `HERCULEX_CHUNKS` to move `cache/chunks/`.

Level 1.1 indexes completed units in `unwrapping/level_route.sqlite`; rebuild it with `./run.sh level.pipeline index`.

## Files

| File | Role |
| --- | --- |
| `data_roots.c` | Resolves library, outputs, cache, and checkout roots from environment variables. |
| `brick_store.c` | Single-file raw CT brick pack + offset table. |
| `node_store.c` | Zstandard-compressed per-chunk sheet node pack. |
| `route_database.c` | SQLite index of finished level-route units, rebuilt from unit `record.json` files. |
| `tile_store.c` | Single-file tile pyramid pack + `tiles.index` for HTTP byte-range serving. |
